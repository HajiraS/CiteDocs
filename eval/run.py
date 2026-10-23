"""Evaluation CLI.  Run with:  python -m eval.run

Measures retrieval hit rate, correct-citation rate, correct "I don't know" rate and average
latency on eval/questions.jsonl, prints them, and saves everything to eval/results.json.

Examples:
    python -m eval.run                              # current ./index, configured threshold
    python -m eval.run --thresholds 0.2,0.3,0.4     # threshold sweep (one LLM call per question)
    python -m eval.run --chunk-sizes 200,350,500    # rebuild temporary indexes per chunk size
    python -m eval.run --no-llm                     # retrieval-only, no GROQ_API_KEY needed
    python -m eval.run --check                      # validate questions.jsonl against the docs
"""

import argparse
import json
import logging
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings, get_settings
from app.ingest import build_index, load_documents
from app.llm import Generator, GroqLLMClient, LLMClient
from app.retriever import META_FILE, Embedder, Retriever, SentenceTransformerEmbedder
from eval.harness import (
    EvalQuestion,
    check_expected_sources,
    format_rate,
    load_questions,
    run_eval,
)

DEFAULT_QUESTIONS = Path("eval/questions.jsonl")
DEFAULT_OUTPUT = Path("eval/results.json")


def _float_list(text: str) -> list[float]:
    try:
        values = [float(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"not a comma-separated list of numbers: {text!r}"
        ) from exc
    if not values or any(not 0.0 <= v <= 1.0 for v in values):
        raise argparse.ArgumentTypeError("thresholds must be numbers between 0 and 1")
    return sorted(set(values))


def _int_list(text: str) -> list[int]:
    try:
        values = [int(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"not a comma-separated list of integers: {text!r}"
        ) from exc
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError("chunk sizes must be positive integers")
    return sorted(set(values))


def parse_args(argv: list[str] | None, settings: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run",
        description="Evaluate retrieval, citations and refusals on a set of questions.",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-dir", type=Path, default=settings.docs_dir)
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument(
        "--thresholds",
        type=_float_list,
        help="comma-separated similarity thresholds to compare "
        f"(default: SIMILARITY_THRESHOLD={settings.similarity_threshold})",
    )
    parser.add_argument(
        "--chunk-sizes",
        type=_int_list,
        help="comma-separated chunk sizes in tokens; builds a temporary index for each "
        "(default: use the existing index as built). Overlap stays at CHUNK_OVERLAP.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="skip the LLM: only retrieval and threshold metrics are computed",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="seconds to wait after each LLM call, to stay under free-tier rate limits",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only validate the questions file against the documents, then exit",
    )
    args = parser.parse_args(argv)

    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")
    if args.delay < 0:
        parser.error("--delay must not be negative")
    if args.chunk_sizes and min(args.chunk_sizes) <= settings.chunk_overlap:
        parser.error(
            f"--chunk-sizes must all be larger than CHUNK_OVERLAP={settings.chunk_overlap}"
        )
    return args


def _describe_questions(questions: list[EvalQuestion]) -> tuple[int, int]:
    answerable = sum(1 for q in questions if q.answerable)
    unanswerable = len(questions) - answerable
    if (answerable, unanswerable) != (15, 5):
        print(
            f"Note: {answerable} answerable and {unanswerable} unanswerable questions "
            "(the project spec calls for 15 and 5)."
        )
    return answerable, unanswerable


def run_check(args: argparse.Namespace) -> int:
    """Validate questions.jsonl against the documents on disk. Needs no model and no index."""
    questions = load_questions(args.questions)
    documents = load_documents(args.docs_dir)
    check_expected_sources(questions, {doc.source_url for doc in documents})
    answerable, unanswerable = _describe_questions(questions)
    print(
        f"OK: {len(questions)} questions ({answerable} answerable, {unanswerable} unanswerable); "
        f"every expected_source_url matches one of {len(documents)} documents in {args.docs_dir}"
    )
    return 0


def run(
    args: argparse.Namespace,
    settings: Settings,
    embedder: Embedder,
    llm_client: LLMClient | None,
) -> int:
    """Run the evaluation, print the report and write the results file. Returns an exit code."""
    questions = load_questions(args.questions)
    answerable, unanswerable = _describe_questions(questions)
    generator = None if args.no_llm else Generator(_require(llm_client))
    thresholds = args.thresholds or [settings.similarity_threshold]
    pause = (lambda: time.sleep(args.delay)) if args.delay > 0 else (lambda: None)

    runs: list[dict] = []
    if args.chunk_sizes:
        documents = load_documents(args.docs_dir)
        check_expected_sources(questions, {doc.source_url for doc in documents})
        for size in args.chunk_sizes:
            # Throwaway index per size; the real ./index is left untouched.
            with tempfile.TemporaryDirectory() as tmp:
                meta = build_index(args.docs_dir, Path(tmp), embedder, size, settings.chunk_overlap)
                retriever = Retriever.load(Path(tmp), embedder)
                results = run_eval(questions, retriever, generator, args.top_k, thresholds, pause)
            runs += [_label(r, size, settings.chunk_overlap, meta["chunks"]) for r in results]
    else:
        retriever = Retriever.load(args.index_dir, embedder)
        check_expected_sources(questions, {s.source_url for s in retriever.sources()})
        meta = json.loads((args.index_dir / META_FILE).read_text(encoding="utf-8"))
        results = run_eval(questions, retriever, generator, args.top_k, thresholds, pause)
        runs = [
            _label(r, meta["chunk_tokens"], meta["chunk_overlap"], meta["chunks"]) for r in results
        ]

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "questions_file": str(args.questions),
        "answerable_questions": answerable,
        "unanswerable_questions": unanswerable,
        "top_k": args.top_k,
        "embedding_model": embedder.model_name,
        "llm_enabled": generator is not None,
        "groq_model": settings.groq_model if generator is not None else None,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print_report(report)
    print(f"\nSaved {len(runs)} run(s) with per-question detail to {args.output}")
    return 1 if any(r["metrics"]["llm_errors"] for r in runs) else 0


def _require(client: LLMClient | None) -> LLMClient:
    if client is None:
        raise RuntimeError("an LLM client is required unless --no-llm is given")
    return client


def _label(result: dict, chunk_tokens: int, chunk_overlap: int, chunks: int) -> dict:
    return {
        "chunk_tokens": chunk_tokens,
        "chunk_overlap": chunk_overlap,
        "chunks": chunks,
        "threshold": result["threshold"],
        "metrics": result["metrics"],
        "per_question": result["per_question"],
    }


def _pct(numerator: int | None, denominator: int | None) -> str:
    if numerator is None or not denominator:
        return "n/a"
    return f"{numerator / denominator:.1%}"


def print_report(report: dict) -> None:
    runs = report["runs"]
    llm = report["llm_enabled"]
    first = runs[0]["metrics"]
    a, u = first["answerable_total"], first["unanswerable_total"]
    print(
        f"\nQuestions: {a} answerable, {u} unanswerable | top-k {report['top_k']} | "
        f"embeddings {report['embedding_model']} | "
        f"LLM {report['groq_model'] if llm else 'disabled (--no-llm)'}"
    )

    if len(runs) == 1:
        _print_single(runs[0], report["top_k"], llm)
    else:
        _print_table(runs)

    if a:
        print(
            f"\nSmall sample: one answerable question is {100 / a:.1f} percentage points, so "
            "differences of a question or two are within noise."
        )


def _print_single(run: dict, top_k: int, llm: bool) -> None:
    m = run["metrics"]
    print(
        f"Chunking {run['chunk_tokens']} tokens / {run['chunk_overlap']} overlap "
        f"({run['chunks']} chunks) | threshold {run['threshold']:.2f}\n"
    )
    print(
        f"Retrieval hit rate (expected source in top-{top_k}): "
        f"{format_rate(m['retrieval_hits'], m['answerable_total'])}"
    )
    print(
        "Answers with correct citation:                 "
        f"{format_rate(m['correct_citations'], m['answerable_evaluated'])}"
    )
    print(
        'Correct "I don\'t know" (unanswerable):         '
        f"{format_rate(m['correct_idks'], m['unanswerable_evaluated'])}"
    )
    print(f"Average latency per question:                  {m['avg_latency_ms']:.1f} ms")
    print(
        "\nThreshold effect (no LLM involved): "
        f"{m['answerable_above_threshold']}/{m['answerable_total']} answerable questions reach "
        f"the threshold; {m['unanswerable_blocked_by_threshold']}/{m['unanswerable_total']} "
        "unanswerable questions are blocked by it."
    )
    if llm:
        print(f"LLM calls: {m['llm_calls']}  |  LLM failures: {m['llm_errors']}")
    if m["llm_errors"]:
        print("WARNING: some LLM calls failed; those questions are excluded from the rates above.")

    problems = [
        q for q in run["per_question"] if q["correct"] is False or q["retrieval_hit"] is False
    ]
    if problems:
        print("\nQuestions to look at:")
        for q in problems:
            kind = "unanswerable" if q["expected_source_url"] is None else "answerable"
            print(f"  [{q['id']}] ({kind}, best score {q['best_score']:.3f}) {q['question']}")
            if q["expected_source_url"]:
                print(f"      expected {q['expected_source_url']}")
            if q["cited_urls"]:
                print(f"      cited    {', '.join(q['cited_urls'])}")


def _print_table(runs: list[dict]) -> None:
    header = (
        f"\n{'chunk':>6} {'overlap':>7} {'chunks':>6} {'thresh':>6}  {'hit rate':>8} "
        f"{'citation':>8} {'IDK':>8} {'blocked':>8} {'latency ms':>10}"
    )
    print(header)
    print("-" * (len(header) - 1))
    for run in runs:
        m = run["metrics"]
        blocked = f"{m['unanswerable_blocked_by_threshold']}/{m['unanswerable_total']}"
        print(
            f"{run['chunk_tokens']:>6} {run['chunk_overlap']:>7} {run['chunks']:>6} "
            f"{run['threshold']:>6.2f}  "
            f"{_pct(m['retrieval_hits'], m['answerable_total']):>8} "
            f"{_pct(m['correct_citations'], m['answerable_evaluated']):>8} "
            f"{_pct(m['correct_idks'], m['unanswerable_evaluated']):>8} "
            f"{blocked:>8} {m['avg_latency_ms']:>10.1f}"
        )
    print(
        "\nhit rate = expected source in top-k | citation = answered and cited the expected "
        "source | IDK = unanswerable correctly refused | blocked = unanswerable stopped by the "
        "threshold alone.\nIn a sweep, latency adds each question's measured retrieval time and "
        "its single measured LLM call (only when the question clears that row's threshold)."
    )
    if any(r["metrics"]["llm_errors"] for r in runs):
        print("WARNING: some LLM calls failed; affected questions are excluded from the rates.")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    args = parse_args(argv, settings)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.check:
            return run_check(args)
        llm_client = None
        if not args.no_llm:
            try:
                api_key, model = settings.require_llm_credentials()
            except RuntimeError as exc:
                print(f"error: {exc}. Set it in .env, or pass --no-llm for retrieval-only metrics.")
                return 2
            llm_client = GroqLLMClient(
                api_key, model, settings.llm_timeout_seconds, settings.llm_max_retries
            )
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        return run(args, settings, embedder, llm_client)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())