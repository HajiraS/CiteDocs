"""The evaluation harness: question loading, metrics, sweeps and the CLI, all offline."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import Generator, LLMUnavailableError
from app.main import create_app
from app.ratelimit import RateLimiter
from app.retriever import META_FILE, Retriever
from eval.harness import (
    EvalQuestion,
    check_expected_sources,
    format_rate,
    load_questions,
    run_eval,
)
from eval.run import main, parse_args, run, run_check
from tests.corpus import (
    DHCP_ID,
    DHCP_QUESTION,
    DHCP_URL,
    DNS_ID,
    DNS_QUESTION,
    DNS_URL,
    OFF_TOPIC_QUESTION,
)
from tests.fakes import FakeEmbedder, FakeLLM, answer_json

# Shares words with the DHCP doc, so it clears the threshold even though no doc answers it.
HARD_UNANSWERABLE = "Who invented DHCP leases?"


def q(question_id: str, text: str, url: str | None = None) -> EvalQuestion:
    return EvalQuestion(id=question_id, question=text, expected_source_url=url)


def standard_questions() -> list[EvalQuestion]:
    return [
        q("a1", DHCP_QUESTION, DHCP_URL),
        q("a2", DNS_QUESTION, DNS_URL),
        q("u1", OFF_TOPIC_QUESTION),
    ]


def evaluate(retriever: Retriever, llm: FakeLLM | None, questions, thresholds=(0.1,), **kwargs):
    generator = Generator(llm) if llm is not None else None
    return run_eval(questions, retriever, generator, 2, list(thresholds), **kwargs)


# --- question file ---


def write_lines(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_questions_reads_answerable_and_unanswerable(tmp_path: Path):
    path = write_lines(
        tmp_path / "q.jsonl",
        json.dumps({"id": "a1", "question": " What? ", "expected_source_url": DHCP_URL}),
        "",
        json.dumps({"id": "u1", "question": "Why?", "expected_source_url": None}),
        json.dumps({"id": "u2", "question": "Where?"}),
    )
    questions = load_questions(path)
    assert [x.answerable for x in questions] == [True, False, False]
    assert questions[0].question == "What?"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "line 1"),
        (json.dumps({"id": "a", "question": "  "}), "line 1"),
        (json.dumps({"id": "a", "question": "Q", "expected_source_url": ""}), "use null"),
        (json.dumps({"id": "a", "question": "Q", "extra": 1}), "line 1"),
    ],
)
def test_load_questions_reports_bad_lines(tmp_path: Path, line: str, message: str):
    with pytest.raises(ValueError, match=message):
        load_questions(write_lines(tmp_path / "q.jsonl", line))


def test_load_questions_rejects_duplicate_ids_empty_and_missing_files(tmp_path: Path):
    one = json.dumps({"id": "a", "question": "Q"})
    with pytest.raises(ValueError, match="duplicate id"):
        load_questions(write_lines(tmp_path / "dup.jsonl", one, one))
    with pytest.raises(ValueError, match="no questions"):
        load_questions(write_lines(tmp_path / "empty.jsonl", ""))
    with pytest.raises(FileNotFoundError):
        load_questions(tmp_path / "missing.jsonl")


def test_check_expected_sources_flags_unknown_urls():
    questions = [q("a", "Q", DHCP_URL), q("b", "Q", "https://typo.example/x"), q("u", "Q")]
    with pytest.raises(ValueError, match="typo.example"):
        check_expected_sources(questions, {DHCP_URL, DNS_URL})
    check_expected_sources(questions[:1] + questions[2:], {DHCP_URL, DNS_URL})


def test_format_rate():
    assert format_rate(14, 15) == "14/15 = 93.3%"
    assert format_rate(0, 5) == "0/5 = 0.0%"
    assert format_rate(None, 5) == "n/a"
    assert format_rate(0, 0) == "n/a"


# --- metrics ---


def test_all_correct_run_and_unanswerable_never_reaches_the_llm(retriever: Retriever):
    llm = FakeLLM(answer_json([DHCP_ID]), answer_json([DNS_ID]))
    (result,) = evaluate(retriever, llm, standard_questions())
    m = result["metrics"]
    assert (m["retrieval_hits"], m["answerable_total"]) == (2, 2)
    assert (m["correct_citations"], m["answerable_evaluated"]) == (2, 2)
    assert (m["correct_idks"], m["unanswerable_evaluated"]) == (1, 1)
    assert m["unanswerable_blocked_by_threshold"] == 1
    assert m["llm_calls"] == 2 and len(llm.calls) == 2  # the off-topic question was not sent
    assert m["llm_errors"] == 0 and m["avg_latency_ms"] >= 0


def test_per_question_detail_keeps_answer_citations_and_scores(retriever: Retriever):
    llm = FakeLLM(answer_json([DHCP_ID], answer="Renew at half the lease time."))
    (result,) = evaluate(retriever, llm, [q("a1", DHCP_QUESTION, DHCP_URL)])
    (detail,) = result["per_question"]
    assert detail["answer"] == "Renew at half the lease time."
    assert detail["cited_urls"] == [DHCP_URL]
    assert detail["retrieval_hit"] is True and detail["correct"] is True
    assert detail["above_threshold"] is True and detail["best_score"] > 0.1


def test_citing_the_wrong_document_is_a_hit_but_not_a_correct_citation(retriever: Retriever):
    llm = FakeLLM(answer_json([DNS_ID]))  # DNS chunk is retrieved (k=2) but is the wrong source
    (result,) = evaluate(retriever, llm, [q("a1", DHCP_QUESTION, DHCP_URL)])
    m = result["metrics"]
    assert m["retrieval_hits"] == 1 and m["correct_citations"] == 0


def test_declining_to_answer_an_answerable_question_is_not_a_correct_citation(
    retriever: Retriever,
):
    llm = FakeLLM(answer_json([], answer="", answerable=False))
    (result,) = evaluate(retriever, llm, [q("a1", DHCP_QUESTION, DHCP_URL)])
    assert result["metrics"]["correct_citations"] == 0
    assert result["per_question"][0]["answered"] is False


def test_unanswerable_question_that_passes_the_threshold_depends_on_the_llm(
    retriever: Retriever,
):
    question = [q("u1", HARD_UNANSWERABLE)]
    declines = FakeLLM(answer_json([], answer="", answerable=False))
    (good,) = evaluate(retriever, declines, question)
    assert good["metrics"]["unanswerable_blocked_by_threshold"] == 0
    assert good["metrics"]["correct_idks"] == 1

    bluffs = FakeLLM(answer_json([DHCP_ID]))
    (bad,) = evaluate(retriever, bluffs, question)
    assert bad["metrics"]["correct_idks"] == 0
    assert bad["per_question"][0]["correct"] is False


def test_threshold_sweep_makes_one_llm_call_per_question(retriever: Retriever):
    llm = FakeLLM(answer_json([DHCP_ID]), answer_json([DNS_ID]))
    low, high = evaluate(retriever, llm, standard_questions(), thresholds=(0.1, 0.99))
    assert len(llm.calls) == 2  # not 2 per threshold
    assert low["threshold"] == 0.1 and high["threshold"] == 0.99
    assert (low["metrics"]["llm_calls"], high["metrics"]["llm_calls"]) == (2, 0)
    # A threshold that blocks everything keeps retrieval quality but loses every answer.
    assert high["metrics"]["retrieval_hits"] == 2
    assert high["metrics"]["correct_citations"] == 0
    assert high["metrics"]["correct_idks"] == 1


def test_llm_failure_is_counted_and_excluded_instead_of_scored(retriever: Retriever):
    llm = FakeLLM(LLMUnavailableError("provider down"))
    (result,) = evaluate(retriever, llm, [q("a1", DHCP_QUESTION, DHCP_URL)])
    m = result["metrics"]
    assert m["llm_errors"] == 1
    assert m["answerable_evaluated"] == 0 and m["correct_citation_rate"] is None
    detail = result["per_question"][0]
    assert detail["error"] == "provider down" and detail["correct"] is None


def test_without_an_llm_only_retrieval_and_threshold_metrics_exist(retriever: Retriever):
    (result,) = evaluate(retriever, None, standard_questions())
    m = result["metrics"]
    assert m["retrieval_hits"] == 2
    assert m["correct_citation_rate"] is None and m["correct_idk_rate"] is None
    assert m["answerable_above_threshold"] == 2 and m["unanswerable_blocked_by_threshold"] == 1
    assert m["llm_calls"] == 0
    assert all(d["answered"] is None for d in result["per_question"])


def test_pause_runs_once_per_llm_call_only(retriever: Retriever):
    pauses: list[int] = []
    llm = FakeLLM(answer_json([DHCP_ID]), answer_json([DNS_ID]))
    evaluate(retriever, llm, standard_questions(), pause=lambda: pauses.append(1))
    assert len(pauses) == 2


# --- the eval must behave exactly like the API ---


@pytest.mark.parametrize(
    ("question", "url", "reply"),
    [
        (DHCP_QUESTION, DHCP_URL, answer_json([DHCP_ID])),
        (OFF_TOPIC_QUESTION, None, None),
        (HARD_UNANSWERABLE, None, answer_json([], answer="", answerable=False)),
    ],
)
def test_eval_matches_the_api_response(
    settings: Settings, retriever: Retriever, question: str, url: str | None, reply: str | None
):
    replies = [reply] if reply else []
    api_llm = FakeLLM(*replies)
    client = TestClient(create_app(settings, retriever, api_llm, RateLimiter(100)))
    api = client.post("/ask", json={"question": question}).json()

    eval_llm = FakeLLM(*replies)
    (result,) = run_eval(
        [q("x", question, url)],
        retriever,
        Generator(eval_llm),
        settings.top_k,
        [settings.similarity_threshold],
    )
    detail = result["per_question"][0]
    assert detail["answered"] == api["answered"]
    assert detail["answer"] == api["answer"]
    assert detail["cited_urls"] == [c["source_url"] for c in api["citations"]]
    assert len(eval_llm.calls) == len(api_llm.calls)


# --- CLI ---


def cli_args(settings: Settings, tmp_path: Path, *extra: str):
    return parse_args(
        [
            "--questions", str(tmp_path / "q.jsonl"),
            "--output", str(tmp_path / "out" / "results.json"),
            "--docs-dir", str(settings.docs_dir),
            "--index-dir", str(settings.index_dir),
            *extra,
        ],
        settings,
    )  # fmt: skip


def write_standard_questions(tmp_path: Path) -> None:
    lines = [
        {"id": x.id, "question": x.question, "expected_source_url": x.expected_source_url}
        for x in standard_questions()
    ]
    write_lines(tmp_path / "q.jsonl", *(json.dumps(line) for line in lines))


def test_run_prints_metrics_and_saves_results(
    settings: Settings, embedder: FakeEmbedder, tmp_path: Path, capsys: pytest.CaptureFixture
):
    write_standard_questions(tmp_path)
    llm = FakeLLM(answer_json([DHCP_ID]), answer_json([DNS_ID]))
    code = run(cli_args(settings, tmp_path), settings, embedder, llm)
    assert code == 0

    printed = capsys.readouterr().out
    assert "Retrieval hit rate (expected source in top-2): 2/2 = 100.0%" in printed
    assert "Answers with correct citation:" in printed and "2/2 = 100.0%" in printed
    assert 'Correct "I don\'t know" (unanswerable):' in printed and "1/1 = 100.0%" in printed
    assert "Average latency per question:" in printed

    saved = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert saved["llm_enabled"] is True and saved["embedding_model"] == "fake-bow"
    assert saved["answerable_questions"] == 2 and saved["unanswerable_questions"] == 1
    (saved_run,) = saved["runs"]
    assert saved_run["chunk_tokens"] == 200 and saved_run["chunk_overlap"] == 20
    assert saved_run["metrics"]["correct_citations"] == 2
    assert len(saved_run["per_question"]) == 3


def test_run_lists_problem_questions_and_fails_on_llm_errors(
    settings: Settings, embedder: FakeEmbedder, tmp_path: Path, capsys: pytest.CaptureFixture
):
    write_standard_questions(tmp_path)
    llm = FakeLLM(answer_json([DNS_ID]), LLMUnavailableError("down"))  # wrong cite, then failure
    code = run(cli_args(settings, tmp_path), settings, embedder, llm)
    printed = capsys.readouterr().out
    assert code == 1
    assert "Questions to look at:" in printed and "[a1]" in printed
    assert f"expected {DHCP_URL}" in printed and f"cited    {DNS_URL}" in printed
    assert "WARNING: some LLM calls failed" in printed


def test_run_without_llm_is_retrieval_only(
    settings: Settings, embedder: FakeEmbedder, tmp_path: Path, capsys: pytest.CaptureFixture
):
    write_standard_questions(tmp_path)
    assert run(cli_args(settings, tmp_path, "--no-llm"), settings, embedder, None) == 0
    printed = capsys.readouterr().out
    assert "disabled (--no-llm)" in printed
    assert "Answers with correct citation:                 n/a" in printed
    saved = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert saved["llm_enabled"] is False and saved["groq_model"] is None


def test_run_requires_an_llm_client_unless_no_llm(
    settings: Settings, embedder: FakeEmbedder, tmp_path: Path
):
    write_standard_questions(tmp_path)
    with pytest.raises(RuntimeError, match="--no-llm"):
        run(cli_args(settings, tmp_path), settings, embedder, None)


def test_chunk_size_sweep_uses_temporary_indexes_and_leaves_the_real_one(
    settings: Settings, embedder: FakeEmbedder, tmp_path: Path, capsys: pytest.CaptureFixture
):
    write_standard_questions(tmp_path)
    meta_path = settings.index_dir / META_FILE
    before = meta_path.read_text(encoding="utf-8")
    args = cli_args(
        settings, tmp_path, "--no-llm", "--chunk-sizes", "30,200", "--thresholds", "0.05,0.5"
    )
    assert run(args, settings, embedder, None) == 0

    saved = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    runs = saved["runs"]
    assert [(r["chunk_tokens"], r["threshold"]) for r in runs] == [
        (30, 0.05), (30, 0.5), (200, 0.05), (200, 0.5),
    ]  # fmt: skip
    assert runs[0]["chunks"] > runs[2]["chunks"]  # smaller chunks, more of them
    assert meta_path.read_text(encoding="utf-8") == before
    assert "hit rate" in capsys.readouterr().out


def test_run_rejects_questions_whose_expected_source_is_not_indexed(
    settings: Settings, embedder: FakeEmbedder, tmp_path: Path
):
    line = {"id": "a1", "question": DHCP_QUESTION, "expected_source_url": "https://nope.example/x"}
    write_lines(tmp_path / "q.jsonl", json.dumps(line))
    with pytest.raises(ValueError, match="nope.example"):
        run(cli_args(settings, tmp_path, "--no-llm"), settings, embedder, None)


def test_run_check_validates_against_the_documents(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture
):
    write_standard_questions(tmp_path)
    assert run_check(cli_args(settings, tmp_path, "--check")) == 0
    printed = capsys.readouterr().out
    assert "OK: 3 questions (2 answerable, 1 unanswerable)" in printed
    assert "project spec calls for 15 and 5" in printed

    line = {"id": "a1", "question": "Q", "expected_source_url": "https://nope.example/x"}
    write_lines(tmp_path / "q.jsonl", json.dumps(line))
    with pytest.raises(ValueError, match="nope.example"):
        run_check(cli_args(settings, tmp_path, "--check"))


@pytest.mark.parametrize(
    "bad",
    [
        ["--thresholds", "0.2,1.5"],
        ["--thresholds", "abc"],
        ["--chunk-sizes", "20"],  # not larger than the overlap (20)
        ["--chunk-sizes", "x"],
        ["--top-k", "0"],
        ["--delay", "-1"],
    ],
)
def test_parse_args_rejects_bad_values(settings: Settings, bad: list[str]):
    with pytest.raises(SystemExit):
        parse_args(bad, settings)


def test_parse_args_sorts_and_deduplicates_sweeps(settings: Settings):
    args = parse_args(["--thresholds", "0.4,0.2,0.4", "--chunk-sizes", "300,100"], settings)
    assert args.thresholds == [0.2, 0.4] and args.chunk_sizes == [100, 300]


def test_main_check_succeeds_and_reports_errors(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CHUNK_OVERLAP", "20")
    write_standard_questions(tmp_path)
    ok = main(
        ["--check", "--questions", str(tmp_path / "q.jsonl"), "--docs-dir", str(settings.docs_dir)]
    )
    assert ok == 0
    missing = main(["--check", "--questions", str(tmp_path / "nope.jsonl")])
    assert missing == 2


def test_main_without_credentials_points_to_no_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    monkeypatch.setenv("GROQ_API_KEY", "")
    write_lines(tmp_path / "q.jsonl", json.dumps({"id": "u", "question": "Q"}))
    assert main(["--questions", str(tmp_path / "q.jsonl")]) == 2
    assert "--no-llm" in capsys.readouterr().out