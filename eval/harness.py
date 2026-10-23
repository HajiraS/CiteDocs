"""Evaluation logic: load questions, observe the system on each one, and compute metrics.

Everything here is measured from real runs of the retriever (and the LLM, unless disabled).
No metric is ever hardcoded or estimated.

Design note for sweeps: the LLM's reply to a question depends only on the retrieved chunks, not
on the threshold. So each question is retrieved once, the LLM is called at most once for it, and
every threshold is then scored from that single observation. A question is sent to the LLM only
if its best similarity reaches the *lowest* threshold being tested.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from app.llm import Generator, LLMUnavailableError, not_found_response
from app.models import AskResponse
from app.retriever import Retriever


class EvalQuestion(BaseModel):
    """One line of questions.jsonl. expected_source_url is null for unanswerable questions."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    expected_source_url: str | None = None

    @field_validator("id", "question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("expected_source_url")
    @classmethod
    def _url_or_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("use null, not an empty string, for unanswerable questions")
        return value

    @property
    def answerable(self) -> bool:
        return self.expected_source_url is not None


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read a JSON Lines file. Blank lines are skipped; errors name the offending line."""
    if not path.is_file():
        raise FileNotFoundError(f"Questions file not found: {path}")
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            question = EvalQuestion.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"{path.name} line {number}: {exc}") from exc
        if question.id in seen:
            raise ValueError(f"{path.name} line {number}: duplicate id {question.id!r}")
        seen.add(question.id)
        questions.append(question)
    if not questions:
        raise ValueError(f"{path.name} contains no questions")
    return questions


def check_expected_sources(questions: list[EvalQuestion], known_urls: set[str]) -> None:
    """Fail loudly if an expected URL is not a known document, instead of scoring a silent miss."""
    unknown = sorted(
        {q.expected_source_url for q in questions if q.expected_source_url} - known_urls
    )
    if unknown:
        raise ValueError(
            "expected_source_url not found among the documents (typo, or docs not ingested?):\n  "
            + "\n  ".join(unknown)
        )


@dataclass(frozen=True)
class Observation:
    """What the system did for one question, before any threshold is applied."""

    question: EvalQuestion
    best_score: float
    retrieved_urls: tuple[str, ...]  # source_url of each top-k chunk, best first
    retrieval_seconds: float
    response: AskResponse | None  # None if the LLM was not called or failed
    llm_called: bool
    llm_seconds: float
    error: str | None  # set when the LLM call failed (provider unavailable)


def observe(
    question: EvalQuestion,
    retriever: Retriever,
    generator: Generator | None,
    top_k: int,
    min_threshold: float,
    clock: Callable[[], float] = time.perf_counter,
) -> Observation:
    """Retrieve once and, if the best score reaches min_threshold, call the LLM once."""
    start = clock()
    result = retriever.retrieve(question.question, top_k, min_threshold)
    retrieval_seconds = clock() - start

    response: AskResponse | None = None
    error: str | None = None
    llm_called = False
    llm_seconds = 0.0
    if generator is not None and result.supported:
        llm_called = True
        start = clock()
        try:
            response = generator.answer(question.question, result)
        except LLMUnavailableError as exc:
            error = str(exc)
        llm_seconds = clock() - start

    return Observation(
        question=question,
        best_score=result.best_score,
        retrieved_urls=tuple(rc.chunk.source_url for rc in result.chunks),
        retrieval_seconds=retrieval_seconds,
        response=response,
        llm_called=llm_called,
        llm_seconds=llm_seconds,
        error=error,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score(observations: list[Observation], threshold: float, llm_enabled: bool) -> dict:
    """Apply `threshold` to the observations and compute the metrics plus per-question detail."""
    answerable_total = answerable_hits = answerable_supported = 0
    answerable_evaluated = answerable_correct = 0
    unanswerable_total = unanswerable_blocked = 0
    unanswerable_evaluated = unanswerable_correct = 0
    errors = 0
    latencies: list[float] = []
    details: list[dict] = []

    for obs in observations:
        q = obs.question
        supported = obs.best_score >= threshold
        hit = q.expected_source_url in obs.retrieved_urls if q.answerable else None

        # With the LLM disabled nothing is "answered"; a failed LLM call leaves the question
        # unscored (it is counted in `errors`, not silently treated as right or wrong).
        failed = supported and obs.error is not None
        scored = llm_enabled and not failed
        if supported and obs.response is not None:
            response = obs.response
        else:
            response = not_found_response(min(1.0, max(0.0, obs.best_score)))
        answered = response.answered if scored else None
        cited_urls = [c.source_url for c in response.citations] if scored else []

        correct: bool | None = None
        if scored:
            if q.answerable:
                correct = answered is True and q.expected_source_url in cited_urls
            else:
                correct = answered is False

        latency = obs.retrieval_seconds + (obs.llm_seconds if supported and obs.llm_called else 0.0)
        latencies.append(latency)
        errors += int(failed)

        if q.answerable:
            answerable_total += 1
            answerable_hits += int(bool(hit))
            answerable_supported += int(supported)
            if scored:
                answerable_evaluated += 1
                answerable_correct += int(bool(correct))
        else:
            unanswerable_total += 1
            unanswerable_blocked += int(not supported)
            if scored:
                unanswerable_evaluated += 1
                unanswerable_correct += int(bool(correct))

        details.append(
            {
                "id": q.id,
                "question": q.question,
                "expected_source_url": q.expected_source_url,
                "best_score": round(obs.best_score, 4),
                "above_threshold": supported,
                "retrieval_hit": hit,
                "answered": answered,
                "cited_urls": cited_urls,
                "correct": correct,
                "answer": response.answer if scored else None,
                "latency_ms": round(latency * 1000, 1),
                "error": obs.error if failed else None,
            }
        )

    return {
        "threshold": threshold,
        "metrics": {
            "retrieval_hit_rate": _rate(answerable_hits, answerable_total),
            "retrieval_hits": answerable_hits,
            "correct_citation_rate": (
                _rate(answerable_correct, answerable_evaluated) if llm_enabled else None
            ),
            "correct_citations": answerable_correct if llm_enabled else None,
            "correct_idk_rate": (
                _rate(unanswerable_correct, unanswerable_evaluated) if llm_enabled else None
            ),
            "correct_idks": unanswerable_correct if llm_enabled else None,
            "answerable_total": answerable_total,
            "answerable_evaluated": answerable_evaluated if llm_enabled else None,
            "unanswerable_total": unanswerable_total,
            "unanswerable_evaluated": unanswerable_evaluated if llm_enabled else None,
            # Diagnostics that need no LLM: how the threshold alone splits the two groups.
            "answerable_above_threshold": answerable_supported,
            "unanswerable_blocked_by_threshold": unanswerable_blocked,
            "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 1),
            "llm_calls": sum(1 for o in observations if o.llm_called and o.best_score >= threshold),
            "llm_errors": errors,
        },
        "per_question": details,
    }


def run_eval(
    questions: list[EvalQuestion],
    retriever: Retriever,
    generator: Generator | None,
    top_k: int,
    thresholds: list[float],
    pause: Callable[[], None] = lambda: None,
) -> list[dict]:
    """Observe every question once, then score each threshold. Returns one dict per threshold.

    `pause` runs after each LLM call (outside the timed section) to respect provider rate limits.
    """
    # Warm-up: the first embedding call is slower (lazy init); keep it out of the latency numbers.
    retriever.search("warm up", 1)
    min_threshold = min(thresholds)
    observations: list[Observation] = []
    for question in questions:
        obs = observe(question, retriever, generator, top_k, min_threshold)
        observations.append(obs)
        if obs.llm_called:
            pause()
    return [score(observations, t, generator is not None) for t in thresholds]


def format_rate(numerator: int | None, denominator: int | None) -> str:
    """'14/15 = 93.3%' or 'n/a'."""
    if numerator is None or not denominator:
        return "n/a"
    return f"{numerator}/{denominator} = {numerator / denominator:.1%}"