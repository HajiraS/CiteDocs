"""Prompting, JSON validation, citation checks, the retry-once rule and Groq backoff."""

from types import SimpleNamespace

import groq
import httpx
import pytest

from app.llm import (
    ANSWER_SYSTEM,
    NOT_FOUND_MESSAGE,
    Generator,
    GroqLLMClient,
    LLMUnavailableError,
)
from app.retriever import RetrievalResult, Retriever
from tests.corpus import DHCP_ID, DHCP_QUESTION, DHCP_URL, DNS_ID
from tests.fakes import FakeLLM, answer_json, summary_json

_REQUEST = httpx.Request("POST", "https://api.groq.com/test")


@pytest.fixture
def result(retriever: Retriever) -> RetrievalResult:
    return retriever.retrieve(DHCP_QUESTION, k=2, threshold=0.1)


# --- Answers and citations ---


def test_prompt_restricts_to_context_and_lists_chunk_ids(result: RetrievalResult):
    llm = FakeLLM(answer_json([DHCP_ID]))
    Generator(llm).answer(DHCP_QUESTION, result)
    system, user = llm.calls[0]
    assert system == ANSWER_SYSTEM and "ONLY" in system and "used_chunk_ids" in system
    assert f'<chunk id="{DHCP_ID}">' in user and f'<chunk id="{DNS_ID}">' in user
    assert DHCP_QUESTION in user


def test_valid_answer_returns_citations_for_the_used_chunks(result: RetrievalResult):
    llm = FakeLLM(answer_json([DHCP_ID]))
    response = Generator(llm).answer(DHCP_QUESTION, result)
    assert response.answered is True
    assert [c.source_url for c in response.citations] == [DHCP_URL]
    assert response.citations[0].title == "DHCP Leases Overview"
    assert len(response.citations[0].snippet) <= 301
    assert 0 < response.confidence <= 1
    assert len(llm.calls) == 1


def test_json_wrapped_in_a_code_fence_is_accepted(result: RetrievalResult):
    llm = FakeLLM(f"```json\n{answer_json([DHCP_ID])}\n```")
    assert Generator(llm).answer(DHCP_QUESTION, result).answered is True


def test_invented_chunk_ids_are_dropped_but_real_ones_kept(result: RetrievalResult):
    llm = FakeLLM(answer_json([DHCP_ID, "fake-999"]))
    response = Generator(llm).answer(DHCP_QUESTION, result)
    assert response.answered is True
    assert [c.source_url for c in response.citations] == [DHCP_URL]


def test_answer_citing_only_unknown_chunks_is_rejected_after_one_retry(result: RetrievalResult):
    llm = FakeLLM(answer_json(["fake-999"]), answer_json(["fake-999"]))
    response = Generator(llm).answer(DHCP_QUESTION, result)
    assert response.answered is False
    assert response.answer == NOT_FOUND_MESSAGE
    assert response.citations == []
    assert len(llm.calls) == 2


def test_model_saying_not_answerable_is_not_retried(result: RetrievalResult):
    llm = FakeLLM(answer_json([], answer="", answerable=False))
    response = Generator(llm).answer(DHCP_QUESTION, result)
    assert response.answered is False and response.citations == []
    assert len(llm.calls) == 1


# --- Invalid JSON: retry once, then fall back ---


def test_invalid_json_is_retried_once_with_a_correction(result: RetrievalResult):
    llm = FakeLLM("this is not json", answer_json([DHCP_ID]))
    response = Generator(llm).answer(DHCP_QUESTION, result)
    assert response.answered is True
    assert len(llm.calls) == 2
    assert "previous reply was not valid" not in llm.calls[0][1]
    assert "previous reply was not valid" in llm.calls[1][1]


def test_invalid_json_twice_falls_back_to_not_answered(result: RetrievalResult):
    llm = FakeLLM("nope", "still nope")
    response = Generator(llm).answer(DHCP_QUESTION, result)
    assert response.answered is False
    assert response.answer == NOT_FOUND_MESSAGE
    assert len(llm.calls) == 2


def test_valid_json_with_the_wrong_shape_counts_as_invalid(result: RetrievalResult):
    llm = FakeLLM('{"answer": "missing the other fields"}', answer_json([DHCP_ID]))
    assert Generator(llm).answer(DHCP_QUESTION, result).answered is True
    assert len(llm.calls) == 2


def test_empty_answer_counts_as_invalid(result: RetrievalResult):
    llm = FakeLLM(answer_json([DHCP_ID], answer="  "), answer_json([DHCP_ID]))
    assert Generator(llm).answer(DHCP_QUESTION, result).answered is True
    assert len(llm.calls) == 2


# --- Summaries ---

_BULLETS = ["First point.", "Second point.", "Third point."]


def test_valid_summary_has_bullets_and_citations(result: RetrievalResult):
    llm = FakeLLM(summary_json(_BULLETS, [DHCP_ID]))
    response = Generator(llm).summarize("DHCP leases", result)
    assert response.answered is True and response.bullets == _BULLETS
    assert [c.source_url for c in response.citations] == [DHCP_URL]


@pytest.mark.parametrize("bullets", [["Only one.", "Only two."], [f"Point {i}." for i in range(6)]])
def test_summary_with_wrong_bullet_count_is_rejected(result: RetrievalResult, bullets: list[str]):
    llm = FakeLLM(summary_json(bullets, [DHCP_ID]), summary_json(bullets, [DHCP_ID]))
    response = Generator(llm).summarize("DHCP leases", result)
    assert response.answered is False and response.bullets == []
    assert response.message == NOT_FOUND_MESSAGE
    assert len(llm.calls) == 2


# --- Groq client: backoff and error mapping ---


class _Completions:
    """Stands in for client.chat.completions; plays back scripted outcomes."""

    def __init__(self, outcomes: list[str | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        message = SimpleNamespace(content=outcome)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _groq_client(outcomes: list[str | Exception], max_retries: int = 3):
    sleeps: list[float] = []
    client = GroqLLMClient("test-key", "test-model", 1.0, max_retries, sleep=sleeps.append)
    completions = _Completions(outcomes)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions, sleeps


def test_transient_errors_are_retried_with_backoff_then_succeed():
    error = groq.APIConnectionError(request=_REQUEST)
    client, completions, sleeps = _groq_client([error, error, "the reply"])
    assert client.complete("sys", "user") == "the reply"
    assert completions.calls == 3
    assert len(sleeps) == 2 and all(0 < s <= 10 for s in sleeps)


def test_gives_up_with_a_clear_error_after_the_retry_budget():
    error = groq.APIConnectionError(request=_REQUEST)
    client, completions, sleeps = _groq_client([error] * 3, max_retries=2)
    with pytest.raises(LLMUnavailableError, match="temporarily unavailable"):
        client.complete("sys", "user")
    assert completions.calls == 3 and len(sleeps) == 2


def test_rate_limit_honours_retry_after_header():
    response = httpx.Response(429, headers={"retry-after": "3"}, request=_REQUEST)
    limited = groq.RateLimitError("slow down", response=response, body=None)
    client, _, sleeps = _groq_client([limited, "ok"])
    assert client.complete("sys", "user") == "ok"
    assert sleeps == [3.0]


def test_client_errors_are_not_retried():
    response = httpx.Response(400, request=_REQUEST)
    bad_request = groq.BadRequestError("bad model", response=response, body=None)
    client, completions, sleeps = _groq_client([bad_request])
    with pytest.raises(LLMUnavailableError, match="HTTP 400"):
        client.complete("sys", "user")
    assert completions.calls == 1 and sleeps == []
