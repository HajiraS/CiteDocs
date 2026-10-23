"""API behaviour through FastAPI's TestClient, with a fake retriever index and a fake LLM."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import NOT_FOUND_MESSAGE, LLMUnavailableError
from app.main import create_app
from app.ratelimit import RateLimiter
from app.retriever import Retriever
from tests.corpus import (
    DHCP_ID,
    DHCP_QUESTION,
    DHCP_URL,
    DNS_URL,
    OFF_TOPIC_QUESTION,
)
from tests.fakes import FakeEmbedder, FakeLLM, answer_json, summary_json


def make_client(settings: Settings, retriever: Retriever | None, llm=None, limiter=None):
    # No `with`: the lifespan is skipped, so the injected fakes are used as they are.
    return TestClient(create_app(settings, retriever, llm, limiter))


def ask(client: TestClient, question: str):
    return client.post("/ask", json={"question": question})


# --- /health and /sources ---


def test_health_reports_the_loaded_index(settings: Settings, retriever: Retriever):
    body = make_client(settings, retriever).get("/health").json()
    assert body == {"status": "ok", "index_loaded": True, "chunks": 2}


def test_health_is_degraded_without_an_index(settings: Settings):
    response = make_client(settings, None).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "index_loaded": False, "chunks": 0}


def test_sources_lists_documents_with_chunk_counts(settings: Settings, retriever: Retriever):
    body = make_client(settings, retriever).get("/sources").json()
    assert {d["source_url"]: d["chunks"] for d in body["documents"]} == {DHCP_URL: 1, DNS_URL: 1}
    assert body["total_chunks"] == 2


def test_sources_is_503_without_an_index(settings: Settings):
    response = make_client(settings, None).get("/sources")
    assert response.status_code == 503
    assert "python -m app.ingest" in response.json()["detail"]


# --- /ask ---


def test_ask_returns_a_grounded_cited_answer(settings: Settings, retriever: Retriever):
    llm = FakeLLM(answer_json([DHCP_ID], answer="Clients renew at the half-way point."))
    response = ask(make_client(settings, retriever, llm), DHCP_QUESTION)
    body = response.json()
    assert response.status_code == 200
    assert set(body) == {"answer", "citations", "confidence", "answered"}
    assert body["answered"] is True
    assert body["answer"] == "Clients renew at the half-way point."
    assert [c["source_url"] for c in body["citations"]] == [DHCP_URL]
    assert set(body["citations"][0]) == {"title", "source_url", "snippet"}


def test_off_topic_question_is_not_answered_and_never_calls_the_llm(
    settings: Settings, retriever: Retriever
):
    llm = FakeLLM()
    response = ask(make_client(settings, retriever, llm), OFF_TOPIC_QUESTION)
    body = response.json()
    assert response.status_code == 200
    assert body["answered"] is False
    assert body["answer"] == NOT_FOUND_MESSAGE
    assert body["citations"] == []
    assert llm.calls == []


def test_unanswerable_question_needs_no_llm_configured(settings: Settings, retriever: Retriever):
    response = ask(make_client(settings, retriever, llm=None), OFF_TOPIC_QUESTION)
    assert response.status_code == 200 and response.json()["answered"] is False


def test_invalid_json_twice_gives_not_answered_not_an_error(
    settings: Settings, retriever: Retriever
):
    llm = FakeLLM("garbage", "more garbage")
    body = ask(make_client(settings, retriever, llm), DHCP_QUESTION).json()
    assert body["answered"] is False and body["citations"] == []
    assert len(llm.calls) == 2


def test_llm_outage_becomes_a_clear_503(settings: Settings, retriever: Retriever):
    llm = FakeLLM(LLMUnavailableError("The language model is temporarily unavailable."))
    response = ask(make_client(settings, retriever, llm), DHCP_QUESTION)
    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]


def test_supported_question_without_llm_credentials_is_503(
    settings: Settings, retriever: Retriever
):
    response = ask(make_client(settings, retriever, llm=None), DHCP_QUESTION)
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_ask_without_an_index_is_503(settings: Settings):
    response = ask(make_client(settings, None, FakeLLM()), DHCP_QUESTION)
    assert response.status_code == 503


# --- Input validation ---


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 501}])
def test_invalid_questions_are_rejected_before_any_work(
    settings: Settings, retriever: Retriever, payload: dict
):
    llm = FakeLLM()
    response = make_client(settings, retriever, llm).post("/ask", json=payload)
    assert response.status_code == 422
    assert llm.calls == []


def test_control_characters_are_stripped_before_reaching_the_llm(
    settings: Settings, retriever: Retriever
):
    llm = FakeLLM(answer_json([DHCP_ID]))
    ask(make_client(settings, retriever, llm), "What\x00 is the DHCP lease renew?")
    user_message = llm.calls[0][1]
    assert "Question: What is the DHCP lease renew?" in user_message
    assert "\x00" not in user_message


# --- /summarize ---


def test_summarize_returns_cited_bullets(settings: Settings, retriever: Retriever):
    bullets = ["Leases limit how long an address is held.", "Clients renew early.", "Reservations pin IPs."]
    llm = FakeLLM(summary_json(bullets, [DHCP_ID]))
    response = make_client(settings, retriever, llm).post(
        "/summarize", json={"topic": "DHCP lease renewal"}
    )
    body = response.json()
    assert response.status_code == 200
    assert body["answered"] is True and body["bullets"] == bullets
    assert [c["source_url"] for c in body["citations"]] == [DHCP_URL]


def test_summarize_off_topic_skips_the_llm(settings: Settings, retriever: Retriever):
    llm = FakeLLM()
    response = make_client(settings, retriever, llm).post(
        "/summarize", json={"topic": "cricket world cup"}
    )
    body = response.json()
    assert response.status_code == 200
    assert body["answered"] is False and body["bullets"] == []
    assert body["message"] == NOT_FOUND_MESSAGE
    assert llm.calls == []


def test_summarize_rejects_an_empty_topic(settings: Settings, retriever: Retriever):
    response = make_client(settings, retriever, FakeLLM()).post("/summarize", json={"topic": " "})
    assert response.status_code == 422


# --- Rate limiting ---


def test_rate_limit_returns_429_with_retry_after(settings: Settings, retriever: Retriever):
    client = make_client(settings, retriever, FakeLLM(), RateLimiter(limit=2))
    statuses = [ask(client, OFF_TOPIC_QUESTION).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]
    blocked = ask(client, OFF_TOPIC_QUESTION)
    assert int(blocked.headers["retry-after"]) >= 1
    assert client.get("/health").status_code == 200  # health checks are never limited


# --- Startup (lifespan) ---


def test_startup_loads_the_index_and_tolerates_missing_llm_credentials(
    settings: Settings, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("app.main.SentenceTransformerEmbedder", lambda name: embedder)
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["index_loaded"] is True
        assert ask(client, DHCP_QUESTION).status_code == 503  # no GROQ_* configured


def test_startup_without_an_index_still_serves_health(
    settings: Settings, embedder: FakeEmbedder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("app.main.SentenceTransformerEmbedder", lambda name: embedder)
    empty = settings.model_copy(update={"index_dir": tmp_path / "no-index"})
    with TestClient(create_app(empty)) as client:
        assert client.get("/health").json()["status"] == "degraded"
