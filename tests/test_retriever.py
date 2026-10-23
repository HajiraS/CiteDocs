"""Similarity search, the threshold decision, source listing and index loading."""

from pathlib import Path

import pytest

from app.ingest import build_index
from app.retriever import Retriever
from tests.corpus import (
    DHCP_ID,
    DHCP_QUESTION,
    DNS_ID,
    DNS_QUESTION,
    OFF_TOPIC_QUESTION,
)
from tests.fakes import FakeEmbedder


def test_on_topic_questions_find_the_right_document(retriever: Retriever):
    dhcp = retriever.retrieve(DHCP_QUESTION, k=2, threshold=0.1)
    dns = retriever.retrieve(DNS_QUESTION, k=2, threshold=0.1)
    assert dhcp.supported and dhcp.chunks[0].chunk.chunk_id == DHCP_ID
    assert dns.supported and dns.chunks[0].chunk.chunk_id == DNS_ID
    assert 0.1 < dhcp.confidence <= 1.0


def test_off_topic_question_falls_below_the_threshold(retriever: Retriever):
    result = retriever.retrieve(OFF_TOPIC_QUESTION, k=2, threshold=0.1)
    assert result.best_score == pytest.approx(0.0)
    assert result.supported is False
    assert result.confidence == 0.0


def test_threshold_is_the_deciding_knob(retriever: Retriever):
    assert retriever.retrieve(DHCP_QUESTION, k=2, threshold=0.1).supported
    assert not retriever.retrieve(DHCP_QUESTION, k=2, threshold=0.99).supported


def test_search_respects_k_and_caps_at_index_size(retriever: Retriever):
    assert len(retriever.search(DHCP_QUESTION, k=1)) == 1
    assert len(retriever.search(DHCP_QUESTION, k=10)) == 2


def test_sources_count_chunks_per_document(
    docs_dir: Path, tmp_path: Path, embedder: FakeEmbedder
):
    out = tmp_path / "small"
    build_index(docs_dir, out, embedder, chunk_tokens=10, overlap=2)
    small = Retriever.load(out, embedder)
    sources = small.sources()
    assert len(sources) == 2
    assert all(s.chunks > 1 for s in sources)
    assert sum(s.chunks for s in sources) == small.num_chunks


def test_load_fails_clearly_when_no_index_exists(tmp_path: Path, embedder: FakeEmbedder):
    with pytest.raises(FileNotFoundError, match="python -m app.ingest"):
        Retriever.load(tmp_path / "missing", embedder)


def test_load_refuses_an_index_built_with_another_model(index_dir: Path, embedder: FakeEmbedder):
    embedder.model_name = "some-other-model"
    with pytest.raises(ValueError, match="re-run ingestion"):
        Retriever.load(index_dir, embedder)
