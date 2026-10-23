"""Front matter parsing, chunking and index building."""

import json
import logging
import re
from pathlib import Path

import pytest

from app.ingest import build_chunks, build_index, chunk_text, load_documents, parse_document
from app.retriever import CHUNKS_FILE, INDEX_FILE, META_FILE
from tests.corpus import CHUNK_OVERLAP, CHUNK_TOKENS, DHCP_TITLE, DHCP_URL
from tests.fakes import FakeEmbedder


def _offsets(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"\S+", text)]


def test_chunk_text_windows_overlap_by_configured_tokens():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = chunk_text(text, _offsets(text), chunk_tokens=4, overlap=1)
    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_chunk_text_covers_every_token_including_the_short_tail():
    text = " ".join(f"w{i}" for i in range(11))
    chunks = chunk_text(text, _offsets(text), chunk_tokens=4, overlap=1)
    assert chunks[-1] == "w9 w10"
    assert set(" ".join(chunks).split()) == set(text.split())


def test_chunk_text_short_and_empty_inputs():
    assert chunk_text("", [], chunk_tokens=4, overlap=1) == []
    exact = "a b c d"
    assert chunk_text(exact, _offsets(exact), chunk_tokens=4, overlap=1) == [exact]


def test_chunk_text_rejects_overlap_that_is_not_smaller_than_chunk():
    with pytest.raises(ValueError, match="smaller"):
        chunk_text("a b c", _offsets("a b c"), chunk_tokens=3, overlap=3)


def test_build_chunks_keeps_metadata_and_sequential_ids(docs_dir: Path, embedder: FakeEmbedder):
    chunks = build_chunks(load_documents(docs_dir), embedder, chunk_tokens=10, overlap=2)
    dhcp = [c for c in chunks if c.source_url == DHCP_URL]
    assert len(dhcp) > 1
    assert [c.chunk_id for c in dhcp] == [f"dhcp-leases-{i:03d}" for i in range(len(dhcp))]
    assert {c.title for c in dhcp} == {DHCP_TITLE}


def test_parse_document_reads_front_matter(docs_dir: Path):
    doc = parse_document(docs_dir / "dhcp-leases.md")
    assert (doc.doc_id, doc.source_url, doc.title) == ("dhcp-leases", DHCP_URL, DHCP_TITLE)
    assert doc.body.startswith("A DHCP server")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("no front matter here", "missing front matter"),
        ("---\ntitle: x\n---\nbody", "needs both"),
        ("---\nsource_url: u\ntitle: Configuring DNS: Overview\n---\nbody", "invalid front matter"),
    ],
)
def test_parse_document_rejects_bad_front_matter(tmp_path: Path, content: str, message: str):
    path = tmp_path / "bad.md"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        parse_document(path)


def test_load_documents_skips_empty_bodies_and_non_doc_files(docs_dir: Path):
    (docs_dir / "empty.md").write_text("---\nsource_url: u\ntitle: t\n---\n", encoding="utf-8")
    (docs_dir / "notes.csv").write_text("a,b", encoding="utf-8")
    (docs_dir / ".gitkeep").write_text("", encoding="utf-8")
    assert [d.doc_id for d in load_documents(docs_dir)] == ["dhcp-leases", "dns-caching"]


def test_load_documents_rejects_duplicate_stems(docs_dir: Path):
    (docs_dir / "dhcp-leases.txt").write_text("---\nsource_url: u\ntitle: t\n---\nx", "utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_documents(docs_dir)


def test_load_documents_missing_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


def test_build_index_writes_all_files_and_consistent_metadata(
    docs_dir: Path, tmp_path: Path, embedder: FakeEmbedder
):
    out = tmp_path / "idx"
    meta = build_index(docs_dir, out, embedder, CHUNK_TOKENS, CHUNK_OVERLAP)
    for name in (INDEX_FILE, CHUNKS_FILE, META_FILE):
        assert (out / name).exists()
    chunks = json.loads((out / CHUNKS_FILE).read_text(encoding="utf-8"))
    assert meta["chunks"] == len(chunks) == 2
    assert meta["documents"] == 2
    assert meta["embedding_model"] == "fake-bow"


def test_build_index_warns_when_chunks_exceed_the_model_limit(
    docs_dir: Path, tmp_path: Path, embedder: FakeEmbedder, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.WARNING):
        build_index(docs_dir, tmp_path / "idx", embedder, chunk_tokens=300, overlap=20)
    assert "exceeds" in caplog.text


def test_build_index_requires_documents(tmp_path: Path, embedder: FakeEmbedder):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No .md or .txt"):
        build_index(empty, tmp_path / "idx", embedder, CHUNK_TOKENS, CHUNK_OVERLAP)
