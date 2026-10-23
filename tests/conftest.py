"""Shared fixtures: two tiny docs on disk, a fake embedder, and a real FAISS index."""

from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.ingest import build_index
from app.retriever import Retriever
from tests.corpus import CHUNK_OVERLAP, CHUNK_TOKENS, DOCS
from tests.fakes import FakeEmbedder


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch):
    """Validators call get_settings(); pin the limit they use and drop any cached instance."""
    monkeypatch.setenv("MAX_QUESTION_LENGTH", "500")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    path = tmp_path / "docs"
    path.mkdir()
    for stem, (url, title, body) in DOCS.items():
        content = f'---\nsource_url: {url}\ntitle: "{title}"\n---\n{body}\n'
        (path / f"{stem}.md").write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def embedder() -> FakeEmbedder:
    corpus = " ".join(f"{title} {body}" for _, title, body in DOCS.values())
    return FakeEmbedder(corpus)


@pytest.fixture
def index_dir(tmp_path: Path, docs_dir: Path, embedder: FakeEmbedder) -> Path:
    path = tmp_path / "index"
    build_index(docs_dir, path, embedder, CHUNK_TOKENS, CHUNK_OVERLAP)
    return path


@pytest.fixture
def retriever(index_dir: Path, embedder: FakeEmbedder) -> Retriever:
    return Retriever.load(index_dir, embedder)


@pytest.fixture
def settings(docs_dir: Path, index_dir: Path) -> Settings:
    # Explicit values override any real environment variables or .env on the developer's machine.
    return Settings(
        _env_file=None,
        groq_api_key=None,
        groq_model=None,
        docs_dir=docs_dir,
        index_dir=index_dir,
        chunk_tokens=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=2,
        similarity_threshold=0.1,
        rate_limit_per_minute=100,
    )
