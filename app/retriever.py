"""Embedding, index loading and similarity search."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from app.models import SourceInfo

# File names inside the index directory (written by app.ingest, read here).
INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_url: str
    title: str
    text: str


class Embedder(Protocol):
    """What ingestion and retrieval need from an embedding model (easy to fake in tests)."""

    model_name: str
    max_seq_length: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 vectors, shape (len(texts), dim)."""
        ...

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Return (start, end) character offsets for each token in text."""
        ...


class SentenceTransformerEmbedder:
    """Local sentence-transformers model; no API key and no network after the first download."""

    def __init__(self, model_name: str) -> None:
        # Imported here so modules that only need the Embedder interface stay fast to import.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.max_seq_length = int(self._model.max_seq_length)
        # Chunking tokenizes whole pages, so lift the tokenizer's 512-token warning threshold.
        self._tokenizer = self._model.tokenizer
        self._tokenizer.model_max_length = 10**9

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        encoded = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(start, end) for start, end in encoded["offset_mapping"]]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]  # best first
    best_score: float
    supported: bool  # True when best_score >= threshold, i.e. worth calling the LLM

    @property
    def confidence(self) -> float:
        """Best similarity clamped to [0, 1]. A retrieval signal, not a calibrated probability."""
        return min(1.0, max(0.0, self.best_score))


class Retriever:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], embedder: Embedder) -> None:
        if index.ntotal != len(chunks):
            raise ValueError(
                f"Index has {index.ntotal} vectors but {len(chunks)} chunks; re-run ingestion."
            )
        self._index = index
        self._chunks = chunks
        self._embedder = embedder

    @classmethod
    def load(cls, index_dir: Path, embedder: Embedder) -> "Retriever":
        index_path = index_dir / INDEX_FILE
        chunks_path = index_dir / CHUNKS_FILE
        meta_path = index_dir / META_FILE
        if not (index_path.exists() and chunks_path.exists() and meta_path.exists()):
            raise FileNotFoundError(
                f"No index found in {index_dir}. Run `python -m app.ingest` first."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Vectors from different models are not comparable, so refuse to mix them.
        if meta["embedding_model"] != embedder.model_name:
            raise ValueError(
                f"Index was built with {meta['embedding_model']} but the configured model is "
                f"{embedder.model_name}; re-run ingestion."
            )
        chunks = [Chunk(**item) for item in json.loads(chunks_path.read_text(encoding="utf-8"))]
        return cls(faiss.read_index(str(index_path)), chunks, embedder)

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Top-k chunks by cosine similarity, best first."""
        if self._index.ntotal == 0:
            return []
        vector = self._embedder.embed([query])
        scores, ids = self._index.search(vector, min(k, self._index.ntotal))
        # FAISS pads with -1 when it has fewer results than requested.
        return [
            RetrievedChunk(self._chunks[i], float(score))
            for score, i in zip(scores[0], ids[0], strict=True)
            if i >= 0
        ]

    def retrieve(self, query: str, k: int, threshold: float) -> RetrievalResult:
        """Search, then decide whether the best match is similar enough to answer from."""
        found = self.search(query, k)
        best = found[0].score if found else 0.0
        return RetrievalResult(chunks=found, best_score=best, supported=best >= threshold)

    def sources(self) -> list[SourceInfo]:
        """Indexed documents with their chunk counts, in the order they were ingested."""
        counts: dict[str, SourceInfo] = {}
        for chunk in self._chunks:
            info = counts.get(chunk.source_url)
            if info is None:
                counts[chunk.source_url] = SourceInfo(
                    title=chunk.title, source_url=chunk.source_url, chunks=1
                )
            else:
                info.chunks += 1
        return list(counts.values())