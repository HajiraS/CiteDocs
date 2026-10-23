"""Ingestion: read local docs, chunk them, embed them, and write a FAISS index to disk.

Run with:  python -m app.ingest
"""

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import faiss
import yaml

from app.config import get_settings
from app.retriever import (
    CHUNKS_FILE,
    INDEX_FILE,
    META_FILE,
    Chunk,
    Embedder,
    SentenceTransformerEmbedder,
)

logger = logging.getLogger(__name__)

# Every doc must start with a "---" front matter block containing source_url and title.
_FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)(.*)\Z", re.DOTALL)
_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class Document:
    doc_id: str  # file name without extension; used to build chunk IDs
    source_url: str
    title: str
    body: str


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8-sig")
    match = _FRONT_MATTER.match(raw)
    if match is None:
        raise ValueError(f"{path.name}: missing front matter (--- block with source_url, title)")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ValueError(
            f"{path.name}: invalid front matter ({exc}). Quote titles that contain ':'."
        ) from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{path.name}: front matter must be 'key: value' lines")
    source_url = str(meta.get("source_url") or "").strip()
    title = str(meta.get("title") or "").strip()
    if not source_url or not title:
        raise ValueError(f"{path.name}: front matter needs both source_url and title")
    return Document(path.stem, source_url, title, match.group(2).strip())


def load_documents(docs_dir: Path) -> list[Document]:
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")
    paths = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in _SUFFIXES)
    stems = [p.stem for p in paths]
    duplicates = sorted({s for s in stems if stems.count(s) > 1})
    if duplicates:
        raise ValueError(f"Duplicate file names (ignoring extension): {', '.join(duplicates)}")
    documents = []
    for path in paths:
        doc = parse_document(path)
        if not doc.body:
            logger.warning("%s has no text after the front matter; skipping", path.name)
            continue
        documents.append(doc)
    return documents


def chunk_text(
    text: str, offsets: list[tuple[int, int]], chunk_tokens: int, overlap: int
) -> list[str]:
    """Split text into windows of chunk_tokens tokens, each sharing `overlap` tokens with the last.

    `offsets` holds one (start, end) character span per token, so each window is cut from the
    original text and keeps its exact wording and formatting.
    """
    if overlap >= chunk_tokens:
        raise ValueError("overlap must be smaller than chunk_tokens")
    total = len(offsets)
    step = chunk_tokens - overlap
    chunks: list[str] = []
    start = 0
    while start < total:
        end = min(start + chunk_tokens, total)
        piece = text[offsets[start][0] : offsets[end - 1][1]].strip()
        if piece:
            chunks.append(piece)
        if end == total:
            break
        start += step
    return chunks


def build_chunks(
    documents: list[Document], embedder: Embedder, chunk_tokens: int, overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        offsets = embedder.token_offsets(doc.body)
        for i, piece in enumerate(chunk_text(doc.body, offsets, chunk_tokens, overlap)):
            chunks.append(Chunk(f"{doc.doc_id}-{i:03d}", doc.source_url, doc.title, piece))
    return chunks


def build_index(
    docs_dir: Path, index_dir: Path, embedder: Embedder, chunk_tokens: int, overlap: int
) -> dict:
    """Chunk and embed every doc, write the index files, and return the metadata written."""
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(f"No .md or .txt documents found in {docs_dir}")
    if chunk_tokens > embedder.max_seq_length:
        logger.warning(
            "CHUNK_TOKENS=%d exceeds the embedding model's limit of %d tokens; "
            "the end of each chunk will not be reflected in its embedding.",
            chunk_tokens,
            embedder.max_seq_length,
        )

    chunks = build_chunks(documents, embedder, chunk_tokens, overlap)
    # Prefix the title so a chunk that never names its product is still findable.
    vectors = embedder.embed([f"{c.title}\n{c.text}" for c in chunks])

    # Vectors are normalised, so inner product equals cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / INDEX_FILE))
    (index_dir / CHUNKS_FILE).write_text(
        json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    per_document = Counter((c.title, c.source_url) for c in chunks)
    meta = {
        "embedding_model": embedder.model_name,
        "chunk_tokens": chunk_tokens,
        "chunk_overlap": overlap,
        "documents": len(documents),
        "chunks": len(chunks),
        "built_at": datetime.now(UTC).isoformat(),
        "per_document": [
            {"title": title, "source_url": url, "chunks": n}
            for (title, url), n in per_document.items()
        ],
    }
    (index_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the FAISS index from local docs.")
    parser.add_argument("--docs-dir", type=Path, default=settings.docs_dir)
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    meta = build_index(
        args.docs_dir, args.index_dir, embedder, settings.chunk_tokens, settings.chunk_overlap
    )
    for item in meta["per_document"]:
        print(f"{item['chunks']:>4} chunks  {item['title']}")
    print(f"Indexed {meta['chunks']} chunks from {meta['documents']} documents -> {args.index_dir}")


if __name__ == "__main__":
    main()