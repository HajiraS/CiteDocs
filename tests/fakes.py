"""Test doubles: a deterministic offline embedder and a scripted LLM client."""

import json
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "is", "are", "in", "on", "for", "how", "its",
        "it", "that", "can", "what", "who", "does", "do", "when", "any", "each", "as", "by",
        "so", "be", "after", "says",
    }
)  # fmt: skip


class FakeEmbedder:
    """Bag-of-words embedder over a fixed vocabulary; words outside it are ignored.

    Cosine similarity is then plain word overlap, so tests are deterministic, and a question
    with no known words scores exactly 0.
    """

    model_name = "fake-bow"
    max_seq_length = 256

    def __init__(self, corpus: str) -> None:
        words = sorted({w for w in _TOKEN.findall(corpus.lower()) if w not in _STOPWORDS})
        self._vocab = {word: i for i, word in enumerate(words)}

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self._vocab)), dtype="float32")
        for row, text in enumerate(texts):
            for word in _TOKEN.findall(text.lower()):
                col = self._vocab.get(word)
                if col is not None:
                    matrix[row, col] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # an all-zero vector stays zero instead of dividing by zero
        return matrix / norms

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return [match.span() for match in re.finditer(r"\S+", text)]


class FakeLLM:
    """Returns scripted replies in order (or raises a scripted exception) and records calls."""

    def __init__(self, *replies: str | Exception) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, str]] = []  # (system, user) per call

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self._replies:
            raise AssertionError("FakeLLM was called more times than it was scripted for")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def answer_json(
    ids: list[str], answer: str = "Clients renew a lease before it expires.", answerable: bool = True
) -> str:
    return json.dumps({"answerable": answerable, "answer": answer, "used_chunk_ids": ids})


def summary_json(bullets: list[str], ids: list[str], answerable: bool = True) -> str:
    return json.dumps({"answerable": answerable, "bullets": bullets, "used_chunk_ids": ids})
