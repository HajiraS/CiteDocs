"""Pydantic models: API request/response shapes and the validated LLM output."""

import re
import unicodedata

from pydantic import BaseModel, Field, field_validator

from app.config import get_settings

_WHITESPACE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    """Turn all whitespace into single spaces, drop remaining control characters, and trim."""
    value = _WHITESPACE.sub(" ", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Cc")
    return value.strip()


class _CleanedInput(BaseModel):
    """Base for user text inputs: sanitise first, then enforce non-empty and max length."""

    @staticmethod
    def _check(value: str, field_name: str) -> str:
        cleaned = clean_text(value)
        if not cleaned:
            raise ValueError(f"{field_name} must not be empty")
        limit = get_settings().max_question_length
        if len(cleaned) > limit:
            raise ValueError(f"{field_name} must be at most {limit} characters")
        return cleaned


class AskRequest(_CleanedInput):
    question: str

    @field_validator("question")
    @classmethod
    def _validate_question(cls, v: str) -> str:
        return cls._check(v, "question")


class SummarizeRequest(_CleanedInput):
    topic: str

    @field_validator("topic")
    @classmethod
    def _validate_topic(cls, v: str) -> str:
        return cls._check(v, "topic")


class Citation(BaseModel):
    title: str
    source_url: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    answered: bool


class SummarizeResponse(BaseModel):
    topic: str
    bullets: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    answered: bool
    message: str | None = None  # set when answered is False


class SourceInfo(BaseModel):
    title: str
    source_url: str
    chunks: int


class SourcesResponse(BaseModel):
    documents: list[SourceInfo]
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    chunks: int


# --- Validated LLM output (what we require the model to return as JSON) ---


class LLMAnswer(BaseModel):
    answerable: bool
    answer: str
    # Chunk IDs the model says it used; phase 3 checks each one was actually retrieved.
    used_chunk_ids: list[str]


class LLMSummary(BaseModel):
    answerable: bool
    bullets: list[str] = Field(max_length=5)
    used_chunk_ids: list[str]