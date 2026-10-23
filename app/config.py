"""Central configuration, loaded from environment variables (and an optional .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Groq: no defaults on purpose. They are checked when the LLM client is built,
    # so ingestion, retrieval and the offline tests work without a key.
    groq_api_key: SecretStr | None = None
    groq_model: str | None = None

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking, measured in embedding-model tokens
    chunk_tokens: int = Field(500, ge=50)
    chunk_overlap: int = Field(50, ge=0)

    # Retrieval
    top_k: int = Field(4, ge=1, le=20)
    similarity_threshold: float = Field(0.30, ge=0.0, le=1.0)

    # Input limits and rate limiting
    max_question_length: int = Field(500, ge=1)
    rate_limit_per_minute: int = Field(20, ge=1)

    # LLM call behaviour
    llm_timeout_seconds: float = Field(20.0, gt=0)
    llm_max_retries: int = Field(3, ge=0, le=6)

    # Paths
    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("index")

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_TOKENS")
        return self

    def require_llm_credentials(self) -> tuple[str, str]:
        """Return (api_key, model) or raise a clear error. Called by the LLM client only."""
        if self.groq_api_key is None or not self.groq_api_key.get_secret_value():
            raise RuntimeError("GROQ_API_KEY is not set")
        if not self.groq_model:
            raise RuntimeError("GROQ_MODEL is not set")
        return self.groq_api_key.get_secret_value(), self.groq_model


@lru_cache
def get_settings() -> Settings:
    return Settings()