"""FastAPI application: /ask, /summarize, /sources, /health."""

import logging
import math
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from app.config import Settings, get_settings
from app.llm import (
    Generator,
    GroqLLMClient,
    LLMClient,
    LLMUnavailableError,
    not_found_response,
    not_found_summary,
)
from app.models import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourcesResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.ratelimit import RateLimiter
from app.retriever import Retriever, SentenceTransformerEmbedder

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    llm_client: LLMClient | None = None,
    limiter: RateLimiter | None = None,
) -> FastAPI:
    """Build the app. Arguments let tests inject a fake retriever, fake LLM and limiter."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        # Missing index or credentials must not stop the service from starting: /health still
        # works and the affected endpoints answer with a clear 503.
        if app.state.retriever is None:
            try:
                embedder = SentenceTransformerEmbedder(settings.embedding_model)
                app.state.retriever = Retriever.load(settings.index_dir, embedder)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Index not loaded: %s", exc)
        if app.state.llm is None:
            try:
                api_key, model = settings.require_llm_credentials()
                app.state.llm = GroqLLMClient(
                    api_key, model, settings.llm_timeout_seconds, settings.llm_max_retries
                )
            except RuntimeError as exc:
                logger.warning("LLM not configured: %s", exc)
        yield

    app = FastAPI(title="Documentation Knowledge Assistant", version="1.0.0", lifespan=lifespan)
    app.state.retriever = retriever
    app.state.llm = llm_client
    app.state.limiter = limiter or RateLimiter(settings.rate_limit_per_minute)

    def get_retriever(request: Request) -> Retriever:
        loaded = request.app.state.retriever
        if loaded is None:
            raise HTTPException(
                status_code=503,
                detail="The documentation index is not loaded. "
                "Run `python -m app.ingest` and restart the service.",
            )
        return loaded

    def get_generator(request: Request) -> Generator:
        client = request.app.state.llm
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="The language model is not configured. Set GROQ_API_KEY and GROQ_MODEL.",
            )
        return Generator(client)

    def enforce_rate_limit(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        wait = request.app.state.limiter.check(client)
        if wait > 0:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please slow down.",
                headers={"Retry-After": str(math.ceil(wait))},
            )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded = request.app.state.retriever
        return HealthResponse(
            status="ok" if loaded is not None else "degraded",
            index_loaded=loaded is not None,
            chunks=loaded.num_chunks if loaded is not None else 0,
        )

    @app.get("/sources", response_model=SourcesResponse)
    def sources(retriever: Retriever = Depends(get_retriever)) -> SourcesResponse:
        return SourcesResponse(documents=retriever.sources(), total_chunks=retriever.num_chunks)

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(enforce_rate_limit)])
    def ask(
        body: AskRequest, request: Request, retriever: Retriever = Depends(get_retriever)
    ) -> AskResponse:
        result = retriever.retrieve(body.question, settings.top_k, settings.similarity_threshold)
        logger.info("ask best_score=%.3f supported=%s", result.best_score, result.supported)
        if not result.supported:
            return not_found_response(result.confidence)  # LLM is not called
        generator = get_generator(request)
        try:
            return generator.answer(body.question, result)
        except LLMUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/summarize",
        response_model=SummarizeResponse,
        dependencies=[Depends(enforce_rate_limit)],
    )
    def summarize(
        body: SummarizeRequest, request: Request, retriever: Retriever = Depends(get_retriever)
    ) -> SummarizeResponse:
        result = retriever.retrieve(body.topic, settings.top_k, settings.similarity_threshold)
        logger.info("summarize best_score=%.3f supported=%s", result.best_score, result.supported)
        if not result.supported:
            return not_found_summary(body.topic, result.confidence)  # LLM is not called
        generator = get_generator(request)
        try:
            return generator.summarize(body.topic, result)
        except LLMUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()