"""FastAPI application for the bilingual course similarity service.

Endpoints:
    POST /similarity   Compute LaBSE cosine similarity for a bilingual pair.
    GET  /health       Liveness check.
    GET  /ready        Readiness check (model loaded).
    GET  /version      Service and model metadata for audit logging.

The LaBSE model is loaded once at start-up using FastAPI's lifespan
context manager. This is the documented pattern for shared, expensive
machine learning resources: load before the application accepts
requests, hold across the application lifetime, release on shutdown.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import (
    DEFAULT_CONFIGURATION,
    FieldConfiguration,
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    EMBEDDING_DIM,
    OPERATING_THRESHOLDS,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from app.inference import LoadedModel, compute_similarity, load_model
from app.preprocessing import build_configuration, warm_spacy_models
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
    SimilarityRequest,
    SimilarityResponse,
    VersionResponse,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s [%(name)s] "
        "%(message)s"
    ),
)
logger = logging.getLogger("similarity_service")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: load model and spaCy at startup, release at shutdown.

    The model and the spaCy pipelines are stored on app.state so that
    request handlers reach them via Request.app.state rather than via
    module globals. This makes testing easier because a test app can
    swap in a mock model without monkeypatching module globals.
    """
    logger.info("Startup: loading LaBSE and spaCy pipelines")
    app.state.model: LoadedModel = load_model()
    warm_spacy_models()
    app.state.model_loaded = True
    logger.info("Startup complete: service is ready to accept requests")

    yield

    # Shutdown
    logger.info("Shutdown: releasing model")
    app.state.model = None
    app.state.model_loaded = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Bilingual Course Description Similarity Service",
    description=(
        "A FastAPI service wrapping the LaBSE pipeline evaluated in the "
        "thesis 'Evaluating Multilingual Sentence Embedding Models for "
        "Bilingual Course Description Similarity in Finnish Higher "
        "Education' (JAMK University of Applied Sciences, May 2026)."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a request_id to every request and surface it in responses.

    The request_id is generated server-side (uuid4) unless the client
    supplies one in the X-Request-ID header. It is then attached to
    request.state for use by handlers and echoed in the response header
    so the client can correlate against server logs.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIdMiddleware)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    """Return a structured 422 instead of FastAPI's default verbose error.

    The default response is friendly during development but leaks
    Pydantic internals. The envelope below is stable and machine-readable,
    which is what the curriculum tool will need.
    """
    rid = getattr(request.state, "request_id", None)
    logger.warning("Validation error rid=%s errors=%s", rid, exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            code="VALIDATION_ERROR",
            message="Request body failed validation.",
            request_id=rid,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    """Catch-all so the client never sees a stack trace.

    The full exception is logged with the request id so that operations
    can match the generic 500 the client receives to the detailed log
    entry on the server.
    """
    rid = getattr(request.state, "request_id", None)
    logger.exception("Unhandled error rid=%s", rid)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred. See server logs.",
            request_id=rid,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["status"],
    summary="Liveness check",
)
async def get_health() -> HealthResponse:
    """Liveness probe. Returns 200 if the process is up.

    Liveness is intentionally weaker than readiness: a load balancer
    using only /health would start sending traffic before the model is
    loaded, which would fail. Use /ready to gate traffic.
    """
    return HealthResponse(service=SERVICE_NAME, version=SERVICE_VERSION)


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["status"],
    summary="Readiness check",
    responses={503: {"model": ErrorResponse}},
)
async def get_ready(request: Request) -> ReadinessResponse:
    """Readiness probe. Returns 200 only if LaBSE has finished loading.

    Returns 503 with an ErrorResponse envelope when the model is not
    yet ready, so that the same response shape applies to both states.
    """
    loaded: bool = getattr(request.app.state, "model_loaded", False)
    model: LoadedModel | None = getattr(request.app.state, "model", None)

    if not loaded or model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                code="NOT_READY",
                message="Model not yet loaded.",
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )

    return ReadinessResponse(
        status="ready",
        model_loaded=True,
        device=str(model.device),
    )


@app.get(
    "/version",
    response_model=VersionResponse,
    tags=["status"],
    summary="Service and model metadata",
)
async def get_version() -> VersionResponse:
    """Return service identity and model configuration.

    Intended for audit logging on the curriculum tool side: every
    similarity decision the tool stores should also store the response
    of /version at the time of the decision, so the decision is
    reproducible from the recorded metadata.
    """
    return VersionResponse(
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        configurations=[c.value for c in FieldConfiguration],
        default_configuration=DEFAULT_CONFIGURATION.value,
    )


@app.post(
    "/similarity",
    response_model=SimilarityResponse,
    tags=["similarity"],
    summary="Compute bilingual similarity for one pair",
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def post_similarity(
    payload: SimilarityRequest, request: Request
) -> SimilarityResponse:
    """Compute LaBSE cosine similarity for a Finnish-English course pair.

    The function follows the order of operations from the thesis
    Notebooks 1 and 2:
        1. Build the chosen configuration on the Finnish side
           (clean + optional lemmatise + concatenate).
        2. Build the chosen configuration on the English side.
        3. Encode both with LaBSE to 768-d L2-normalised vectors.
        4. Compute cosine similarity as a dot product.
        5. Apply the threshold (request override or ICT operating point).
    """
    rid = request.state.request_id
    model: LoadedModel | None = getattr(request.app.state, "model", None)
    if model is None:
        # Readiness gate. Mirrors /ready but lives here for safety in
        # case /similarity is called before /ready.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                code="NOT_READY",
                message="Model not yet loaded.",
                request_id=rid,
            ).model_dump(),
        )

    cfg = payload.field_configuration
    threshold = (
        payload.threshold
        if payload.threshold is not None
        else OPERATING_THRESHOLDS[cfg]
    )

    # Build the chosen configuration on each side, identical to Notebook 1.
    fi_text = build_configuration(
        outcomes=payload.finnish.outcomes,
        contents=payload.finnish.contents,
        assessment=payload.finnish.assessment,
        configuration=cfg,
        language="fi",
    )
    en_text = build_configuration(
        outcomes=payload.english.outcomes,
        contents=payload.english.contents,
        assessment=payload.english.assessment,
        configuration=cfg,
        language="en",
    )

    if not fi_text or not en_text:
        # Should not happen because outcomes is required to be non-empty,
        # but defensive against unusual preprocessing outputs.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                code="EMPTY_AFTER_PREPROCESSING",
                message=(
                    "Input text was empty after preprocessing for the "
                    "selected configuration."
                ),
                request_id=rid,
            ).model_dump(),
        )

    result = compute_similarity(
        loaded=model,
        finnish_text=fi_text,
        english_text=en_text,
        threshold=threshold,
    )

    # Structured log line. Matches the fields listed in the May 12
    # research notes: request id, configuration, token counts, truncation,
    # similarity, threshold, decision, latency.
    logger.info(
        "similarity rid=%s cfg=%s fi_tok=%d en_tok=%d trunc_fi=%s "
        "trunc_en=%s sim=%.4f thr=%.2f is_similar=%s ms=%.1f",
        rid,
        cfg.value,
        result.finnish_tokens,
        result.english_tokens,
        result.truncated_fi,
        result.truncated_en,
        result.cosine_similarity,
        result.threshold,
        result.is_similar,
        result.inference_ms,
    )

    return SimilarityResponse(
        cosine_similarity=result.cosine_similarity,
        threshold=result.threshold,
        is_similar=result.is_similar,
        model_name=MODEL_NAME,
        field_configuration=cfg,
        truncated_fi=result.truncated_fi,
        truncated_en=result.truncated_en,
        finnish_tokens=result.finnish_tokens,
        english_tokens=result.english_tokens,
        inference_ms=result.inference_ms,
    )
