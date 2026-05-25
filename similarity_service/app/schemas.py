"""Pydantic request and response schemas for the similarity service.

Schemas are intentionally explicit so the auto-generated OpenAPI docs at
/docs describe the contract precisely. Examples are provided on every
field so the Swagger UI Try-it-out panel is usable without any extra
typing.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator

from app.config import (
    DEFAULT_CONFIGURATION,
    FieldConfiguration,
    MAX_TEXT_LENGTH,
    MIN_TEXT_LENGTH,
)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class SimilarityRequest(BaseModel):
    """Request body for POST /similarity.

    Two course descriptions plus an optional configuration and an
    optional override threshold. The course descriptions are split into
    their three Peppi fields (outcomes, contents, assessment) because
    configurations 2, 3, 5, and 6 concatenate fields explicitly, and the
    field boundary information is lost once they are joined.
    """

    model_config = ConfigDict(
        extra="forbid",  # reject unknown keys instead of ignoring them
        json_schema_extra={
            "example": {
                "finnish": {
                    "outcomes": "Opiskelija ymmärtää ohjelmistosuunnittelun perusteet.",
                    "contents": "Olio-ohjelmointi, luokat, oliot, perintä.",
                    "assessment": "Tentti ja harjoitustyö.",
                },
                "english": {
                    "outcomes": "The student understands the basics of software design.",
                    "contents": "Object-oriented programming, classes, objects, inheritance.",
                    "assessment": "Exam and assignment.",
                },
                "field_configuration": "outcomes_raw",
                "threshold": None,
            }
        },
    )

    finnish: "CourseDescription" = Field(
        ..., description="Finnish-side course description fields."
    )
    english: "CourseDescription" = Field(
        ..., description="English-side course description fields."
    )
    field_configuration: FieldConfiguration = Field(
        default=DEFAULT_CONFIGURATION,
        description=(
            "Which of the six text configurations to compute similarity on. "
            "Defaults to outcomes_raw, the recommended configuration from "
            "the thesis Notebook 3."
        ),
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description=(
            "Optional decision threshold in [-1, 1]. If omitted, the "
            "service uses the operating threshold calibrated on the ICT "
            "dataset for the chosen field_configuration."
        ),
    )


class CourseDescription(BaseModel):
    """The three fields scraped from one Peppi course description.

    All three fields are required. Configurations 1 and 4 use only
    `outcomes`, but the other fields are still validated to keep the
    request shape stable across configurations. Empty `contents` or
    `assessment` are allowed because Peppi entries occasionally omit
    them.
    """

    model_config = ConfigDict(extra="forbid")

    outcomes: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="Learning outcomes field.",
    )
    contents: str = Field(
        default="",
        max_length=MAX_TEXT_LENGTH,
        description="Course contents field. May be empty.",
    )
    assessment: str = Field(
        default="",
        max_length=MAX_TEXT_LENGTH,
        description="Assessment description field. May be empty.",
    )

    @field_validator("outcomes")
    @classmethod
    def outcomes_not_whitespace(cls, value: str) -> str:
        """Reject outcomes fields that are only whitespace.

        Pydantic's min_length=1 accepts a string of spaces because it
        measures the raw length. A whitespace-only outcomes field would
        survive clean_text as the empty string, and the embedding
        produced for it would be meaningless. Fail fast at the schema.
        """
        if not value.strip():
            raise ValueError("outcomes must not be empty or whitespace only")
        return value


# Resolve the forward reference now that CourseDescription is defined.
SimilarityRequest.model_rebuild()


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class SimilarityResponse(BaseModel):
    """Response body for POST /similarity.

    Carries the cosine similarity, the applied threshold, the boolean
    decision, and the diagnostic fields a downstream UI may want to
    display alongside the score: which configuration ran, whether the
    token-length budget was exceeded on either side, and the
    server-side inference latency.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cosine_similarity": 0.93,
                "threshold": 0.78,
                "is_similar": True,
                "model_name": "sentence-transformers/LaBSE",
                "field_configuration": "outcomes_raw",
                "truncated_fi": False,
                "truncated_en": False,
                "finnish_tokens": 32,
                "english_tokens": 28,
                "inference_ms": 142.7,
            }
        }
    )

    cosine_similarity: float = Field(
        ..., ge=-1.0, le=1.0, description="LaBSE cosine similarity in [-1, 1]."
    )
    threshold: float = Field(
        ..., ge=-1.0, le=1.0, description="The threshold applied to the score."
    )
    is_similar: bool = Field(
        ..., description="cosine_similarity >= threshold."
    )
    model_name: str = Field(
        ..., description="Hugging Face identifier of the embedding model."
    )
    field_configuration: FieldConfiguration = Field(
        ..., description="The configuration that was applied."
    )
    truncated_fi: bool = Field(
        ...,
        description="True if the Finnish input exceeded the 512-token limit.",
    )
    truncated_en: bool = Field(
        ...,
        description="True if the English input exceeded the 512-token limit.",
    )
    finnish_tokens: int = Field(
        ..., ge=0, description="Finnish token count before any truncation."
    )
    english_tokens: int = Field(
        ..., ge=0, description="English token count before any truncation."
    )
    inference_ms: float = Field(
        ..., ge=0.0, description="Server-side inference latency in milliseconds."
    )


# ---------------------------------------------------------------------------
# Health and version
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for GET /health.

    Liveness only: the process is up.
    """

    status: str = Field(default="ok", description="Always 'ok' when the process is up.")
    service: str = Field(..., description="Service name.")
    version: str = Field(..., description="Service version.")


class ReadinessResponse(BaseModel):
    """Response body for GET /ready.

    Readiness: the model is loaded and the service can answer
    /similarity requests. Returns HTTP 503 when the model is not yet
    loaded.
    """

    status: str = Field(..., description="'ready' or 'loading'.")
    model_loaded: bool = Field(..., description="Whether LaBSE is loaded.")
    device: str = Field(..., description="The device LaBSE is loaded on.")


class VersionResponse(BaseModel):
    """Response body for GET /version.

    Versioning the deployed system so the curriculum tool can record
    exactly which model and threshold combination produced a given
    similarity decision. This is the auditability path identified as
    open work in the May 12 research notes.
    """

    service: str
    version: str
    model_name: str
    max_seq_length: int
    embedding_dim: int
    configurations: list[str]
    default_configuration: str


class ErrorResponse(BaseModel):
    """Standard error envelope returned by exception handlers.

    Stack traces are never returned to the client. The request_id
    correlates with the structured log record on the server.
    """

    code: str
    message: str
    request_id: Optional[str] = None
