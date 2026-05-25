"""LaBSE-based similarity inference for the bilingual similarity service.

Reproduces the LaBSE encoding path of Notebook 2 of the thesis pipeline:
  - sentence-transformers SentenceTransformer wrapper
  - max_seq_length set to 512 after model load
  - encode() with convert_to_numpy=True
  - L2 normalisation of every embedding before cosine similarity
  - Cosine similarity computed as dot product on unit vectors

If the encoding path diverges from Notebook 2, the similarity scores
returned by this service no longer correspond to the F1 numbers
reported in the thesis evaluation.

Source: notebooks/02_embedding_generation.ipynb in the thesis repository.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


@dataclass
class LoadedModel:
    """Container for the loaded LaBSE model and its tokenizer.

    Both fields are populated by load_model. The tokenizer is the
    SentenceTransformer's own tokenizer, retained separately so that
    truncation tracking can be done without going through encode().
    """

    model: SentenceTransformer
    tokenizer: object
    device: torch.device


def load_model() -> LoadedModel:
    """Load LaBSE onto the available device and set max_seq_length.

    Identical to the loading block for the 'labse' branch in
    Notebook 2: SentenceTransformer is constructed with the device
    argument, then max_seq_length is set explicitly. The explicit
    assignment is necessary because some SentenceTransformer models
    default to a smaller value than 512 even when the underlying
    transformer supports more.

    Verifies the model produces 768-dimensional output via a one-token
    probe, matching the assertion used in Notebook 2.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading LaBSE on device=%s", device)

    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device=str(device))
    model.max_seq_length = MAX_SEQ_LENGTH
    load_seconds = time.perf_counter() - t0

    logger.info(
        "LaBSE loaded in %.2fs (max_seq_length=%d)",
        load_seconds,
        model.max_seq_length,
    )

    # Dimensionality probe. Matches the (n, 768) assertion in Notebook 2.
    probe = model.encode(
        ["dimension probe"],
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    if probe.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Expected {EMBEDDING_DIM}-d embeddings, got {probe.shape[1]}-d"
        )

    return LoadedModel(model=model, tokenizer=model.tokenizer, device=device)


# ---------------------------------------------------------------------------
# Encoding and similarity
# ---------------------------------------------------------------------------


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """Normalize each row of a 2-D array to unit L2 norm.

    Reproduces Notebook 2's l2_normalize. The clamp at 1e-10 prevents
    division by zero on degenerate (all-zero) vectors. After this step
    cosine similarity collapses to a simple dot product, which is what
    cosine_similarity() below expects.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-10, a_max=None)


def encode(loaded: LoadedModel, texts: list[str]) -> np.ndarray:
    """Encode a list of texts into L2-normalised LaBSE embeddings.

    Wraps SentenceTransformer.encode with the same arguments used by
    the labse branch of Notebook 2. Output shape is (len(texts), 768).
    The L2 normalisation step matches Notebook 2's post-encode
    normalisation that is verified per array by an assertion.
    """
    embeddings = loaded.model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    embeddings = l2_normalize(embeddings)
    return embeddings


def token_length(loaded: LoadedModel, text: str) -> int:
    """Tokenise without truncation and return the resulting token count.

    Used to set the `truncated` flag in the response. Mirrors Notebook 2's
    truncation tracking: encode with add_special_tokens=True and
    truncation=False, then count the tokens.
    """
    tokens = loaded.tokenizer.encode(text, add_special_tokens=True, truncation=False)
    return len(tokens)


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors.

    After L2 normalisation cosine reduces to a dot product, which is the
    same computation used in Notebook 3 for pairwise scoring.
    Returns a Python float so the result is JSON-serialisable.
    """
    return float(np.dot(emb_a, emb_b))


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


@dataclass
class SimilarityResult:
    """The full result of one similarity computation.

    Captures both the similarity score and the diagnostic fields the
    response schema needs to surface: which configuration ran, the
    threshold actually applied, whether truncation occurred on either
    side, and the inference latency.
    """

    cosine_similarity: float
    threshold: float
    is_similar: bool
    truncated_fi: bool
    truncated_en: bool
    finnish_tokens: int
    english_tokens: int
    inference_ms: float


def compute_similarity(
    loaded: LoadedModel,
    finnish_text: str,
    english_text: str,
    threshold: float,
) -> SimilarityResult:
    """Compute LaBSE cosine similarity for one bilingual text pair.

    The function does not take field configuration: it expects the
    caller (the FastAPI endpoint) to have already passed the texts
    through `preprocessing.build_configuration` for the chosen
    configuration. This keeps inference focused on the model call.

    Truncation is reported separately for each language because their
    profiles can differ: Finnish tends to produce more subword tokens
    than English for the same content because of morphological richness.
    """
    t0 = time.perf_counter()

    # Token-length analysis BEFORE encoding so truncation status is
    # measured on the same text the model will see.
    fi_tokens = token_length(loaded, finnish_text)
    en_tokens = token_length(loaded, english_text)

    # Encode both sides as a batch of 2 for one forward pass.
    with torch.no_grad():
        embeddings = encode(loaded, [finnish_text, english_text])

    fi_emb, en_emb = embeddings[0], embeddings[1]
    score = cosine_similarity(fi_emb, en_emb)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    return SimilarityResult(
        cosine_similarity=score,
        threshold=threshold,
        is_similar=score >= threshold,
        truncated_fi=fi_tokens > MAX_SEQ_LENGTH,
        truncated_en=en_tokens > MAX_SEQ_LENGTH,
        finnish_tokens=fi_tokens,
        english_tokens=en_tokens,
        inference_ms=inference_ms,
    )
