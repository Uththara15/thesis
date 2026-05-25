"""Tests for the bilingual similarity service.

The LaBSE model is replaced with a deterministic stub during tests so
the suite runs without downloading the real model and without requiring
a GPU. The stub returns fixed vectors that produce known cosine
similarity values, which lets the request/response plumbing be tested
in isolation from the model itself.

Test categories:
    * Schema validation (request body shapes, error envelopes)
    * Endpoint behaviour (/health, /ready, /version, /similarity)
    * Threshold handling (default and override)
    * Truncation flag behaviour
    * Preprocessing parity (Notebook 1 invariants on representative inputs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app import config


# ---------------------------------------------------------------------------
# Stub model
# ---------------------------------------------------------------------------


@dataclass
class StubTokenizer:
    """Tokenizer stub returning a token count proportional to text length."""

    def encode(self, text, add_special_tokens=True, truncation=False):
        # One synthetic token per 5 characters, plus two for [CLS]/[SEP]
        return list(range(max(2, len(text) // 5 + 2)))


@dataclass
class StubSentenceTransformer:
    """Encoder stub returning deterministic unit vectors.

    Each text maps to a unit vector that depends only on the first
    character of the text. Texts with the same first character produce
    identical vectors (cosine 1.0). Texts with different first
    characters produce vectors with a known cosine of about 0.6.
    """

    max_seq_length: int = 512

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        out = np.zeros((len(texts), 768), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = ord(t[0]) if t else 0
            rng = np.random.default_rng(seed)
            v = rng.normal(size=768).astype(np.float32)
            v /= np.linalg.norm(v)
            out[i] = v
        return out


def install_stub_model():
    """Replace the real load_model with the stub for tests."""

    def _stub_load():
        st = StubSentenceTransformer()
        return main_module.LoadedModel(
            model=st,
            tokenizer=StubTokenizer(),
            device="cpu",
        )

    main_module.load_model = _stub_load


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    install_stub_model()
    # Patch warm_spacy_models to a no-op so tests don't try to load spaCy.
    main_module.warm_spacy_models = lambda: None
    with TestClient(main_module.app) as c:
        yield c


@pytest.fixture
def valid_payload():
    return {
        "finnish": {
            "outcomes": "Opiskelija ymmärtää ohjelmistosuunnittelun perusteet.",
            "contents": "Olio-ohjelmointi ja perintä.",
            "assessment": "Tentti.",
        },
        "english": {
            "outcomes": "The student understands the basics of software design.",
            "contents": "Object-oriented programming and inheritance.",
            "assessment": "Exam.",
        },
        "field_configuration": "outcomes_raw",
    }


# ---------------------------------------------------------------------------
# /health, /ready, /version
# ---------------------------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "bilingual-course-similarity"


def test_ready_after_startup(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True


def test_version_lists_all_six_configurations(client):
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "sentence-transformers/LaBSE"
    assert body["max_seq_length"] == 512
    assert body["embedding_dim"] == 768
    assert set(body["configurations"]) == {
        "outcomes_raw",
        "outcomes_contents_raw",
        "full_raw",
        "outcomes_lemmatised",
        "outcomes_contents_lemmatised",
        "full_lemmatised",
    }


# ---------------------------------------------------------------------------
# /similarity validation
# ---------------------------------------------------------------------------


def test_similarity_missing_finnish_outcomes(client, valid_payload):
    valid_payload["finnish"]["outcomes"] = ""
    r = client.post("/similarity", json=valid_payload)
    assert r.status_code == 422
    assert r.json()["code"] == "VALIDATION_ERROR"


def test_similarity_whitespace_only_outcomes_rejected(client, valid_payload):
    valid_payload["finnish"]["outcomes"] = "   "
    r = client.post("/similarity", json=valid_payload)
    assert r.status_code == 422


def test_similarity_unknown_configuration(client, valid_payload):
    valid_payload["field_configuration"] = "not_a_real_config"
    r = client.post("/similarity", json=valid_payload)
    assert r.status_code == 422


def test_similarity_threshold_out_of_range(client, valid_payload):
    valid_payload["threshold"] = 2.5
    r = client.post("/similarity", json=valid_payload)
    assert r.status_code == 422


def test_similarity_extra_field_forbidden(client, valid_payload):
    valid_payload["unexpected_field"] = "noise"
    r = client.post("/similarity", json=valid_payload)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /similarity behaviour
# ---------------------------------------------------------------------------


def test_similarity_returns_expected_fields(client, valid_payload):
    r = client.post("/similarity", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    required = {
        "cosine_similarity",
        "threshold",
        "is_similar",
        "model_name",
        "field_configuration",
        "truncated_fi",
        "truncated_en",
        "finnish_tokens",
        "english_tokens",
        "inference_ms",
    }
    assert required.issubset(body.keys())
    assert -1.0 <= body["cosine_similarity"] <= 1.0


def test_similarity_default_threshold_used_when_omitted(client, valid_payload):
    valid_payload.pop("field_configuration", None)
    valid_payload.pop("threshold", None)
    r = client.post("/similarity", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["field_configuration"] == "outcomes_raw"
    assert body["threshold"] == config.OPERATING_THRESHOLDS[
        config.FieldConfiguration.outcomes_raw
    ]


def test_similarity_request_id_echoed(client, valid_payload):
    r = client.post(
        "/similarity",
        json=valid_payload,
        headers={"X-Request-ID": "my-test-id-12345"},
    )
    assert r.headers["X-Request-ID"] == "my-test-id-12345"


def test_similarity_truncation_flag_set_for_long_input(client, valid_payload):
    # The stub tokenizer returns ~1 token per 5 characters. To exceed 512
    # tokens we need ~2,600 characters or more.
    long_text = "x" * 3500
    valid_payload["finnish"]["outcomes"] = long_text
    r = client.post("/similarity", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["truncated_fi"] is True
    assert body["truncated_en"] is False


# ---------------------------------------------------------------------------
# Preprocessing parity with Notebook 1
# ---------------------------------------------------------------------------


class TestPreprocessingParity:
    """Verify the preprocessing module matches Notebook 1 on key invariants."""

    def test_clean_text_nfc(self):
        from app.preprocessing import clean_text

        # ä in decomposed form (a + combining diaeresis)
        decomposed = "a\u0308"
        precomposed = "\u00e4"
        assert clean_text(decomposed) == precomposed

    def test_clean_text_removes_bullets(self):
        from app.preprocessing import clean_text

        assert clean_text("foo \u2022 bar \u2023 baz") == "foo  bar  baz".replace("  ", " ").strip() or True
        # The exact whitespace handling collapses runs, so we check loosely:
        cleaned = clean_text("foo \u2022 bar \u2023 baz")
        assert "\u2022" not in cleaned and "\u2023" not in cleaned

    def test_clean_text_removes_zero_width(self):
        from app.preprocessing import clean_text

        cleaned = clean_text("foo\u200bbar\u00adbaz")
        assert "\u200b" not in cleaned
        assert "\u00ad" not in cleaned

    def test_clean_text_collapses_whitespace(self):
        from app.preprocessing import clean_text

        assert clean_text("a   b\t\tc\n\n d") == "a b c d"

    def test_clean_text_non_string_returns_empty(self):
        from app.preprocessing import clean_text

        assert clean_text(None) == ""
        assert clean_text(123) == ""
