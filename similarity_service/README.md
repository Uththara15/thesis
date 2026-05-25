# Bilingual Course Description Similarity Service

A FastAPI service wrapping the LaBSE-based bilingual similarity pipeline evaluated in the JAMK Bachelor's thesis *Evaluating Multilingual Sentence Embedding Models for Bilingual Course Description Similarity in Finnish Higher Education* (May 2026).

The service exposes the thesis pipeline through HTTP so the AI and Curricula curriculum tool can call it without taking a Python machine learning dependency.

## Scope

The service implements one decision: given a Finnish and an English course description (split into the three Peppi fields: outcomes, contents, assessment), compute the LaBSE cosine similarity and return both the score and the boolean decision against a calibrated threshold.

The preprocessing in `app/preprocessing.py` reproduces Notebook 1 of the thesis exactly. The encoding in `app/inference.py` reproduces the LaBSE branch of Notebook 2 exactly. Any change to those modules must be mirrored in the notebooks or the F1 numbers reported in the thesis no longer apply to the service.

The thresholds in `app/config.py` are **placeholders** based on the LaBSE plateau region described in the thesis (0.70 to 0.85). They must be replaced with the actual `transfer_threshold` values from Notebook 4's `ict_reference` dictionary before the service is used against real Peppi data.

## Requirements

- Python 3.10 or later
- A CUDA-capable GPU is optional but recommended. The service detects CUDA at startup and falls back to CPU if no GPU is present. On CPU, LaBSE inference is roughly an order of magnitude slower per pair but still responsive for interactive use.

## Installation

```bash
git clone <repo-url> bilingual-course-similarity
cd bilingual-course-similarity
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m spacy download fi_core_news_sm
python -m spacy download en_core_web_sm
```

The two spaCy models are the same small models used in Notebook 1 of the thesis. Larger Finnish spaCy models do not exist at the time of writing, so the symmetric small-model choice matches the thesis design.

## Running the service

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The first start-up will take a few seconds while LaBSE loads onto the GPU (or CPU). The `/ready` endpoint returns 503 during that window and 200 once the model is loaded.

## Endpoints

| Method | Path           | Purpose                                    |
|--------|----------------|--------------------------------------------|
| POST   | `/similarity`  | Compute LaBSE cosine similarity for a pair |
| GET    | `/health`      | Liveness check                             |
| GET    | `/ready`       | Readiness check (model loaded)             |
| GET    | `/version`     | Service and model metadata                 |
| GET    | `/docs`        | Interactive Swagger UI                     |
| GET    | `/redoc`       | Alternative ReDoc documentation            |

## Example request

```bash
curl -X POST http://localhost:8000/similarity \
  -H 'Content-Type: application/json' \
  -d '{
    "finnish": {
      "outcomes": "Opiskelija ymmärtää ohjelmistosuunnittelun perusteet.",
      "contents": "Olio-ohjelmointi, luokat, oliot, perintä.",
      "assessment": "Tentti ja harjoitustyö."
    },
    "english": {
      "outcomes": "The student understands the basics of software design.",
      "contents": "Object-oriented programming, classes, objects, inheritance.",
      "assessment": "Exam and assignment."
    },
    "field_configuration": "outcomes_raw"
  }'
```

Example response:

```json
{
  "cosine_similarity": 0.93,
  "threshold": 0.78,
  "is_similar": true,
  "model_name": "sentence-transformers/LaBSE",
  "field_configuration": "outcomes_raw",
  "truncated_fi": false,
  "truncated_en": false,
  "finnish_tokens": 32,
  "english_tokens": 28,
  "inference_ms": 142.7
}
```

The corresponding Python client is in `examples/client.py`.

## The six configurations

| Configuration                       | Fields                                         | Lemmatised |
|-------------------------------------|------------------------------------------------|------------|
| `outcomes_raw`                      | outcomes                                       | No         |
| `outcomes_contents_raw`             | outcomes + contents                            | No         |
| `full_raw`                          | outcomes + contents + assessment               | No         |
| `outcomes_lemmatised`               | outcomes                                       | Yes        |
| `outcomes_contents_lemmatised`      | outcomes + contents                            | Yes        |
| `full_lemmatised`                   | outcomes + contents + assessment               | Yes        |

These map one-to-one onto Notebook 1's `config_1` to `config_6`.

## Audit logging

Every `/similarity` call emits one structured log line containing the request id, the chosen configuration, the Finnish and English token counts, both truncation flags, the cosine similarity, the threshold, the boolean decision, and the server-side inference latency. The request id is also returned in the `X-Request-ID` response header so the curriculum tool can correlate a stored decision against the server log.

## Tests

```bash
pytest
```

The test suite uses FastAPI's `TestClient` and does not require GPU; the model is replaced with a deterministic stub during tests so the suite runs in seconds without downloading LaBSE.

## Known limitations and open work

This is a prototype produced during the AI and Curricula project, not a production service. The items below are documented in the May 12 working notes and are listed here for the next iteration:

- **Operating thresholds are placeholders.** Pull the real per-configuration `transfer_threshold` values from Notebook 4 and update `app/config.py` before any real use.
- **Authentication is not implemented.** The current deployment assumes the service runs inside the JAMK-internal network. Token-based auth has not been added.
- **Boilerplate-aware scoring is not implemented.** Notebook 5 (Section 5.6.4) shows that institutional templates can inflate similarity. A second score that excludes boilerplate would let reviewers see when similarity might be template-driven.
- **Concurrency under load has not been measured.** The model is shared across requests and inference is serialised by the GIL. The current request rate (a human clicking buttons in the curriculum tool) is well within bounds, but no measurement exists.
- **Deployment target is undecided.** Built and tested on Windows with an NVIDIA RTX 5060 Laptop GPU. CPU-only deployment works but is slower.
