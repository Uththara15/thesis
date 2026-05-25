# FastAPI similarity service: handover

Bilingual course description similarity component, exposed as a small HTTP service. Built as the implementation side of the AI and Curricula internship work.

This document is the handover. It says what exists, what works, what doesn't, and what I'd suggest doing next. The full research notes and source list are in `Research_Notes.md` alongside this.

---

## What this is

A FastAPI service that wraps the LaBSE-based bilingual similarity component. The curriculum tool sends a Finnish-English course description pair over HTTP and gets back a similarity score, threshold decision, and diagnostic fields.

The service exists to solve two problems:

1. LaBSE takes about 12 seconds to load. The service loads it once at startup and keeps it in memory, so subsequent calls are fast (~200 ms on GPU).
2. The curriculum tool team doesn't have to take a Python ML dependency. They talk to the service over HTTP with JSON.

## What's in the repository

```
similarity_service/
  app/
    __init__.py
    main.py             FastAPI app, endpoints, lifespan, middleware
    config.py           Model identifier, configurations, thresholds
    schemas.py          Pydantic request/response models
    preprocessing.py    Text cleaning and lemmatisation (mirrors pipeline)
    inference.py        LaBSE loading and cosine computation
  tests/
    conftest.py
    test_service.py     17 unit tests, all passing
  examples/
    client.py           Minimal Python client
    parity_check.py     Verification script
  pyproject.toml
  README.md
  .gitignore
```

Total about 600 lines of source plus tests.

## Endpoints

| Method | Path           | Purpose                                |
|--------|----------------|----------------------------------------|
| POST   | `/similarity`  | Main inference call                    |
| GET    | `/health`      | Liveness check                         |
| GET    | `/ready`       | Readiness check (model loaded)         |
| GET    | `/version`     | Service and model metadata             |
| GET    | `/docs`        | Interactive Swagger UI (auto-generated)|

Sample request to `/similarity`:

```json
{
  "finnish": {
    "outcomes": "Opiskelija ymmärtää ohjelmistosuunnittelun perusteet.",
    "contents": "Olio-ohjelmointi, luokat, oliot, perintä.",
    "assessment": "Tentti."
  },
  "english": {
    "outcomes": "The student understands the basics of software design.",
    "contents": "Object-oriented programming, classes, objects.",
    "assessment": "Exam."
  },
  "field_configuration": "outcomes_raw"
}
```

Sample response:

```json
{
  "cosine_similarity": 0.94,
  "threshold": 0.78,
  "is_similar": true,
  "model_name": "sentence-transformers/LaBSE",
  "field_configuration": "outcomes_raw",
  "truncated_fi": false,
  "truncated_en": false,
  "finnish_tokens": 14,
  "english_tokens": 11,
  "inference_ms": 212.8
}
```

## What works

- Service starts cleanly on GPU or CPU. PyTorch auto-detects.
- LaBSE loaded once at startup via FastAPI's lifespan context manager. ~12 seconds on first launch.
- All 17 tests pass (`pytest -v`). About 20 seconds with the model stubbed out.
- Cosine similarities match the original pipeline's saved embeddings to ~1e-07 (verified by `parity_check.py`).
- Audit logging: every `/similarity` call emits one structured log line with request id, configuration, token counts, truncation flags, similarity, threshold, decision, and latency. Request id also returned in `X-Request-ID` response header.
- Schema validation rejects malformed requests with structured 422 responses.

## What doesn't (intentionally)

These are decisions for the team, not code I should have written:

**Operating thresholds are placeholders.** All six configurations currently use 0.78. The real per-configuration values come from the original pipeline's calibration step. They need to be pulled in before the service is used on real data. The placeholder is clearly commented in `app/config.py`.

**No authentication.** The service assumes it runs on a trusted internal network. If the deployment target is anywhere else, this needs a token or similar.

**No Dockerfile.** Depends on whether the deployment target has a GPU, whether LaBSE is baked into the image or downloaded at startup, and how thresholds are passed in. Once those are decided, the Dockerfile is small (about 10-15 lines).

**Single similarity score.** The earlier work showed institutional template text inflates scores. The service currently returns one cosine. A future version should return two, original and boilerplate-removed, so the reviewer can see when similarity is template-driven rather than content-driven.

## What I'd suggest as next steps

Listed smallest-first. Each is its own task, not a dependency chain.

1. **Pull real thresholds** from the calibration outputs into `app/config.py`. Mechanical change.
2. **Decide deployment target** (CPU/GPU, where it runs, who maintains it).
3. **Add `/version` endpoint fields** for preprocessing version and threshold values. Currently the endpoint returns model info but not the full audit fingerprint needed to reproduce a decision later.
4. **Implement boilerplate-aware second score**. The boilerplate detection logic exists in the original pipeline; moving it into the service preprocessing module would mirror how other functions are shared.
5. **Authentication**, if deployment target requires it.
6. **Dockerfile**, after step 2 is decided.

## How to run

```powershell
conda activate thesis_env
cd similarity_service
pip install -e ".[dev]"
python -m spacy download fi_core_news_sm
python -m spacy download en_core_web_sm
uvicorn app.main:app --port 8000
```

Then `http://localhost:8000/docs` for the interactive UI. `README.md` in the repository has a `curl` example and instructions for the client script.

To run tests:

```powershell
pytest -v
```

To re-run the parity check (requires the original dataset and saved embeddings to be available):

```powershell
python examples\parity_check.py `
  --data-csv "path\to\final_dataset.csv" `
  --embeddings-dir "path\to\embeddings\ict"
```

Writes `parity_report.md` to the current directory.

## Honest scope statement

This is a prototype, not a production service. The design comes from research (see `Research_Notes.md`). The implementation was built with Claude's help, since FastAPI specifically was new to me though I had used Flask with models before. The architectural recommendations should hold up. The code itself would benefit from review by someone with deeper FastAPI experience before deployment.

The parity result is the strongest piece of evidence in the package: the service reproduces the original pipeline's similarity computations to floating-point precision. That means decisions made on top of this service are anchored in the same evaluation work that justified choosing LaBSE in the first place.

## Files in this handover

- `Research_Notes.md` — what I read, what I found, full source list
- `Internship_Handover.md` — this file
- `similarity_service/` — the code, tests, examples
- `parity_report.md` — auto-generated proof of parity with the original pipeline

For anything not covered here, the code is the source of truth.
