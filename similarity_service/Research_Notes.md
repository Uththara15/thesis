# FastAPI service research notes

Notes I kept while researching how to expose the LaBSE-based similarity component as a service, between 6 May and 12 May 2026.

---

## The question

Take the LaBSE-based similarity pipeline and figure out how the curriculum tool can actually call it. The five-notebook pipeline works for evaluation but not for runtime use: every call would reload LaBSE from disk (12 seconds), and the curriculum tool would have to take a heavy Python ML dependency on its side.

## Three approaches considered

### 1. CLI script invoked by subprocess

Curriculum tool runs `python similarity.py --finnish "..." --english "..."` and parses stdout.

Rejected. LaBSE loads from scratch on every invocation. 12 seconds per call is unusable for interactive use. Also fragile: subprocess stderr, exit codes, encoding, all become the curriculum tool's problem.

### 2. Python library

Package the pipeline as a pip-installable library. Curriculum tool imports it directly: `from similarity import predict`.

Rejected. Forces the curriculum tool to be Python and to install PyTorch, sentence-transformers, spaCy, CUDA, etc. That's a lot to push onto another team, and it couples the curriculum tool's deployment to the model's deployment.

### 3. HTTP service

Model lives in one process, in memory, behind an HTTP endpoint. Curriculum tool sends JSON, gets JSON back.

Selected. Standard pattern for serving ML models. Language-agnostic. Model loads once. Clean contract between systems. Independently deployable.

## Framework choice

### Flask

- Lightweight, easy, I've used it before.
- Synchronous by default. Validation has to be written manually.
- Auto-generated docs only via add-on extensions (flask-restx, flasgger).
- Good fit for simple form-input model demos. Less good for an API the curriculum tool will rely on long-term.

### Django REST Framework

- Heavy. Brings the whole Django ecosystem.
- Excellent if there's also user management, admin panels, database models around it.
- For a single inference endpoint, disproportionate.

### FastAPI

- Built on Starlette (async HTTP) and Pydantic (validation).
- Request and response validation built in via type annotations.
- Auto-generated interactive docs at `/docs` (Swagger UI) and `/redoc`.
- Documented lifespan mechanism for loading expensive resources once at startup.
- Standard choice for ML model serving now.

Chose FastAPI. The three things I get for free (validation, docs, lifespan) would all be manual work in Flask.

## What I learned that mattered

### Lifespan is non-negotiable

The single most important pattern in serving an ML model with FastAPI. The lifespan context manager loads the model before the application accepts requests and keeps it in memory for the lifetime of the service.

The official documentation example uses an ML model load as the canonical use case (Tiangolo, n.d.-a). The pattern is:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["labse"] = load_labse()
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)
```

Everything before `yield` runs at startup, after `yield` runs at shutdown. The framework guarantees the model is loaded before the first request.

If a future implementation skips this and loads the model per request, LaBSE's 12-second load makes the service unusable. The older `@app.on_event("startup")` decorator works but is deprecated in favour of `lifespan` (Tiangolo, n.d.-a; PythonForDeveloper, 2025).

### Preprocessing must stay in sync

If the service preprocesses text differently from the original pipeline, even slightly, the evaluation numbers stop applying to the deployed system. The fix is structural: the service imports the same preprocessing functions, not a reimplementation. This way, changes in one place propagate to both.

### Pydantic v2 syntax

FastAPI uses Pydantic v2. Key differences from v1 that caught me out:

- `model_config = ConfigDict(...)` replaces `class Config:`
- `@field_validator` replaces `@validator`
- `model_dump()` replaces `dict()`
- Performance is 5 to 50x faster than v1 because of a Rust-based core (OneUptime, 2026).

The framework returns 422 with structured error messages automatically when validation fails. No manual error handling needed for type mismatches, missing fields, out-of-range values, etc. (PyTutorial, 2025; Rana, 2025).

### Liveness versus readiness

Two separate concerns. Liveness = process is up. Readiness = service can actually answer requests (model is loaded). They are different because the process accepts socket connections well before LaBSE finishes loading on the GPU. If a load balancer only checks `/health` (liveness), it routes traffic to the service before it can answer, and the first few users get errors.

Standard pattern: `/health` for liveness, `/ready` for readiness. Industry-standard convention from Kubernetes deployments, also applicable to simpler setups.

### Audit logging

For any decision the curriculum tool stores, the corresponding server-side log line needs enough information to reproduce it: request id, input fingerprint, model version, threshold, output. Without that, "why did the system say these courses were similar in May?" is unanswerable in November.

I added a request id middleware (uuid4 per request, echoed in the `X-Request-ID` response header) and one structured log line per `/similarity` call. The curriculum tool can store the request id alongside its decisions and any audit can correlate the two.

### The boilerplate problem

A finding from my own earlier work: institutional templates in course descriptions inflate similarity scores. Two genuinely different courses can score >0.9 just because both descriptions share JAMK's standard template phrasing. A single similarity number doesn't surface this. A two-score response (original + boilerplate-removed) would let a reviewer in the curriculum tool see when similarity is content-driven versus template-driven.

I left this as a suggestion rather than implementing it, because the boilerplate removal logic has policy choices that should not be decided unilaterally (which phrases count as boilerplate, what threshold for "frequent enough to remove", etc.).

## Quick verification of design

I built a prototype and ran a parity check: six representative pairs from the dataset, comparing the cosine the service returns against the cosine computed directly from saved embeddings. Max difference across the six pairs was about 1.2e-07. That's single-precision float rounding noise, not a real difference. So the design holds: the service reproduces the original pipeline's output exactly.

Prototype was 12 source files, about 600 lines, with 17 unit tests passing. Built with Claude's help on the boilerplate code; design and parity verification are from the research above.

## What I'd add in a production version

Listed in the order I'd do them. Each is small in isolation.

1. Pull the real per-configuration thresholds into the service config (currently placeholders).
2. Add `/version` endpoint returning model identifier, preprocessing version, threshold values — needed for audit trail.
3. Add the boilerplate-aware second score to the response schema.
4. Authentication (currently none, assumes trusted internal network).
5. Dockerfile, after deployment target is decided.

---

## Sources

All URLs accessed between 6 and 12 May 2026.

**FastAPI documentation**

- Tiangolo, S. (n.d.-a). *Lifespan events*. FastAPI documentation. https://fastapi.tiangolo.com/advanced/events/
- Tiangolo, S. (n.d.-b). *Body — Multiple parameters*. FastAPI documentation. https://fastapi.tiangolo.com/tutorial/body/
- Tiangolo, S. (n.d.-c). *Path parameters and numeric validations*. FastAPI documentation. https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/

**Pydantic v2 documentation**

- Pydantic Services Inc. (n.d.). *Pydantic v2 documentation*. https://docs.pydantic.dev/latest/

**LaBSE model**

- Sentence Transformers. (n.d.). *LaBSE model card*. Hugging Face. https://huggingface.co/sentence-transformers/LaBSE
- Sentence Transformers. (n.d.). *sentence-transformers documentation*. https://huggingface.co/sentence-transformers
- Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2022). Language-agnostic BERT sentence embedding. *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics* (Volume 1: Long Papers), 878–891. https://doi.org/10.18653/v1/2022.acl-long.62

**FastAPI for ML**

- JetBrains. (2026, March). *How to use FastAPI for machine learning*. PyCharm blog. https://blog.jetbrains.com/pycharm/2024/09/how-to-use-fastapi-for-machine-learning/
- ApXML. (n.d.). *Loading models into FastAPI applications*. https://apxml.com/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models/loading-models-fastapi
- Malhotra, V. (2025, November 28). *FastAPI — Lifespan: Scenario*. Medium. https://medium.com/@vipulm124/fastapi-lifespan-bbdd7c32c6c4
- PythonForDeveloper. (2025, October 25). *How to use lifespan in FastAPI?* https://pythonfordeveloper.com/how-to-use-lifespan-in-fastapi/

**Pydantic v2 tutorials**

- Rana, B. (2025, October 17). *FastAPI + Pydantic v2: Validation without the drag*. Medium. https://medium.com/@bhagyarana80/fastapi-pydantic-v2-validation-without-the-drag-17fe1f1771e2
- OneUptime. (2026, January 21). *How to validate data with Pydantic v2 models*. https://oneuptime.com/blog/post/2026-01-21-python-pydantic-v2-validation/view
- PyTutorial. (2025, December 1). *FastAPI validation with Pydantic models*. https://pytutorial.com/fastapi-validation-with-pydantic-models/

**Framework comparison**

- Codecademy. (n.d.). *FastAPI vs Flask: Key differences, performance, and use cases*. https://www.codecademy.com/article/fastapi-vs-flask-key-differences-performance-and-use-cases
- ProjectPro. (2024, October 28). *Python FastAPI vs. Flask for machine learning projects*. https://www.projectpro.io/article/fastapi-vs-flask/652
- Strapi. (2025). *FastAPI vs Flask 2025: Performance, speed & when to choose*. https://strapi.io/blog/fastapi-vs-flask-python-framework-comparison
- Imarticus. (2025, July 2). *Flask vs FastAPI: Which is better for deploying ML models?* https://imarticus.org/blog/flask-vs-fastapi-which-is-better-for-deploying-ml-models/
