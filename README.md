# Evaluating Multilingual Sentence Embedding Models for Bilingual Course Description Similarity in Finnish Higher Education

Bachelor's thesis, JAMK University of Applied Sciences, 2026.
Part of the LuoVuutta strategic project, AI and Curricula subtask.

Live demo: https://huggingface.co/spaces/uththara/course-similarity-demo

---

## Overview

Curriculum management across institutions requires comparing course descriptions written in different languages. Manual comparison is time-consuming and inconsistent. This thesis investigates whether pre-trained multilingual sentence embedding models can reliably detect semantic equivalence between Finnish and English course descriptions, and which model and configuration performs best in this domain.

The project delivers three things:

1. A purpose-built bilingual evaluation dataset of 154 labelled course description pairs from JAMK's ICT degree programme and an independent 72-pair generalisation dataset from the BIT degree programme
2. A reproducible six-notebook Python pipeline covering data collection, embedding generation, evaluation, generalisation testing, extended robustness analysis, and a demo
3. A production-ready FastAPI similarity service (`similarity_service/`) with preprocessing, inference, schemas, tests, and a client example, handed over to JAMK for integration

---

## Live Demo

Try the live demo at: https://huggingface.co/spaces/uththara/course-similarity-demo

Paste any English course description and the app returns the most semantically similar Finnish courses from JAMK's ICT degree programme, ranked by cosine similarity using LaBSE.

---

## Key Results

LaBSE achieved the top four cross-validated F1 positions across all 30 model-configuration combinations tested.

| Model | Best CV F1 | Best AUC | Backbone |
|---|---|---|---|
| LaBSE | 0.976 | 0.992 | LaBSE |
| STSB | 0.952 | 0.988 | XLM-RoBERTa |
| E5 | below STSB | below STSB | XLM-RoBERTa |
| MPNet | below STSB | below STSB | XLM-RoBERTa |
| FinBERT | lowest | lowest | Finnish BERT (monolingual baseline) |

LaBSE also transferred to the independent BIT dataset without fine-tuning, confirming generalisation within a closely related programme area.

---

## Models Evaluated

Five models compared across six text preprocessing configurations (30 combinations total):

| Short Name | Hugging Face Identifier | Type |
|---|---|---|
| LaBSE | sentence-transformers/LaBSE | Multilingual |
| STSB | stsb-xlm-r-multilingual | Multilingual |
| E5 | intfloat/multilingual-e5-base | Multilingual |
| MPNet | paraphrase-multilingual-mpnet-base-v2 | Multilingual |
| FinBERT | TurkuNLP/bert-base-finnish-cased-v1 | Finnish monolingual baseline |

All models produce 768-dimensional embeddings with a 512-token maximum sequence length. Similarity is measured with cosine similarity. Classification uses a threshold swept from 0.00 to 1.00 at 0.01 steps. Model selection used stratified 5-fold cross-validation with bootstrap confidence intervals.

---

## Dataset

Two datasets were constructed from JAMK's Peppi curriculum management system:

**ICT Evaluation Dataset (primary)**
- 154 labelled course description pairs
- 77 equivalent / 77 non-equivalent
- Source: Bachelor's Degree Programme in Information and Communications Technology
- Used for model selection and cross-validation

**BIT Generalisation Dataset (held-out)**
- 72 labelled pairs
- Source: Bachelor's Degree Programme in Business Information Technology, Game Production
- Held out entirely from model selection, used only to test transfer

---

## Text Preprocessing Configurations

Six configurations per model varying two dimensions:

- Field selection: outcomes only / outcomes and contents / full field
- Lemmatisation: raw text vs. lemmatised text

---

## Repository Structure

```
thesis/
|
|-- data collection/
|   |-- jamk_scraper.ipynb               # Peppi course data scraper
|   |-- build_balanced_dataset.py        # ICT dataset builder
|   |-- build_balanced_dataset_bit.py    # BIT dataset builder
|   |-- data_quality_verification.ipynb
|   |-- validate_dataset.py
|
|-- data/
|   |-- raw/
|   |   |-- final_dataset.csv            # ICT evaluation dataset (154 pairs)
|   |   |-- verified_final_dataset_BIT.csv  # BIT generalisation dataset (72 pairs)
|   |-- embeddings/ict/                  # Precomputed .npy embeddings (5 models x 6 configs)
|   |-- processed/                       # Cleaned CSVs, token stats, truncation reports
|   |-- results/
|       |-- summary_tables/              # CV results, full evaluation summary, McNemar tests
|       |-- threshold_sweeps/            # Per-model, per-config threshold sweep CSVs
|       |-- extended_analysis/           # Boilerplate, chunking, truncation experiments
|       |-- validation/                  # BIT generalisation results
|
|-- notebooks/
|   |-- 01_data_preparation.ipynb
|   |-- 02_embedding_generation.ipynb
|   |-- 03_evaluation.ipynb
|   |-- 04_generalization_validation.ipynb
|   |-- 05_extended_analysis.ipynb
|   |-- 06_demo.ipynb
|
|-- similarity_service/                  # Production FastAPI service
|   |-- app/
|   |   |-- main.py                      # FastAPI app
|   |   |-- inference.py                 # Embedding and similarity logic
|   |   |-- preprocessing.py            # Text cleaning and field selection
|   |   |-- schemas.py                   # Request/response schemas
|   |   |-- config.py                    # Model and threshold config
|   |-- examples/
|   |   |-- client.py                    # API client example
|   |   |-- parity_check.py             # Notebook vs service parity validation
|   |-- tests/
|   |   |-- test_service.py
|   |   |-- conftest.py
|   |-- pyproject.toml
|   |-- README.md
|
|-- demo/                                # Lightweight Flask demo (deployed on HF Spaces)
|   |-- app.py
|   |-- static/index.html
|   |-- tietojenkasittely.json
|   |-- Dockerfile
|   |-- requirements.txt
|
|-- figures/
|   |-- main/                            # All result charts used in thesis
|   |-- appendices/
|
|-- docs/
|   |-- methodology_notes.md
|   |-- limitations_tracker.md
|
|-- environment/
|   |-- environment.yml
|   |-- requirements.txt
|
|-- thesis.md                            # Full thesis document
|-- README.md
```

---

## Pipeline Overview

### 1. Data Collection
Scraped Finnish and English course descriptions from JAMK's Peppi curriculum management system. Built balanced datasets with equal numbers of equivalent and non-equivalent pairs. Applied quality verification and translation consistency checks.

### 2. Embedding Generation
Generated 768-dimensional embeddings for all five models across six preprocessing configurations. Precomputed and saved as `.npy` files to avoid redundant computation during evaluation. Recorded token statistics and truncation rates per configuration.

### 3. Evaluation
Swept cosine similarity thresholds from 0.00 to 1.00. Selected optimal threshold per model-configuration combination using stratified 5-fold cross-validation. Computed F1, AUC, precision, recall, and bootstrap confidence intervals. Applied McNemar tests for pairwise statistical comparison.

### 4. Generalisation Validation
Applied ICT-calibrated thresholds to the held-out BIT dataset. Evaluated transfer performance without any BIT-specific tuning. Compared BIT results against ICT cross-validated baselines.

### 5. Extended Robustness Analysis
Tested four additional conditions: error analysis of misclassified pairs, input truncation effects at different sequence lengths, text chunking strategies for long descriptions, and boilerplate text contamination and removal.

### 6. Production Service
Built and packaged a FastAPI similarity service for integration into JAMK's AI and Curricula pipeline. Includes preprocessing, inference, parity validation against notebook results, and a test suite. Handed over with an internship handover document.

---

## Conclusions

LaBSE is the recommended first-candidate model for this context. It produced the strongest cross-validated ICT results and transferred to the BIT dataset without fine-tuning.

Three important qualifications:

1. Model choice alone is not sufficient. Field selection, threshold calibration, and human interpretation are each necessary for reliable deployment.
2. Lemmatisation did not provide a consistent improvement across models.
3. This method is appropriate as a review-support tool that surfaces likely-aligned pairs for human inspection. It is not suitable as an automatic final judge of curriculum equivalence.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/Uththara15/thesis.git
cd thesis

# Set up environment
conda env create -f environment/environment.yml
conda activate thesis_env

# Run notebooks in order
jupyter notebook notebooks/01_data_preparation.ipynb

# Run the production API
cd similarity_service
pip install -e .
uvicorn app.main:app --reload

# Run tests
pytest tests/
```

---

## Tech Stack

- Python 3.10
- sentence-transformers, Hugging Face Transformers
- Scikit-learn (cross-validation, McNemar test, threshold calibration)
- FastAPI (production similarity service)
- Flask (demo app)
- Pandas, NumPy
- Matplotlib, Seaborn
- Docker (HF Spaces deployment)

---

## Context

This thesis was completed as the Bachelor's thesis for the degree of Bachelor of Engineering in Information and Communications Technology at JAMK University of Applied Sciences, Finland, 2026.

The full thesis document including all chapters, result tables, threshold sweep outputs, McNemar test results, BIT generalisation tables, and appendices is available as `thesis.md` in this repository.

---

## Author

Madee Uththara Deegoda Gamage
Bachelor of Engineering in ICT (Data Analytics & AI)
JAMK University of Applied Sciences, Finland

GitHub: https://github.com/Uththara15
LinkedIn: https://linkedin.com/in/uththara15
Demo: https://huggingface.co/spaces/uththara/course-similarity-demo
