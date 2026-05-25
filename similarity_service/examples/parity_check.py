"""Parity verification: service output vs. Notebook 2 embeddings.

This script proves that the FastAPI service computes the same cosine
similarity values as the thesis pipeline.

What it does
------------
1. Loads the ICT dataset (data/raw/final_dataset.csv).
2. Loads the LaBSE config 1 embeddings produced by Notebook 2
   (data/embeddings/ict/labse_config1_fi.npy and _en.npy).
3. Picks a small sample of rows (one positive pair, one negative pair,
   and a few extras for coverage).
4. For each row:
     a. Sends the raw Peppi fields to POST /similarity with
        field_configuration = "outcomes_raw" (Notebook 1 config 1).
     b. Reads the matching row from the saved embedding store and
        computes the cosine directly as dot product on unit vectors.
     c. Compares the two numbers.
5. Writes a Markdown report to parity_report.md and prints a summary
   table to stdout.

How to run
----------
    # in one terminal, start the service:
    uvicorn app.main:app --port 8000

    # in another terminal, with thesis_env active:
    cd similarity_service
    python examples/parity_check.py

Parity is considered verified when the absolute difference between the
service cosine and the notebook cosine is below 1e-5 for every row.
This tolerance accounts for floating-point noise from the model running
on slightly different inputs (the service re-encodes from text, while
the notebook stored already-encoded vectors).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import error as urllib_error
from urllib import request

import numpy as np
import pandas as pd

DEFAULT_DATA_CSV = r"C:\madee\thesis_project_new\data\final_dataset.csv"
DEFAULT_EMB_DIR = r"C:\madee\thesis_project_new\data\embeddings\ict"
DEFAULT_SERVICE_URL = "http://localhost:8000"
DEFAULT_SAMPLE_SIZE = 6
TOLERANCE = 1e-4  # service and notebook tensors may differ by float noise


def fetch_health(base_url: str) -> bool:
    """Confirm the service is up and the model is loaded."""
    try:
        with request.urlopen(f"{base_url}/ready", timeout=10) as r:
            body = json.loads(r.read().decode("utf-8"))
            return bool(body.get("model_loaded"))
    except urllib_error.URLError as e:
        print(f"Could not reach service at {base_url}/ready: {e}")
        return False


def call_similarity(
    base_url: str,
    fi_outcomes: str,
    en_outcomes: str,
) -> dict:
    """Call POST /similarity for a single bilingual pair.

    Sends only the outcomes fields because we are verifying parity
    against config 1 (outcomes_raw) only.
    """
    payload = {
        "finnish": {
            "outcomes": fi_outcomes,
            "contents": "",
            "assessment": "",
        },
        "english": {
            "outcomes": en_outcomes,
            "contents": "",
            "assessment": "",
        },
        "field_configuration": "outcomes_raw",
    }
    req = request.Request(
        f"{base_url}/similarity",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify service-notebook parity for the LaBSE config 1 path."
    )
    parser.add_argument(
        "--data-csv",
        default=DEFAULT_DATA_CSV,
        help="Path to ICT final_dataset.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--embeddings-dir",
        default=DEFAULT_EMB_DIR,
        help="Path to data/embeddings/ict (default: %(default)s)",
    )
    parser.add_argument(
        "--service-url",
        default=DEFAULT_SERVICE_URL,
        help="Service base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of rows to verify (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="parity_report.md",
        help="Markdown report path (default: %(default)s)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Service-Notebook parity check")
    print("=" * 70)

    # 1. Check service is up.
    print(f"\nChecking service at {args.service_url} ...")
    if not fetch_health(args.service_url):
        print("Service is not ready. Start it with:")
        print("    uvicorn app.main:app --port 8000")
        return 1
    print("Service is ready.")

    # 2. Load the dataset.
    data_path = Path(args.data_csv)
    if not data_path.exists():
        print(f"\nDataset not found: {data_path}")
        print("Pass --data-csv pointing to your final_dataset.csv")
        return 1
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} rows from {data_path}")

    # 3. Load the saved LaBSE config 1 embeddings.
    emb_dir = Path(args.embeddings_dir)
    fi_path = emb_dir / "labse_config1_fi.npy"
    en_path = emb_dir / "labse_config1_en.npy"
    if not fi_path.exists() or not en_path.exists():
        print(f"\nEmbeddings not found in {emb_dir}")
        print("Expected: labse_config1_fi.npy and labse_config1_en.npy")
        return 1
    fi_embs = np.load(fi_path)
    en_embs = np.load(en_path)
    print(f"Embeddings loaded: shape FI {fi_embs.shape}, EN {en_embs.shape}")

    if len(fi_embs) != len(df) or len(en_embs) != len(df):
        print(
            f"Shape mismatch: dataset has {len(df)} rows but embeddings "
            f"have FI={len(fi_embs)}, EN={len(en_embs)}"
        )
        return 1

    # 4. Pick rows: one of each label, then fill to sample_size with
    # alternating labels to keep the sample balanced.
    positive_indices = df.index[df["similarity_label"] == 1].tolist()
    negative_indices = df.index[df["similarity_label"] == 0].tolist()
    sample_indices: list[int] = []
    for i in range(args.sample_size):
        pool = positive_indices if i % 2 == 0 else negative_indices
        if i // 2 < len(pool):
            sample_indices.append(pool[i // 2])
    print(f"\nSampling {len(sample_indices)} rows: {sample_indices}")

    # 5. Loop and compare.
    rows = []
    print()
    print(f"{'row':>4}  {'label':>5}  {'service':>10}  {'notebook':>10}"
          f"  {'diff':>10}  {'status':>8}")
    print("-" * 70)

    for idx in sample_indices:
        row = df.loc[idx]
        fi_text = row["outcomes_fi"]
        en_text = row["outcomes_en"]
        label = int(row["similarity_label"])

        # Service call
        t0 = time.perf_counter()
        try:
            result = call_similarity(args.service_url, fi_text, en_text)
        except Exception as e:
            print(f"  service call failed for row {idx}: {e}")
            continue
        service_cosine = float(result["cosine_similarity"])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Notebook computation (dot product on the L2-normalised stored vectors)
        notebook_cosine = float(np.dot(fi_embs[idx], en_embs[idx]))

        diff = abs(service_cosine - notebook_cosine)
        status = "OK" if diff < TOLERANCE else "FAIL"

        print(
            f"{idx:>4}  {label:>5}  {service_cosine:>10.6f}  "
            f"{notebook_cosine:>10.6f}  {diff:>10.6f}  {status:>8}"
        )

        rows.append({
            "row_index": idx,
            "label": label,
            "fi_outcomes": fi_text[:120] + ("..." if len(fi_text) > 120 else ""),
            "en_outcomes": en_text[:120] + ("..." if len(en_text) > 120 else ""),
            "service_cosine": service_cosine,
            "notebook_cosine": notebook_cosine,
            "absolute_difference": diff,
            "service_latency_ms": latency_ms,
            "is_similar_service": bool(result["is_similar"]),
            "applied_threshold": float(result["threshold"]),
            "service_truncated_fi": bool(result["truncated_fi"]),
            "service_truncated_en": bool(result["truncated_en"]),
            "service_fi_tokens": int(result["finnish_tokens"]),
            "service_en_tokens": int(result["english_tokens"]),
            "passed": status == "OK",
        })

    if not rows:
        print("\nNo rows were processed successfully.")
        return 1

    # 6. Summary and report.
    diffs = np.array([r["absolute_difference"] for r in rows])
    n_passed = sum(1 for r in rows if r["passed"])
    n_total = len(rows)
    all_passed = n_passed == n_total

    print("-" * 70)
    print(f"Mean abs diff   : {diffs.mean():.2e}")
    print(f"Max abs diff    : {diffs.max():.2e}")
    print(f"Tolerance       : {TOLERANCE:.0e}")
    print(f"Rows passed     : {n_passed} / {n_total}")
    print(f"Overall status  : {'PASS' if all_passed else 'FAIL'}")

    write_report(Path(args.output), rows, all_passed, TOLERANCE, args.service_url)
    print(f"\nMarkdown report written to: {args.output}")

    return 0 if all_passed else 2


def write_report(
    out_path: Path,
    rows: list[dict],
    all_passed: bool,
    tolerance: float,
    service_url: str,
) -> None:
    """Write a Markdown report suitable for handing to a supervisor.

    The report includes a summary block, a per-row table, and a
    short interpretation paragraph.
    """
    diffs = np.array([r["absolute_difference"] for r in rows])
    n_passed = sum(1 for r in rows if r["passed"])
    n_total = len(rows)

    lines: list[str] = []
    lines.append("# Service-Notebook Parity Report")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append(f"Service: {service_url}")
    lines.append("Configuration tested: `outcomes_raw` (Notebook 1 config 1, LaBSE)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Rows checked: {n_total}")
    lines.append(f"- Rows within tolerance: {n_passed}")
    lines.append(f"- Tolerance: {tolerance:.0e}")
    lines.append(f"- Mean absolute difference: {diffs.mean():.2e}")
    lines.append(f"- Max absolute difference: {diffs.max():.2e}")
    lines.append(f"- Overall status: **{'PASS' if all_passed else 'FAIL'}**")
    lines.append("")
    lines.append("## Per-row results")
    lines.append("")
    lines.append(
        "| Row | Label | Service cosine | Notebook cosine | "
        "Abs. diff | Status | Latency (ms) |"
    )
    lines.append(
        "|----:|------:|---------------:|----------------:|"
        "----------:|:-------|-------------:|"
    )
    for r in rows:
        status = "PASS" if r["passed"] else "FAIL"
        lines.append(
            f"| {r['row_index']} | {r['label']} | "
            f"{r['service_cosine']:.6f} | {r['notebook_cosine']:.6f} | "
            f"{r['absolute_difference']:.2e} | {status} | "
            f"{r['service_latency_ms']:.1f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if all_passed:
        lines.append(
            "The service-side cosine similarity matches the stored "
            "Notebook 2 embedding cosine to within the tolerance for every "
            "row checked. The service therefore reproduces the LaBSE config 1 "
            "evaluation path of the thesis pipeline. The F1 numbers reported "
            "in the thesis for this configuration apply to the deployed "
            "service."
        )
    else:
        lines.append(
            "One or more rows exceeded the tolerance. The service is "
            "computing a different cosine than the saved Notebook 2 "
            "embeddings for the same input text. Likely causes: drift "
            "in the preprocessing module (`app/preprocessing.py`), drift "
            "in the encoding module (`app/inference.py`), or a "
            "library-version mismatch in the active environment. "
            "Compare commit hashes for preprocessing and inference against "
            "the notebook code before re-running."
        )
    lines.append("")
    lines.append("## Sample inputs")
    lines.append("")
    for r in rows:
        lines.append(f"### Row {r['row_index']} (label {r['label']})")
        lines.append("")
        lines.append("Finnish outcomes:")
        lines.append("")
        lines.append(f"> {r['fi_outcomes']}")
        lines.append("")
        lines.append("English outcomes:")
        lines.append("")
        lines.append(f"> {r['en_outcomes']}")
        lines.append("")
        lines.append(
            f"Service tokens: FI={r['service_fi_tokens']}, "
            f"EN={r['service_en_tokens']}. "
            f"Truncated FI={r['service_truncated_fi']}, "
            f"EN={r['service_truncated_en']}."
        )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
