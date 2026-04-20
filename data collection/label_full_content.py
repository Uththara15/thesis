"""
Universal Full Content Labelling Tool
Thesis: Evaluating Multilingual Sentence Embedding Models
        for Bilingual Course Description Similarity in Finnish Higher Education
Author: Madee Uththara Deegoda Gamage
JAMK University of Applied Sciences

Works with ANY validated_dataset.csv regardless of degree programme.
Automatically detects available fields and shows all non-empty ones.

Usage:
    python label_full_content.py --input validated_dataset.csv
    python label_full_content.py --input validated_dataset.csv --stats
    python label_full_content.py --input validated_dataset.csv --name "BIT"

Controls:
    1   → Equivalent
    0   → Different
    s   → Skip (come back later)
    b   → Go back one pair
    q   → Save and quit
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

PROGRESS_FILE = "labelling_full_progress.json"
OUTPUT_FILE   = "labelled_full_dataset.csv"

# All recognised field pairs — uses whichever exist in the dataset
FIELD_PAIRS = [
    ("outcomes_fi",   "outcomes_en",   "OUTCOMES"),
    ("contents_fi",   "contents_en",   "CONTENTS"),
    ("assessment_fi", "assessment_en", "ASSESSMENT"),
]

# Possible course ID column names
POSSIBLE_ID_COLS = ["course_id", "course_code", "id", "code"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def detect_id_col(df: pd.DataFrame) -> str:
    for col in POSSIBLE_ID_COLS:
        if col in df.columns:
            return col
    sys.exit(
        f"ERROR: Cannot find course ID column.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Expected one of: {POSSIBLE_ID_COLS}"
    )

def detect_available_fields(df: pd.DataFrame) -> list:
    available = []
    for fi_col, en_col, label in FIELD_PAIRS:
        if fi_col in df.columns and en_col in df.columns:
            available.append((fi_col, en_col, label))
    return available

def load_progress() -> dict:
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

def wrap(text: str, width: int = 72) -> str:
    if not isinstance(text, str) or not text.strip():
        return "[empty]"
    text = text[:500] + "..." if len(text) > 500 else text
    words = text.split()
    lines = []
    current = []
    current_len = 0
    for word in words:
        if current_len + len(word) + 1 > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return "\n     ".join(lines)

def save_output(df: pd.DataFrame, id_col: str, progress: dict):
    df_out = df.copy()
    df_out["similarity_label"] = df_out[id_col].map(
        lambda x: progress.get(x, {}).get("label", "")
    )
    df_out["label_note"] = df_out[id_col].map(
        lambda x: progress.get(x, {}).get("note", "")
    )
    df_out["label_timestamp"] = df_out[id_col].map(
        lambda x: progress.get(x, {}).get("timestamp", "")
    )
    df_out["pair_type"] = "real"
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# ── Display ────────────────────────────────────────────────────────────────────

def print_header(labelled: int, total: int, positive: int,
                 negative: int, dataset_name: str):
    clear()
    print("=" * 70)
    print("  FULL CONTENT LABELLING TOOL")
    print(f"  Dataset: {dataset_name}")
    print("=" * 70)
    bar_filled = int((labelled / max(total, 1)) * 40)
    bar = "X" * bar_filled + "." * (40 - bar_filled)
    print(f"  [{bar}] {labelled}/{total}")
    print(f"  Label 1: {positive}  |  Label 0: {negative}  |  Remaining: {total - labelled}")
    print("=" * 70)

def print_criteria():
    print("""
  LABELLING CRITERIA:
  1 = EQUIVALENT  Same skills, topics and assessment — minor wording OK
  0 = DIFFERENT   Any meaningful difference a curriculum manager should know

  BORDERLINE RULE: Would I flag this for a curriculum manager? YES=0  NO=1
""")

def print_pair(row: pd.Series, id_col: str, available_fields: list,
               idx: int, total: int):
    code     = row[id_col]
    title_fi = row.get("title_fi", "")
    title_en = row.get("title_en", "")
    credits  = row.get("credits", "")

    print(f"\n  Course {idx + 1} of {total} -- {code}  ({credits} cr)")
    print(f"  FI: {title_fi}")
    print(f"  EN: {title_en}")
    print()

    for fi_col, en_col, field_label in available_fields:
        fi_text = str(row.get(fi_col, "")).strip()
        en_text = str(row.get(en_col, "")).strip()

        if not fi_text and not en_text:
            continue

        print(f"  {'-'*66}")
        print(f"  FINNISH {field_label}:")
        print(f"     {wrap(fi_text if fi_text else '[empty]')}")
        print()
        print(f"  ENGLISH {field_label}:")
        print(f"     {wrap(en_text if en_text else '[empty]')}")
        print()

def get_input():
    print("-" * 70)
    print("  1 = Equivalent | 0 = Different | s = Skip | b = Back | q = Quit")
    print("-" * 70)
    return input("  Your label: ").strip().lower()

# ── Statistics ─────────────────────────────────────────────────────────────────

def print_stats(progress: dict, total: int, dataset_name: str):
    labels    = [v["label"] for v in progress.values()
                 if "label" in v and v["label"] in ("0", "1")]
    positive  = labels.count("1")
    negative  = labels.count("0")
    skipped   = sum(1 for v in progress.values() if v.get("label") == "skipped")
    remaining = total - len(labels) - skipped

    print("\n" + "=" * 70)
    print(f"  LABELLING STATISTICS -- {dataset_name}")
    print("=" * 70)
    print(f"  Total pairs     : {total}")
    print(f"  Labelled        : {len(labels)}")
    print(f"  Equivalent (1)  : {positive}  ({positive/max(len(labels),1)*100:.1f}%)")
    print(f"  Different  (0)  : {negative}  ({negative/max(len(labels),1)*100:.1f}%)")
    print(f"  Skipped         : {skipped}")
    print(f"  Remaining       : {remaining}")
    print("=" * 70)

    if remaining == 0 and skipped == 0:
        print(f"\n  ALL PAIRS LABELLED -- {OUTPUT_FILE} is ready!\n")
    else:
        print(f"\n  Progress saved. Re-run to continue.\n")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Universal full content labelling tool for any JAMK dataset."
    )
    parser.add_argument("--input",  required=True, help="Path to validated_dataset.csv")
    parser.add_argument("--stats",  action="store_true", help="Show stats only, do not label")
    parser.add_argument("--name",   default="", help="Optional dataset name for display")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: File not found: {input_path}")

    df    = pd.read_csv(input_path, dtype=str).fillna("")
    total = len(df)

    # Auto-detect column names
    id_col           = detect_id_col(df)
    available_fields = detect_available_fields(df)
    dataset_name     = args.name if args.name else input_path.stem

    print(f"\n  Loaded       : {total} courses from {input_path}")
    print(f"  ID column    : {id_col}")
    print(f"  Fields found : {[f[2] for f in available_fields]}")
    print(f"  Dataset      : {dataset_name}")

    if not available_fields:
        sys.exit(
            "ERROR: No recognised field pairs found.\n"
            f"Available columns: {list(df.columns)}\n"
            "Expected: outcomes_fi/en, contents_fi/en, assessment_fi/en"
        )

    progress = load_progress()

    if args.stats:
        print_stats(progress, total, dataset_name)
        save_output(df, id_col, progress)
        return

    done = sum(1 for v in progress.values() if "label" in v)
    if done > 0:
        print(f"\n  Resuming -- {done} pairs already labelled.")

    input("\n  Press Enter to start labelling...")

    unlabelled = [
        i for i, row in df.iterrows()
        if df.loc[i, id_col] not in progress
        or "label" not in progress.get(df.loc[i, id_col], {})
    ]

    history = []
    ptr = 0

    while ptr < len(unlabelled):
        df_idx = unlabelled[ptr]
        row    = df.loc[df_idx]
        code   = row[id_col]

        labels   = [v["label"] for v in progress.values()
                    if "label" in v and v["label"] in ("0", "1")]
        positive = labels.count("1")
        negative = labels.count("0")

        print_header(len(labels), total, positive, negative, dataset_name)
        print_criteria()
        print_pair(row, id_col, available_fields, len(labels), total)

        decision = get_input()

        if decision == "q":
            print("\n  Saving and quitting...")
            break

        if decision == "b":
            if history:
                prev = history.pop()
                if prev in progress:
                    del progress[prev]
                    save_progress(progress)
                if ptr > 0:
                    ptr -= 1
            else:
                print("  Nothing to go back to.")
                input("  Press Enter to continue...")
            continue

        if decision == "s":
            progress[code] = {
                "label":     "skipped",
                "timestamp": datetime.now().isoformat()
            }
            save_progress(progress)
            save_output(df, id_col, progress)
            history.append(code)
            ptr += 1
            continue

        if decision in ("0", "1"):
            note = input("  Optional note (Enter to skip): ").strip()
            progress[code] = {
                "label":     decision,
                "note":      note,
                "timestamp": datetime.now().isoformat()
            }
            save_progress(progress)
            save_output(df, id_col, progress)
            history.append(code)
            ptr += 1
            continue

        print("  Invalid input. Enter 1, 0, s, b, or q.")
        input("  Press Enter to try again...")

    clear()
    print_stats(progress, total, dataset_name)
    save_output(df, id_col, progress)

if __name__ == "__main__":
    main()
