"""
Universal Full Content Verification Tool
Thesis: Evaluating Multilingual Sentence Embedding Models
        for Bilingual Course Description Similarity in Finnish Higher Education
Author: Madee Uththara Deegoda Gamage
JAMK University of Applied Sciences

Works with any balanced dataset regardless of degree programme.
Verifies all constructed pairs (easy + hard negatives).
Shows all available fields: outcomes + contents + assessment.
Real pairs are kept unchanged.

Usage:
    python verify_constructed.py --input bit_balanced_dataset.csv
    python verify_constructed.py --input final_dataset_outcome.csv
    python verify_constructed.py --input bit_balanced_dataset.csv --stats

Controls:
    c   -> Confirm Label 0 is correct
    1   -> Change to Label 1 (surprisingly similar)
    r   -> Remove this pair entirely
    s   -> Skip (come back later)
    b   -> Go back one pair
    q   -> Save and quit
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

PROGRESS_FILE = "verification_progress.json"
OUTPUT_FILE   = "verified_final_dataset.csv"
MAX_CHARS     = 400

# All recognised field pairs
FIELD_PAIRS = [
    ("outcomes_fi",   "outcomes_en",   "OUTCOMES"),
    ("contents_fi",   "contents_en",   "CONTENTS"),
    ("assessment_fi", "assessment_en", "ASSESSMENT"),
]

# Possible course ID column names
POSSIBLE_ID_COLS = ["course_id", "course_code", "id", "code"]

LABEL_COL = "similarity_label"

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

def wrap(text: str, width: int = 70) -> str:
    if not isinstance(text, str) or not text.strip():
        return "[empty]"
    text = text[:MAX_CHARS] + "..." if len(text) > MAX_CHARS else text
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

def save_output(df_real: pd.DataFrame, df_constructed: pd.DataFrame,
                progress: dict, id_col: str):
    df_verified = df_constructed.copy()
    rows_to_remove = []

    for idx, row in df_verified.iterrows():
        code = row[id_col]
        if code in progress:
            action = progress[code].get("action")
            if action == "remove":
                rows_to_remove.append(idx)
            elif action == "correct":
                df_verified.at[idx, LABEL_COL] = progress[code]["new_label"]
                df_verified.at[idx, "label_note"] = (
                    str(row.get("label_note", "")) +
                    f" | VERIFIED: changed to {progress[code]['new_label']}"
                )
            elif action == "confirm":
                df_verified.at[idx, "label_note"] = (
                    str(row.get("label_note", "")) + " | VERIFIED: confirmed"
                )

    df_verified = df_verified.drop(index=rows_to_remove)
    df_final    = pd.concat([df_real, df_verified], ignore_index=True)
    df_final    = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# ── Display ────────────────────────────────────────────────────────────────────

def print_header(verified: int, total: int, confirmed: int,
                 corrected: int, removed: int, dataset_name: str):
    clear()
    print("=" * 70)
    print("  FULL CONTENT VERIFICATION TOOL")
    print(f"  Dataset: {dataset_name}")
    print("=" * 70)
    bar_filled = int((verified / max(total, 1)) * 40)
    bar = "X" * bar_filled + "." * (40 - bar_filled)
    print(f"  [{bar}] {verified}/{total}")
    print(f"  Confirmed: {confirmed}  |  Corrected: {corrected}  |  Removed: {removed}")
    print("=" * 70)

def print_criteria():
    print("""
  ALL CONSTRUCTED PAIRS SHOULD BE LABEL 0:
  c = CONFIRM  Label 0 correct, clearly different courses
  1 = CORRECT  Change to Label 1, content surprisingly similar
  r = REMOVE   Remove pair, too ambiguous to use
""")

def print_pair(row: pd.Series, id_col: str, available_fields: list,
               idx: int, total: int):
    code       = row[id_col]
    title_fi   = row.get("title_fi", "")
    title_en   = row.get("title_en", "")
    difficulty = row.get("pair_difficulty", "unknown")
    note       = row.get("label_note", "")

    diff_label = {
        "easy_negative": "EASY NEGATIVE (distant groups)",
        "hard_negative": "HARD NEGATIVE (same group)"
    }.get(difficulty, difficulty.upper())

    print(f"\n  Pair {idx + 1} of {total} -- {code}")
    print(f"  Type : {diff_label}")
    print(f"  Note : {note}")
    print(f"  FI   : {title_fi}")
    print(f"  EN   : {title_en}")
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
    print("  c = Confirm(0) | 1 = Change to Positive | r = Remove")
    print("  s = Skip | b = Back | q = Save and Quit")
    print("-" * 70)
    return input("  Your decision: ").strip().lower()

# ── Statistics ─────────────────────────────────────────────────────────────────

def print_final_stats(df_real, df_constructed, progress, dataset_name):
    confirmed = sum(1 for v in progress.values() if v.get("action") == "confirm")
    corrected = sum(1 for v in progress.values() if v.get("action") == "correct")
    removed   = sum(1 for v in progress.values() if v.get("action") == "remove")
    skipped   = sum(1 for v in progress.values() if v.get("action") == "skip")
    remaining = len(df_constructed) - len(progress)
    kept      = len(df_constructed) - removed
    total     = len(df_real) + kept

    print("\n" + "=" * 70)
    print(f"  VERIFICATION SUMMARY -- {dataset_name}")
    print("=" * 70)
    print(f"  Constructed pairs total : {len(df_constructed)}")
    print(f"  Confirmed (0)           : {confirmed}")
    print(f"  Corrected to (1)        : {corrected}")
    print(f"  Removed                 : {removed}")
    print(f"  Skipped                 : {skipped}")
    print(f"  Remaining               : {remaining}")
    print(f"\n  Final dataset estimate:")
    print(f"  Real pairs              : {len(df_real)}")
    print(f"  Verified kept           : {kept}")
    print(f"  Total                   : {total}")

    if remaining == 0:
        print(f"\n  ALL PAIRS VERIFIED -- {OUTPUT_FILE} is ready!")
    else:
        print(f"\n  Progress saved. Re-run to continue.")
    print("=" * 70)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Universal verification tool for any JAMK balanced dataset."
    )
    parser.add_argument("--input",  required=True, help="Path to balanced dataset CSV")
    parser.add_argument("--stats",  action="store_true", help="Show stats only")
    parser.add_argument("--name",   default="", help="Optional dataset name for display")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: File not found: {input_path}")

    df = pd.read_csv(input_path, dtype=str).fillna("")

    # Auto-detect columns
    id_col           = detect_id_col(df)
    available_fields = detect_available_fields(df)
    dataset_name     = args.name if args.name else input_path.stem

    # Split real vs constructed
    df_real        = df[df["pair_type"] == "real"].copy()
    df_constructed = df[df["pair_type"] == "constructed"].copy()

    # Sort: easy negatives first then hard negatives
    df_constructed = df_constructed.sort_values(
        "pair_difficulty",
        key=lambda x: x.map({"easy_negative": 0, "hard_negative": 1})
    ).reset_index(drop=True)

    n_easy = len(df_constructed[df_constructed["pair_difficulty"] == "easy_negative"])
    n_hard = len(df_constructed[df_constructed["pair_difficulty"] == "hard_negative"])

    print(f"\n  Dataset      : {dataset_name}")
    print(f"  Total pairs  : {len(df)}")
    print(f"  Real pairs   : {len(df_real)}")
    print(f"  Easy negatives to verify : {n_easy}")
    print(f"  Hard negatives to verify : {n_hard}")
    print(f"  Fields found : {[f[2] for f in available_fields]}")

    if args.stats:
        progress = load_progress()
        print_final_stats(df_real, df_constructed, progress, dataset_name)
        save_output(df_real, df_constructed, progress, id_col)
        return

    progress = load_progress()
    done = len(progress)
    if done > 0:
        print(f"\n  Resuming -- {done} pairs already verified.")

    input("\n  Press Enter to start -- easy negatives first, then hard negatives...")

    unverified = [
        i for i, row in df_constructed.iterrows()
        if row[id_col] not in progress
    ]

    history = []
    ptr = 0

    while ptr < len(unverified):
        df_idx = unverified[ptr]
        row    = df_constructed.loc[df_idx]
        code   = row[id_col]

        verified  = len(progress)
        total     = len(df_constructed)
        confirmed = sum(1 for v in progress.values() if v.get("action") == "confirm")
        corrected = sum(1 for v in progress.values() if v.get("action") == "correct")
        removed   = sum(1 for v in progress.values() if v.get("action") == "remove")

        print_header(verified, total, confirmed, corrected, removed, dataset_name)
        print_criteria()
        print_pair(row, id_col, available_fields, verified, total)

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
                "action":    "skip",
                "timestamp": datetime.now().isoformat()
            }
            save_progress(progress)
            save_output(df_real, df_constructed, progress, id_col)
            history.append(code)
            ptr += 1
            continue

        if decision == "r":
            progress[code] = {
                "action":    "remove",
                "timestamp": datetime.now().isoformat()
            }
            save_progress(progress)
            save_output(df_real, df_constructed, progress, id_col)
            history.append(code)
            ptr += 1
            continue

        if decision == "c":
            progress[code] = {
                "action":    "confirm",
                "label":     "0",
                "timestamp": datetime.now().isoformat()
            }
            save_progress(progress)
            save_output(df_real, df_constructed, progress, id_col)
            history.append(code)
            ptr += 1
            continue

        if decision == "1":
            progress[code] = {
                "action":    "correct",
                "old_label": "0",
                "new_label": "1",
                "timestamp": datetime.now().isoformat()
            }
            save_progress(progress)
            save_output(df_real, df_constructed, progress, id_col)
            history.append(code)
            ptr += 1
            continue

        print("  Invalid input. Enter c, 1, r, s, b, or q.")
        input("  Press Enter to try again...")

    clear()
    print_final_stats(df_real, df_constructed, progress, dataset_name)
    save_output(df_real, df_constructed, progress, id_col)

if __name__ == "__main__":
    main()
