"""
BIT Balanced Dataset Construction Script
Thesis: Evaluating Multilingual Sentence Embedding Models
        for Bilingual Course Description Similarity in Finnish Higher Education
Author: Madee Uththara Deegoda Gamage
JAMK University of Applied Sciences

Target dataset: ~72 pairs
  - 36 Positive       : All real labelled BIT pairs
  - 18 Easy negative  : Constructed from distant thematic groups
  - 18 Hard negative  : Constructed from same thematic group

Usage:
    python build_balanced_dataset_bit.py --input labelled_full_dataset.csv
"""

import argparse
import random
from pathlib import Path

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

COURSE_CODE_COL  = "course_id"
FI_OUTCOMES_COL  = "outcomes_fi"
EN_OUTCOMES_COL  = "outcomes_en"
FI_CONTENTS_COL  = "contents_fi"
EN_CONTENTS_COL  = "contents_en"
FI_ASSESS_COL    = "assessment_fi"
EN_ASSESS_COL    = "assessment_en"
LABEL_COL        = "similarity_label"

RANDOM_SEED      = 42

TARGET_POSITIVE  = 36
TARGET_EASY_NEG  = 18
TARGET_HARD_NEG  = 18

# ── BIT Thematic Groups ────────────────────────────────────────────────────────

THEMATIC_GROUPS = {
    "game_programming": [
        "HG00CI44", "HG00CI49", "HG00CH07",
        "HG00CF54", "HG00CF56"
    ],
    "game_art": [
        "HG00CF48", "HG00CF52", "HG00CF53",
        "HG00CI48", "HG00CF55"
    ],
    "game_design": [
        "HG00CF49", "HG00CF51", "HG00CI51",
        "HG00CI52", "HG00CF63"
    ],
    "game_engine": [
        "HG00CI47", "HG00CH08", "HG00CI50",
        "HG00CF57", "HG00BY32"
    ],
    "game_business": [
        "HG00CF50", "HG00CF59", "HG00CF60",
        "HG00CF61", "HG00CF62"
    ],
    "foundations": [
        "HG00CI43", "HG00CI45", "HG00CI46", "HG00CQ39"
    ],
    "professional": [
        "HT00CF20", "HT00CL43", "HT00CL11",
        "HT00BY31", "HTP20840", "HT00CU96"
    ],
}

# ── Adjacent Groups — Hard Negatives ──────────────────────────────────────────

ADJACENT_PAIRS = [
    ("game_programming", "game_engine"),
    ("game_programming", "game_design"),
    ("game_art",         "game_design"),
    ("game_art",         "game_engine"),
    ("game_design",      "game_business"),
    ("game_engine",      "game_programming"),
    ("game_business",    "professional"),
    ("foundations",      "game_programming"),
    ("professional",     "game_business"),
    ("game_design",      "game_art"),
]

# ── Distant Groups — Easy Negatives ───────────────────────────────────────────

DISTANT_PAIRS = [
    ("game_programming", "professional"),
    ("game_programming", "game_business"),
    ("game_art",         "professional"),
    ("game_art",         "foundations"),
    ("game_design",      "foundations"),
    ("game_engine",      "professional"),
    ("game_business",    "game_programming"),
    ("foundations",      "game_business"),
    ("professional",     "game_art"),
    ("professional",     "game_engine"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def construct_easy_negatives(df, needed, used_pairs):
    random.seed(RANDOM_SEED)
    constructed = []
    attempts = 0

    while len(constructed) < needed and attempts < 50000:
        attempts += 1
        group_fi, group_en = random.choice(DISTANT_PAIRS)
        codes_fi = THEMATIC_GROUPS.get(group_fi, [])
        codes_en = THEMATIC_GROUPS.get(group_en, [])

        if not codes_fi or not codes_en:
            continue

        code_fi = random.choice(codes_fi)
        code_en = random.choice(codes_en)

        if code_fi == code_en:
            continue

        pair_key = (code_fi, code_en)
        if pair_key in used_pairs:
            continue

        rows_fi = df[df[COURSE_CODE_COL] == code_fi]
        rows_en = df[df[COURSE_CODE_COL] == code_en]

        if rows_fi.empty or rows_en.empty:
            continue

        row_fi = rows_fi.iloc[0]
        row_en = rows_en.iloc[0]

        fi_outcomes = str(row_fi[FI_OUTCOMES_COL]).strip()
        fi_contents = str(row_fi[FI_CONTENTS_COL]).strip()
        fi_assess   = str(row_fi[FI_ASSESS_COL]).strip()
        en_outcomes = str(row_en[EN_OUTCOMES_COL]).strip()
        en_contents = str(row_en[EN_CONTENTS_COL]).strip()
        en_assess   = str(row_en[EN_ASSESS_COL]).strip()

        if not all([fi_outcomes, en_outcomes]):
            continue

        constructed.append({
            COURSE_CODE_COL:   f"SYN_EASY_{code_fi}_{code_en}",
            "title_fi":        f"[SYNTHETIC] {row_fi.get('title_fi', '')}",
            "title_en":        f"[SYNTHETIC] {row_en.get('title_en', '')}",
            "credits":         "N/A",
            FI_OUTCOMES_COL:   fi_outcomes,
            EN_OUTCOMES_COL:   en_outcomes,
            FI_CONTENTS_COL:   fi_contents,
            EN_CONTENTS_COL:   en_contents,
            FI_ASSESS_COL:     fi_assess,
            EN_ASSESS_COL:     en_assess,
            LABEL_COL:         "0",
            "label_note":      f"Easy negative — distant: FI={code_fi}({group_fi}) EN={code_en}({group_en})",
            "label_timestamp": "constructed",
            "pair_type":       "constructed",
            "pair_difficulty": "easy_negative"
        })
        used_pairs.add(pair_key)

    return constructed


def construct_hard_negatives(df, needed, used_pairs):
    random.seed(RANDOM_SEED + 1)
    constructed = []
    attempts = 0

    eligible_groups = {
        g: codes for g, codes in THEMATIC_GROUPS.items()
        if len(codes) >= 2
    }

    while len(constructed) < needed and attempts < 100000:
        attempts += 1
        group  = random.choice(list(eligible_groups.keys()))
        codes  = eligible_groups[group]

        if len(codes) < 2:
            continue

        code_fi, code_en = random.sample(codes, 2)
        pair_key = (code_fi, code_en)

        if pair_key in used_pairs:
            continue

        rows_fi = df[df[COURSE_CODE_COL] == code_fi]
        rows_en = df[df[COURSE_CODE_COL] == code_en]

        if rows_fi.empty or rows_en.empty:
            continue

        row_fi = rows_fi.iloc[0]
        row_en = rows_en.iloc[0]

        fi_outcomes = str(row_fi[FI_OUTCOMES_COL]).strip()
        fi_contents = str(row_fi[FI_CONTENTS_COL]).strip()
        fi_assess   = str(row_fi[FI_ASSESS_COL]).strip()
        en_outcomes = str(row_en[EN_OUTCOMES_COL]).strip()
        en_contents = str(row_en[EN_CONTENTS_COL]).strip()
        en_assess   = str(row_en[EN_ASSESS_COL]).strip()

        if not all([fi_outcomes, en_outcomes]):
            continue

        constructed.append({
            COURSE_CODE_COL:   f"SYN_HARD_{code_fi}_{code_en}",
            "title_fi":        f"[SYNTHETIC] {row_fi.get('title_fi', '')}",
            "title_en":        f"[SYNTHETIC] {row_en.get('title_en', '')}",
            "credits":         "N/A",
            FI_OUTCOMES_COL:   fi_outcomes,
            EN_OUTCOMES_COL:   en_outcomes,
            FI_CONTENTS_COL:   fi_contents,
            EN_CONTENTS_COL:   en_contents,
            FI_ASSESS_COL:     fi_assess,
            EN_ASSESS_COL:     en_assess,
            LABEL_COL:         "0",
            "label_note":      f"Hard negative — same group ({group}): FI={code_fi} EN={code_en}",
            "label_timestamp": "constructed",
            "pair_type":       "constructed",
            "pair_difficulty": "hard_negative"
        })
        used_pairs.add(pair_key)

    return constructed

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return

    df = pd.read_csv(input_path, dtype=str).fillna("")
    print(f"\n  Loaded: {len(df)} courses from {input_path}")

    # Real positive pairs
    df_real = df.copy()
    df_real[LABEL_COL]         = "1"
    df_real["label_note"]      = "Real bilingual pair — manually labelled on full content"
    df_real["label_timestamp"] = "real"
    df_real["pair_type"]       = "real"
    df_real["pair_difficulty"] = "positive"
    print(f"  Real positive pairs: {len(df_real)}")

    # Track used pairs
    used_pairs = set()
    for code in df[COURSE_CODE_COL].tolist():
        used_pairs.add((code, code))

    # Easy negatives
    print(f"\n  Constructing {TARGET_EASY_NEG} easy negatives (distant groups)...")
    easy_negatives = construct_easy_negatives(df, TARGET_EASY_NEG, used_pairs)
    print(f"  Constructed {len(easy_negatives)} easy negatives.")

    # Hard negatives
    print(f"\n  Constructing {TARGET_HARD_NEG} hard negatives (same group)...")
    hard_negatives = construct_hard_negatives(df, TARGET_HARD_NEG, used_pairs)
    print(f"  Constructed {len(hard_negatives)} hard negatives.")

    # Sample hard negatives
    if hard_negatives:
        print(f"\n  Sample hard negative pairs:")
        for pair in hard_negatives[:5]:
            print(f"    {pair[COURSE_CODE_COL]}")
            print(f"    {pair['label_note']}")
            print()

    # Assemble
    parts = [df_real]
    if easy_negatives:
        parts.append(pd.DataFrame(easy_negatives))
    if hard_negatives:
        parts.append(pd.DataFrame(hard_negatives))

    df_final = pd.concat(parts, ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Summary
    total  = len(df_final)
    n_pos  = len(df_final[df_final[LABEL_COL] == "1"])
    n_neg  = len(df_final[df_final[LABEL_COL] == "0"])
    n_easy = len(df_final[df_final["pair_difficulty"] == "easy_negative"])
    n_hard = len(df_final[df_final["pair_difficulty"] == "hard_negative"])
    n_real = len(df_final[df_final["pair_type"] == "real"])
    n_con  = len(df_final[df_final["pair_type"] == "constructed"])

    print(f"\n  {'='*44}")
    print(f"  FINAL BALANCED BIT DATASET")
    print(f"  {'='*44}")
    print(f"  Total pairs          : {total}")
    print(f"  Positive    (1)      : {n_pos}  ({n_pos/total*100:.1f}%)")
    print(f"  Negative    (0)      : {n_neg}  ({n_neg/total*100:.1f}%)")
    print(f"    - Easy negatives   : {n_easy}  (distant groups)")
    print(f"    - Hard negatives   : {n_hard}  (same group)")
    print(f"  Real pairs           : {n_real}")
    print(f"  Constructed pairs    : {n_con}")
    print(f"  {'='*44}")

    # Field completeness
    print(f"\n  Field completeness check:")
    for col in [FI_OUTCOMES_COL, EN_OUTCOMES_COL]:
        empty  = (df_final[col].str.strip() == "").sum()
        status = "OK" if empty == 0 else f"EMPTY: {empty}"
        print(f"  {col:<22}: {status}")

    # Save
    out_path = Path("bit_balanced_dataset.csv")
    df_final.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved -> {out_path}")
    print(f"\n  Random seed : {RANDOM_SEED}")
    print(f"  Next step   : Run verify_constructed.py --input bit_balanced_dataset.csv\n")

if __name__ == "__main__":
    main()
