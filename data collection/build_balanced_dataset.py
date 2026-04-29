"""
Balanced Dataset Construction Script v3
Thesis: Evaluating Multilingual Sentence Embedding Models
        for Bilingual Course Description Similarity in Finnish Higher Education
Author: Madee Uththara Deegoda Gamage
JAMK University of Applied Sciences

Target dataset: 154 pairs
  - 77  Positive       : All real labelled pairs (all fields populated)
  - 40  Easy negative  : Finnish from one course + English from distant group
  - 37  Hard negative  : Finnish from one course + English from SAME group
                         (same topic domain, different courses)

Hard negatives are the most challenging — same vocabulary and domain
but different course content. This tests genuine semantic understanding.

Usage:
    python build_balanced_dataset.py --input validated_dataset.csv
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

TARGET_POSITIVE  = 77
TARGET_EASY_NEG  = 40
TARGET_HARD_NEG  = 37
# Total           = 154

# ── Thematic Groups ────────────────────────────────────────────────────────────

THEMATIC_GROUPS = {
    "mathematics": [
        "TT00CD55", "TT00CD56", "TT00CD57", "TT00CD58",
        "TZLM1300", "TZLM2300", "TZLM3300", "TT00CD65",
        "TT00CK80", "TT00CD98"
    ],
    "physics": [
        "TZLF1300", "TZLF2300", "TT00CD68", "TZLF8020"
    ],
    "cybersecurity": [
        "TT00CQ78", "TT00CQ79", "TT00CQ80", "TT00CD86",
        "TT00CD87", "TT00CE17", "TT00CE18", "TT00CE13",
        "TT00CE07", "TT00CE08", "TT00CE09", "TT00CE10",
        "TT00CE11", "TT00CE12", "TT00CD88", "TT00CE14",
        "TT00CE15", "TT00CE16"
    ],
    "programming": [
        "TT00CD77", "TT00CD78", "TT00CD79", "TT00CD80",
        "TT00CD81", "TT00CD83", "TT00CD85", "TT00CD91",
        "TT00CD61", "TT00CD73"
    ],
    "networking": [
        "TT00CD70", "TT00CE20", "TT00CE22", "TT00CE23",
        "TT00CE24", "TT00CE28", "TT00CE29", "TT00CE27"
    ],
    "ai_data": [
        "TT00CD84", "TT00CD99", "TT00CE00", "TT00CE01",
        "TT00CE02", "TT00CE03", "TT00CE04", "TT00CE05",
        "TT00CE06"
    ],
    "systems": [
        "TT00CD71", "TT00CD72", "TT00CD75", "TT00CD76",
        "TT00CE21", "TT00CE25", "TT00CE26"
    ],
    "software_engineering": [
        "TT00CD74", "TT00CD82", "TT00CD89", "TT00CD90"
    ],
    "mobile_web": [
        "TT00CD92", "TT00CD93", "TT00CD94"
    ],
    "professional": [
        "ZZ00CB60", "TT00CJ28",
        "TT00CL50", "TT00CQ81"
    ]
}

# Courses that are essentially the same despite different codes
# These must NOT be paired as hard negatives
KNOWN_DUPLICATES = {
    "TT00CJ28", "TT00CJ29", "TT00CJ30", "TT00CJ31"
}

# ── Distant Groups → Easy Negatives ───────────────────────────────────────────

DISTANT_PAIRS = [
    ("mathematics",    "cybersecurity"),
    ("mathematics",    "mobile_web"),
    ("mathematics",    "professional"),
    ("physics",        "cybersecurity"),
    ("physics",        "ai_data"),
    ("physics",        "professional"),
    ("physics",        "mobile_web"),
    ("cybersecurity",  "mathematics"),
    ("cybersecurity",  "mobile_web"),
    ("cybersecurity",  "professional"),
    ("programming",    "physics"),
    ("programming",    "professional"),
    ("networking",     "mathematics"),
    ("networking",     "professional"),
    ("networking",     "mobile_web"),
    ("ai_data",        "cybersecurity"),
    ("ai_data",        "professional"),
    ("ai_data",        "physics"),
    ("professional",   "mathematics"),
    ("professional",   "programming"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def construct_easy_negatives(df, needed, used_pairs):
    """
    Easy negatives: Finnish fields from one course +
                    English fields from a course in a DISTANT group.
    """
    random.seed(RANDOM_SEED)
    constructed = []
    attempts = 0
    max_attempts = 50000

    while len(constructed) < needed and attempts < max_attempts:
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

        if not all([fi_outcomes, fi_contents, fi_assess,
                    en_outcomes, en_contents, en_assess]):
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
    """
    Hard negatives: Finnish fields from one course +
                    English fields from a DIFFERENT course in the SAME group.
    Same topic domain — different course content.
    """
    random.seed(RANDOM_SEED + 1)  # Different seed to avoid same sequence
    constructed = []
    attempts = 0
    max_attempts = 100000

    # Get list of groups with at least 2 courses
    eligible_groups = {
        group: codes
        for group, codes in THEMATIC_GROUPS.items()
        if len(codes) >= 2
    }

    while len(constructed) < needed and attempts < max_attempts:
        attempts += 1

        # Pick a random group
        group = random.choice(list(eligible_groups.keys()))
        codes = eligible_groups[group]

        # Pick two different courses from the same group
        if len(codes) < 2:
            continue

        code_fi, code_en = random.sample(codes, 2)

        # Skip known duplicates
        if code_fi in KNOWN_DUPLICATES and code_en in KNOWN_DUPLICATES:
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

        if not all([fi_outcomes, fi_contents, fi_assess,
                    en_outcomes, en_contents, en_assess]):
            continue

        title_fi = row_fi.get("title_fi", "")
        title_en = row_en.get("title_en", "")

        constructed.append({
            COURSE_CODE_COL:   f"SYN_HARD_{code_fi}_{code_en}",
            "title_fi":        f"[SYNTHETIC] {title_fi}",
            "title_en":        f"[SYNTHETIC] {title_en}",
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

    # Track used pairs — real pairs use (code, code) to block self-pairing
    used_pairs = set()
    for code in df[COURSE_CODE_COL].tolist():
        used_pairs.add((code, code))

    # Easy negatives
    print(f"\n  Constructing {TARGET_EASY_NEG} easy negatives (distant groups)...")
    easy_negatives = construct_easy_negatives(df, TARGET_EASY_NEG, used_pairs)
    print(f"  ✅ Constructed {len(easy_negatives)} easy negatives.")

    # Hard negatives
    print(f"\n  Constructing {TARGET_HARD_NEG} hard negatives (same group)...")
    hard_negatives = construct_hard_negatives(df, TARGET_HARD_NEG, used_pairs)
    print(f"  ✅ Constructed {len(hard_negatives)} hard negatives.")

    # Show some examples of hard negative pairs
    if hard_negatives:
        print(f"\n  Sample hard negative pairs:")
        for pair in hard_negatives[:5]:
            note = pair["label_note"]
            print(f"    {pair[COURSE_CODE_COL]}")
            print(f"    {note}")
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

    print(f"\n  ══════════════════════════════════════════")
    print(f"  FINAL BALANCED DATASET")
    print(f"  ══════════════════════════════════════════")
    print(f"  Total pairs          : {total}")
    print(f"  • Positive    (1)    : {n_pos}  ({n_pos/total*100:.1f}%)")
    print(f"  • Negative    (0)    : {n_neg}  ({n_neg/total*100:.1f}%)")
    print(f"    - Easy negatives   : {n_easy}  (distant groups)")
    print(f"    - Hard negatives   : {n_hard}  (same group)")
    print(f"  ──────────────────────────────────────────")
    print(f"  • Real pairs         : {n_real}")
    print(f"  • Constructed pairs  : {n_con}")
    print(f"  ══════════════════════════════════════════")

    # Field completeness check
    print(f"\n  Field completeness check:")
    for col in [FI_OUTCOMES_COL, EN_OUTCOMES_COL,
                FI_CONTENTS_COL, EN_CONTENTS_COL,
                FI_ASSESS_COL,   EN_ASSESS_COL]:
        empty  = (df_final[col].str.strip() == "").sum()
        status = "✅" if empty == 0 else f"❌ {empty} empty"
        print(f"  {col:<22}: {status}")

    # Save
    out_path = Path("final_dataset.csv")
    df_final.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ Saved → {out_path}")
    print(f"\n  THESIS NOTES:")
    print(f"  Random seed     : {RANDOM_SEED} — report in methodology section")
    print(f"  Easy negatives  : Finnish + English from distant thematic domains")
    print(f"  Hard negatives  : Finnish + English from same thematic group,")
    print(f"                    different courses — tests fine-grained discrimination")
    print(f"  All fields      : outcomes + contents + assessment for all 154 pairs\n")

if __name__ == "__main__":
    main()