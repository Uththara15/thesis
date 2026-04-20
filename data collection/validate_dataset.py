"""
Dataset Validation Script
Thesis: Evaluating Multilingual Sentence Embedding Models
        for Bilingual Course Description Similarity in Finnish Higher Education
Author: Madee Uththara Deegoda Gamage
JAMK University of Applied Sciences

Run this script BEFORE moving to the manual labelling step.
It checks your scraped CSV or JSON file against the full validation checklist.

Usage:
    python validate_dataset.py --input jamk_courses_bilingual_clean.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

# Courses confirmed excluded during data quality review
CRITICAL_EXCLUSIONS = {
    "TTC8830": "English-only source page — Finnish and English outcomes are identical",
    "TTC8840": "English-only source page — Finnish and English outcomes are identical",
    "TTC8860": "English-only source page — Finnish and English outcomes are identical",
    "TZLM7040": "Scraper captured course objective paragraph, not learning outcomes",
    "TZLM7050": "Scraper captured course objective paragraph, not learning outcomes",
}

# Column names — matched to your actual JSON field names
COURSE_CODE_COL   = "course_id"
FI_OUTCOMES_COL   = "outcomes_fi"
EN_OUTCOMES_COL   = "outcomes_en"

# Full-text configuration columns (for Grade 5 field comparison)
FULL_TEXT_COLS_FI = ["outcomes_fi", "contents_fi", "assessment_fi"]
FULL_TEXT_COLS_EN = ["outcomes_en", "contents_en", "assessment_en"]

# Thresholds
MIN_WORD_COUNT       = 20    # Minimum words per outcome field
MAX_TOKEN_ESTIMATE   = 512   # BERT token limit (1 token ≈ 0.75 words)
SIMILARITY_THRESHOLD = 0.98  # Word overlap above this → flag as copy-paste

# Language detection patterns
FINNISH_PATTERN = re.compile(r'[äöåÄÖÅ]|oppia|osaa|opiskelija|kurssi|suorita', re.IGNORECASE)
ENGLISH_PATTERN = re.compile(r'\b(the|and|or|of|to|in|is|are|student|learn|understand|able)\b', re.IGNORECASE)

# ── Helpers ────────────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())

def estimated_tokens(text: str) -> int:
    """Rough BERT token estimate: words / 0.75 (Finnish morphology inflates this)."""
    return int(word_count(text) / 0.75)

def is_empty(text) -> bool:
    if not isinstance(text, str):
        return True
    return text.strip() == "" or text.strip() in ["|", "N/A", "None", "nan"]

def texts_are_identical(text1: str, text2: str) -> bool:
    if not isinstance(text1, str) or not isinstance(text2, str):
        return False
    return text1.strip().lower() == text2.strip().lower()

def cosine_sim_simple(text1: str, text2: str) -> float:
    """
    Lightweight word-overlap similarity (no model needed).
    Used only for copy-paste detection — not the thesis evaluation metric.
    """
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    return len(intersection) / max(len(words1), len(words2))

def unwrap_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    JSON fields are stored as single-element lists e.g. ['text...'].
    This unwraps them to plain strings so word counts work correctly.
    """
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0
                      else ("" if isinstance(x, list) else x)
        )
    return df

def load_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str).fillna("")
        df = unwrap_lists(df)
        print(f"  Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            df = pd.DataFrame(raw).fillna("")
        elif isinstance(raw, dict):
            df = pd.DataFrame(list(raw.values())).fillna("")
        else:
            sys.exit("  ERROR: Unrecognised JSON structure.")
        df = unwrap_lists(df)
        print(f"  Loaded JSON: {len(df)} rows, {len(df.columns)} columns")
    else:
        sys.exit(f"  ERROR: Unsupported file type '{suffix}'. Use .csv or .json")
    return df

# ── Validation Checks ──────────────────────────────────────────────────────────

def check_columns(df: pd.DataFrame) -> bool:
    required = [COURSE_CODE_COL, FI_OUTCOMES_COL, EN_OUTCOMES_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n  ❌ MISSING COLUMNS: {missing}")
        print(f"     Available columns: {list(df.columns)}")
        print(f"     → Edit the column name constants at the top of this script.")
        return False
    print(f"  ✅ Required columns found.")
    return True

def check_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    found = df[df[COURSE_CODE_COL].isin(CRITICAL_EXCLUSIONS.keys())]
    if not found.empty:
        print(f"\n  ⚠️  CRITICAL EXCLUSIONS FOUND — removing {len(found)} course(s):")
        for _, row in found.iterrows():
            code = row[COURSE_CODE_COL]
            print(f"     • {code}: {CRITICAL_EXCLUSIONS[code]}")
    else:
        print(f"\n  ✅ No critical exclusion courses found in data.")
    df_clean = df[~df[COURSE_CODE_COL].isin(CRITICAL_EXCLUSIONS.keys())].copy()
    print(f"     Remaining after exclusions: {len(df_clean)} courses")
    return df_clean

def check_pair_completeness(df: pd.DataFrame) -> list:
    issues = []
    for _, row in df.iterrows():
        code = row[COURSE_CODE_COL]
        if is_empty(row[FI_OUTCOMES_COL]):
            issues.append((code, "Finnish outcomes field is empty"))
        if is_empty(row[EN_OUTCOMES_COL]):
            issues.append((code, "English outcomes field is empty"))
    if issues:
        print(f"\n  ❌ PAIR COMPLETENESS — {len(issues)} issue(s):")
        for code, reason in issues[:10]:
            print(f"     • {code}: {reason}")
        if len(issues) > 10:
            print(f"     ... and {len(issues) - 10} more.")
    else:
        print(f"\n  ✅ All pairs have both Finnish and English outcomes.")
    return issues

def check_language_detection(df: pd.DataFrame) -> list:
    issues = []
    for _, row in df.iterrows():
        code = row[COURSE_CODE_COL]
        fi = row[FI_OUTCOMES_COL]
        en = row[EN_OUTCOMES_COL]

        if isinstance(fi, str) and fi.strip():
            if not FINNISH_PATTERN.search(fi):
                issues.append((code, "Finnish field may not contain Finnish text"))
            if ENGLISH_PATTERN.search(fi) and not FINNISH_PATTERN.search(fi):
                issues.append((code, "Finnish field may be English — possible language swap"))

        if isinstance(en, str) and en.strip():
            if not ENGLISH_PATTERN.search(en):
                issues.append((code, "English field may not contain English text"))

    if issues:
        print(f"\n  ⚠️  LANGUAGE DETECTION — {len(issues)} flag(s) (manual review needed):")
        for code, reason in issues[:10]:
            print(f"     • {code}: {reason}")
        if len(issues) > 10:
            print(f"     ... and {len(issues) - 10} more.")
    else:
        print(f"\n  ✅ Language detection passed for all pairs.")
    return issues

def check_identical_pairs(df: pd.DataFrame) -> list:
    issues = []
    for _, row in df.iterrows():
        code = row[COURSE_CODE_COL]
        fi = row[FI_OUTCOMES_COL]
        en = row[EN_OUTCOMES_COL]
        if texts_are_identical(fi, en):
            issues.append((code, "Finnish and English outcomes are IDENTICAL TEXT"))
        elif cosine_sim_simple(fi, en) >= SIMILARITY_THRESHOLD:
            issues.append((code, f"Suspiciously similar word overlap ≥ {SIMILARITY_THRESHOLD}"))
    if issues:
        print(f"\n  ❌ IDENTICAL/NEAR-IDENTICAL PAIRS — {len(issues)} issue(s):")
        for code, reason in issues:
            print(f"     • {code}: {reason}")
    else:
        print(f"\n  ✅ No identical Finnish–English pairs detected.")
    return issues

def check_minimum_length(df: pd.DataFrame) -> list:
    issues = []
    for _, row in df.iterrows():
        code = row[COURSE_CODE_COL]
        fi_wc = word_count(row[FI_OUTCOMES_COL])
        en_wc = word_count(row[EN_OUTCOMES_COL])
        if fi_wc < MIN_WORD_COUNT:
            issues.append((code, f"Finnish outcomes too short: {fi_wc} words (minimum {MIN_WORD_COUNT})"))
        if en_wc < MIN_WORD_COUNT:
            issues.append((code, f"English outcomes too short: {en_wc} words (minimum {MIN_WORD_COUNT})"))
    if issues:
        print(f"\n  ⚠️  MINIMUM LENGTH — {len(issues)} field(s) below {MIN_WORD_COUNT} words:")
        for code, reason in issues[:10]:
            print(f"     • {code}: {reason}")
        if len(issues) > 10:
            print(f"     ... and {len(issues) - 10} more.")
    else:
        print(f"\n  ✅ All fields meet minimum word count ({MIN_WORD_COUNT} words).")
    return issues

def check_token_length(df: pd.DataFrame) -> list:
    issues = []

    # Outcomes only
    for _, row in df.iterrows():
        code = row[COURSE_CODE_COL]
        fi_tok = estimated_tokens(row[FI_OUTCOMES_COL])
        en_tok = estimated_tokens(row[EN_OUTCOMES_COL])
        if fi_tok > MAX_TOKEN_ESTIMATE:
            issues.append((code, f"Finnish outcomes ≈{fi_tok} tokens — TRUNCATION NEEDED"))
        if en_tok > MAX_TOKEN_ESTIMATE:
            issues.append((code, f"English outcomes ≈{en_tok} tokens — TRUNCATION NEEDED"))

    # Full text
    full_fi_cols = [c for c in FULL_TEXT_COLS_FI if c in df.columns]
    full_en_cols = [c for c in FULL_TEXT_COLS_EN if c in df.columns]

    if full_fi_cols and full_en_cols:
        for _, row in df.iterrows():
            code = row[COURSE_CODE_COL]
            full_fi = " ".join(str(row[c]) for c in full_fi_cols if not is_empty(row.get(c, "")))
            full_en = " ".join(str(row[c]) for c in full_en_cols if not is_empty(row.get(c, "")))
            fi_tok = estimated_tokens(full_fi)
            en_tok = estimated_tokens(full_en)
            if fi_tok > MAX_TOKEN_ESTIMATE:
                issues.append((code, f"FULL TEXT Finnish ≈{fi_tok} tokens — will be truncated"))
            if en_tok > MAX_TOKEN_ESTIMATE:
                issues.append((code, f"FULL TEXT English ≈{en_tok} tokens — will be truncated"))

    if issues:
        print(f"\n  ⚠️  TOKEN LENGTH — {len(issues)} field(s) exceed {MAX_TOKEN_ESTIMATE} tokens:")
        for code, reason in issues[:10]:
            print(f"     • {code}: {reason}")
        if len(issues) > 10:
            print(f"     ... and {len(issues) - 10} more.")
        print(f"     → Document truncation handling in your thesis methodology section.")
    else:
        print(f"\n  ✅ All fields within {MAX_TOKEN_ESTIMATE} token estimate.")
    return issues

def check_duplicates(df: pd.DataFrame) -> list:
    dupes = df[df.duplicated(subset=[COURSE_CODE_COL], keep=False)]
    if not dupes.empty:
        codes = dupes[COURSE_CODE_COL].unique().tolist()
        print(f"\n  ❌ DUPLICATE COURSE CODES — {len(codes)} duplicate(s):")
        for code in codes:
            print(f"     • {code}")
        return codes
    print(f"\n  ✅ No duplicate course codes found.")
    return []

def check_encoding(df: pd.DataFrame) -> list:
    garbled_pattern = re.compile(r'\\x[0-9a-fA-F]{2}|â€|Ã¤|Ã¶')
    issues = []
    for _, row in df.iterrows():
        code = row[COURSE_CODE_COL]
        fi = row[FI_OUTCOMES_COL]
        if isinstance(fi, str) and garbled_pattern.search(fi):
            issues.append((code, "Garbled characters in Finnish field — encoding issue"))
    if issues:
        print(f"\n  ❌ ENCODING — {len(issues)} issue(s):")
        for code, reason in issues:
            print(f"     • {code}: {reason}")
    else:
        print(f"\n  ✅ No encoding issues detected.")
    return issues

# ── Word Count Preview ─────────────────────────────────────────────────────────

def print_word_count_preview(df: pd.DataFrame):
    """Print a quick word count table for the first 5 courses — sanity check."""
    print(f"\n  WORD COUNT PREVIEW (first 5 courses):")
    print(f"  {'Course':<12} {'FI words':>10} {'EN words':>10}")
    print(f"  {'-'*34}")
    for _, row in df.head(5).iterrows():
        code = row[COURSE_CODE_COL]
        fi_wc = word_count(row[FI_OUTCOMES_COL])
        en_wc = word_count(row[EN_OUTCOMES_COL])
        print(f"  {code:<12} {fi_wc:>10} {en_wc:>10}")

# ── Summary Report ─────────────────────────────────────────────────────────────

def print_summary(df_original: pd.DataFrame, df_clean: pd.DataFrame, all_issues: dict):
    print("\n" + "═" * 60)
    print("  VALIDATION SUMMARY")
    print("═" * 60)
    print(f"  Original dataset size  : {len(df_original)} courses")
    print(f"  After exclusions       : {len(df_clean)} courses")

    critical = (
        len(all_issues.get("pair_completeness", [])) +
        len(all_issues.get("identical_pairs", [])) +
        len(all_issues.get("duplicates", []))
    )
    warnings = (
        len(all_issues.get("language", [])) +
        len(all_issues.get("min_length", [])) +
        len(all_issues.get("token_length", []))
    )

    print(f"\n  ❌ Critical issues     : {critical}  (must fix before proceeding)")
    print(f"  ⚠️  Warnings            : {warnings}  (review and document in thesis)")

    if critical == 0 and warnings == 0:
        print(f"\n  🟢 READY FOR MANUAL LABELLING")
        print(f"     Report in thesis: final dataset = {len(df_clean)} valid course description pairs")
    elif critical == 0:
        print(f"\n  🟡 READY WITH WARNINGS — review flagged courses before labelling")
        print(f"     Report in thesis: final dataset = {len(df_clean)} valid course description pairs")
    else:
        print(f"\n  🔴 NOT READY — resolve critical issues first")

    print("═" * 60)

    # Save clean dataset
    out_csv = Path("validated_dataset.csv")
    df_clean.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  Clean dataset saved → {out_csv}")
    print(f"  (utf-8-sig encoding ensures ä ö å display correctly in Excel)\n")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate scraped Peppi dataset for thesis.")
    parser.add_argument("--input", required=True, help="Path to your scraped .csv or .json file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: File not found: {input_path}")

    print("\n" + "═" * 60)
    print("  THESIS DATASET VALIDATION")
    print("  JAMK — Multilingual Sentence Embedding Evaluation")
    print("═" * 60)
    print(f"\n  Input file: {input_path}\n")

    # Load and unwrap list fields
    df = load_data(input_path)

    # Check columns first
    if not check_columns(df):
        sys.exit("\nFix column names in the script constants and re-run.")

    df_original = df.copy()
    all_issues = {}

    # Run all checks
    df = check_exclusions(df)
    all_issues["pair_completeness"] = check_pair_completeness(df)
    all_issues["language"]          = check_language_detection(df)
    all_issues["identical_pairs"]   = check_identical_pairs(df)
    all_issues["min_length"]        = check_minimum_length(df)
    all_issues["token_length"]      = check_token_length(df)
    all_issues["duplicates"]        = check_duplicates(df)
    all_issues["encoding"]          = check_encoding(df)

    # Sanity check preview
    print_word_count_preview(df)

    # Final summary
    print_summary(df_original, df, all_issues)

if __name__ == "__main__":
    main()