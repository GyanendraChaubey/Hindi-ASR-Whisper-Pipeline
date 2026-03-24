import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import pandas as pd
from datasets import load_dataset
from rapidfuzz import fuzz, process
from tqdm import tqdm

from .question1 import default_manifest_path, ensure_dir, normalize_text, preprocess_dataset


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ROMAN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
ONLY_DEVANAGARI_WORD_RE = re.compile(r"^[\u0900-\u097F]+$")
TOKEN_STRIP_CHARS = ".,!?;:\"'`[]{}()<>|/\\।,`~"
MATRA_CHARS = "ािीुूृेैोौंःॅॉॆॊ"
HALANT = "्"


TOKEN_CANONICAL_OVERRIDES = {
    "हज़ार": "हजार",
    "करोड़": "करोड",
    "पन्द्रह": "पंद्रह",
    "चौतीस": "चौंतीस",
    "अठ्ठावन": "अट्ठावन",
    "दोनो": "दोनों",
}

DEVANAGARI_ENGLISH_LEXICON = {
    "इंटरव्यू",
    "कंप्यूटर",
    "कम्प्यूटर",
    "मोबाइल",
    "फोन",
    "जॉब",
    "जॉब्स",
    "प्रॉब्लम",
    "प्रॉब्लेम",
    "ट्रेनिंग",
    "प्रोजेक्ट",
    "मैनेजर",
    "मैनेजमेंट",
    "स्टार्टअप",
    "मार्केटिंग",
    "सेल्स",
    "स्कूल",
    "कॉलेज",
    "ऑफिस",
    "ऑनलाइन",
    "इंटरनेट",
    "सॉफ्टवेयर",
    "हार्डवेयर",
    "इंजीनियर",
    "डॉक्टर",
    "कोचिंग",
    "टेस्ट",
    "रिजल्ट",
    "बैंक",
    "कोडिंग",
    "डेवलपर",
    "अपडेट",
    "सिस्टम",
    "फीडबैक",
    "कॉल",
    "वीडियो",
    "ऑडियो",
    "फाइल",
    "डेटा",
    "प्लान",
    "डिजाइन",
    "फॉर्म",
    "लिंक",
    "यूजर",
    "पासवर्ड",
    "आईडी",
    "ईमेल",
}


CATEGORY_EXPLANATIONS = {
    "roman_or_mixed_script": "Roman/mixed script violates Devanagari-only Hindi transcription guideline and is likely unreliable.",
    "tokenization_artifact": "Digits/punctuation artifacts are usually tokenization noise rather than lexical words.",
    "high_support_lexicon_match": "Word appears in trusted lexicon and has strong in-domain support.",
    "lexicon_supported_low_frequency": "Word appears in trusted lexicon but low in-domain count makes confidence moderate.",
    "likely_typo_near_lexicon": "Word is very close to a trusted lexicon word and likely a spelling mistake.",
    "suspicious_form": "Word has suspicious orthographic pattern (matra/halant/repetition) without strong supporting evidence.",
    "rare_or_name_ambiguous": "Rare word with weak evidence; may be valid proper noun or may be misspelling.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Question-3 spelling quality pipeline.")
    parser.add_argument("--manifest-csv", type=Path, default=None, help="Path to FT Data CSV for corpus evidence.")
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/q3"), help="Output folder for Q3 artifacts.")
    parser.add_argument(
        "--wordlist-file",
        type=Path,
        default=None,
        help="Optional unique-word list file (csv/txt/json). If omitted, unique words are derived from corpus.",
    )
    parser.add_argument(
        "--word-column",
        type=str,
        default="",
        help="CSV column name to read words from. If not set, auto-detects.",
    )
    parser.add_argument("--max-recordings", type=int, default=0, help="Debug limiter for preprocessing. 0 means all.")
    parser.add_argument("--min-segment-sec", type=float, default=1.0, help="Drop segments shorter than this.")
    parser.add_argument("--max-segment-sec", type=float, default=30.0, help="Drop segments longer than this.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild preprocessing cache.")

    parser.add_argument(
        "--use-fleurs-lexicon",
        action="store_true",
        default=True,
        help="Use FLEURS Hindi transcriptions as external lexical evidence.",
    )
    parser.add_argument(
        "--disable-fleurs-lexicon",
        action="store_true",
        help="Disable FLEURS lexicon loading.",
    )
    parser.add_argument("--max-fleurs-per-split", type=int, default=0, help="Optional cap per FLEURS split for faster debug.")

    parser.add_argument("--review-sample-size", type=int, default=50, help="Low-confidence review sample size.")
    parser.add_argument(
        "--manual-review-file",
        type=Path,
        default=None,
        help="Completed review CSV with `manual_label` column (required in strict mode).",
    )
    parser.add_argument(
        "--allow-proxy-review",
        action="store_true",
        help="Use internal proxy labels when manual review is unavailable (non-compliant with assignment c).",
    )
    return parser.parse_args()


def canonicalize_token(token: str) -> str:
    token = token.strip()
    token = token.strip(TOKEN_STRIP_CHARS)
    token = token.replace("।", "")
    token = TOKEN_CANONICAL_OVERRIDES.get(token, token)
    return token


def extract_tokens(text: str) -> list[str]:
    norm = normalize_text(text)
    out: list[str] = []
    for raw in norm.split():
        token = canonicalize_token(raw)
        if token:
            out.append(token)
    return out


def has_suspicious_pattern(word: str) -> list[str]:
    reasons: list[str] = []
    if not word:
        return reasons
    if word.endswith(HALANT):
        reasons.append("ends_with_halant")
    if word[0] in MATRA_CHARS:
        reasons.append("starts_with_matra")
    if re.search(r"(.)\1\1", word):
        reasons.append("triple_char_repeat")
    if re.search(rf"[{MATRA_CHARS}]{{2,}}", word):
        reasons.append("repeated_matra_sequence")
    if "़़" in word:
        reasons.append("duplicate_nukta")
    return reasons


def load_word_list(path: Path, word_column: str) -> list[str]:
    suffix = path.suffix.lower()
    words: list[str] = []

    if suffix == ".csv":
        df = pd.read_csv(path)
        if df.empty:
            return []
        col = word_column.strip()
        if not col:
            candidates = ["word", "words", "token", "unique_word", "unique_words"]
            for candidate in candidates:
                if candidate in df.columns:
                    col = candidate
                    break
        if not col:
            col = df.columns[0]
        words = [str(item) for item in df[col].dropna().tolist()]
    elif suffix in {".txt", ".tsv"}:
        words = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            words = [str(item) for item in payload]
        elif isinstance(payload, dict):
            candidates = ["words", "tokens", "data"]
            for key in candidates:
                if key in payload and isinstance(payload[key], list):
                    words = [str(item) for item in payload[key]]
                    break
    else:
        raise ValueError(f"Unsupported word list format: {path.suffix}")

    cleaned: list[str] = []
    seen: set[str] = set()
    for word in words:
        token = canonicalize_token(str(word))
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(token)
    return cleaned


def build_in_domain_frequency(segments_df: pd.DataFrame) -> Counter:
    freq: Counter = Counter()
    for text in segments_df["text"].tolist():
        for token in extract_tokens(text):
            freq[token] += 1
    return freq


def build_fleurs_lexicon(max_per_split: int) -> set[str]:
    lexicon: set[str] = set()
    splits = ["train", "validation", "test"]
    for split in splits:
        ds = load_dataset("google/fleurs", "hi_in", split=split, trust_remote_code=True)
        if max_per_split > 0:
            ds = ds.select(range(min(max_per_split, len(ds))))
        for raw_row in ds:
            row = cast(dict[str, Any], raw_row)
            for token in extract_tokens(str(row["transcription"])):
                if ONLY_DEVANAGARI_WORD_RE.match(token):
                    lexicon.add(token)
    return lexicon


def build_reference_lexicons(
    domain_freq: Counter,
    use_fleurs_lexicon: bool,
    max_fleurs_per_split: int,
) -> tuple[set[str], set[str], set[str]]:
    domain_lexicon = {
        word
        for word, count in domain_freq.items()
        if count >= 2 and ONLY_DEVANAGARI_WORD_RE.match(word) and not has_suspicious_pattern(word)
    }

    fleurs_lexicon: set[str] = set()
    if use_fleurs_lexicon:
        fleurs_lexicon = build_fleurs_lexicon(max_fleurs_per_split)

    reference_lexicon = set(domain_lexicon) | set(fleurs_lexicon) | set(DEVANAGARI_ENGLISH_LEXICON)
    return domain_lexicon, fleurs_lexicon, reference_lexicon


def build_first_char_buckets(reference_lexicon: set[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for word in reference_lexicon:
        if not word:
            continue
        buckets[word[0]].append(word)
    return buckets


def find_nearest_reference_word(word: str, buckets: dict[str, list[str]]) -> tuple[str | None, float | None]:
    if not word:
        return None, None
    candidates = buckets.get(word[0], [])
    if not candidates:
        return None, None

    best = process.extractOne(
        word,
        candidates,
        scorer=fuzz.ratio,
        score_cutoff=80,
    )
    if not best:
        return None, None
    return str(best[0]), float(best[1])


def classify_word(
    word: str,
    domain_freq: Counter,
    domain_lexicon: set[str],
    fleurs_lexicon: set[str],
    reference_lexicon: set[str],
    first_char_buckets: dict[str, list[str]],
) -> dict[str, Any]:
    has_devanagari = bool(DEVANAGARI_RE.search(word))
    has_roman = bool(ROMAN_RE.search(word))
    has_digit = bool(DIGIT_RE.search(word))
    pure_devanagari = bool(ONLY_DEVANAGARI_WORD_RE.match(word))
    suspicious_reasons = has_suspicious_pattern(word)

    in_domain_freq = int(domain_freq.get(word, 0))
    in_domain_lexicon = word in domain_lexicon
    in_fleurs_lexicon = word in fleurs_lexicon
    in_reference_lexicon = word in reference_lexicon

    nearest_word: str | None = None
    nearest_score: float | None = None
    if pure_devanagari and not in_reference_lexicon and len(word) >= 3:
        nearest_word, nearest_score = find_nearest_reference_word(word, first_char_buckets)

    label = "correct"
    confidence = "low"
    category = "rare_or_name_ambiguous"
    reason = "Rare or ambiguous token without strong supporting evidence."

    if has_roman:
        label = "incorrect"
        confidence = "high"
        category = "roman_or_mixed_script"
        reason = "Contains Roman script; guideline expects Hindi words in Devanagari."
    elif has_digit:
        label = "incorrect"
        confidence = "high"
        category = "tokenization_artifact"
        reason = "Contains digits; likely tokenization/normalization artifact, not a clean lexical word."
    elif not has_devanagari:
        label = "incorrect"
        confidence = "high"
        category = "tokenization_artifact"
        reason = "No Devanagari characters detected."
    elif in_reference_lexicon and in_domain_freq >= 5:
        label = "correct"
        confidence = "high"
        category = "high_support_lexicon_match"
        reason = "Strong lexical support and frequent in-domain evidence."
    elif in_reference_lexicon and in_domain_freq >= 2:
        label = "correct"
        confidence = "high"
        category = "high_support_lexicon_match"
        reason = "Found in trusted lexicon and repeated in-domain."
    elif in_reference_lexicon:
        label = "correct"
        confidence = "medium"
        category = "lexicon_supported_low_frequency"
        reason = "Found in trusted lexicon but low in-domain frequency."
    elif nearest_word and nearest_score is not None and nearest_score >= 92:
        label = "incorrect"
        confidence = "high"
        category = "likely_typo_near_lexicon"
        reason = f"Very close to known form `{nearest_word}` (similarity {nearest_score:.1f})."
    elif suspicious_reasons and nearest_word and nearest_score is not None and nearest_score >= 86:
        label = "incorrect"
        confidence = "medium"
        category = "likely_typo_near_lexicon"
        reason = f"Suspicious form with close known alternative `{nearest_word}` (similarity {nearest_score:.1f})."
    elif suspicious_reasons:
        label = "incorrect"
        confidence = "low"
        category = "suspicious_form"
        reason = f"Suspicious orthographic pattern: {', '.join(suspicious_reasons)}."
    elif nearest_word and nearest_score is not None and nearest_score >= 86:
        label = "incorrect"
        confidence = "low"
        category = "likely_typo_near_lexicon"
        reason = f"Possibly misspelled near `{nearest_word}` (similarity {nearest_score:.1f})."
    elif in_domain_freq >= 2:
        label = "correct"
        confidence = "medium"
        category = "lexicon_supported_low_frequency"
        reason = "Repeated in-domain despite no external lexicon match."
    else:
        label = "correct"
        confidence = "low"
        category = "rare_or_name_ambiguous"
        reason = "May be valid rare/proper noun, but evidence is weak."

    return {
        "word": word,
        "predicted_label": label,
        "confidence": confidence,
        "reason": reason,
        "category": category,
        "category_explanation": CATEGORY_EXPLANATIONS.get(category, ""),
        "in_domain_frequency": in_domain_freq,
        "in_domain_lexicon_hit": int(in_domain_lexicon),
        "fleurs_lexicon_hit": int(in_fleurs_lexicon),
        "reference_lexicon_hit": int(in_reference_lexicon),
        "nearest_reference_word": nearest_word or "",
        "nearest_similarity_score": round(float(nearest_score), 3) if nearest_score is not None else "",
        "suspicious_pattern_flags": "|".join(suspicious_reasons),
    }


def classify_all_words(
    words: list[str],
    domain_freq: Counter,
    domain_lexicon: set[str],
    fleurs_lexicon: set[str],
    reference_lexicon: set[str],
) -> pd.DataFrame:
    buckets = build_first_char_buckets(reference_lexicon)
    rows: list[dict[str, Any]] = []
    for word in tqdm(words, desc="Classifying unique words"):
        rows.append(
            classify_word(
                word=word,
                domain_freq=domain_freq,
                domain_lexicon=domain_lexicon,
                fleurs_lexicon=fleurs_lexicon,
                reference_lexicon=reference_lexicon,
                first_char_buckets=buckets,
            )
        )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        columns=[
            "word",
            "predicted_label",
            "confidence",
            "reason",
            "category",
            "category_explanation",
            "in_domain_frequency",
            "in_domain_lexicon_hit",
            "fleurs_lexicon_hit",
            "reference_lexicon_hit",
            "nearest_reference_word",
            "nearest_similarity_score",
            "suspicious_pattern_flags",
        ]
    )


def build_low_confidence_review_sample(classified_df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    low = classified_df[classified_df["confidence"] == "low"].copy()
    if low.empty:
        out = low.copy()
        out["manual_label"] = ""
        out["manual_notes"] = ""
        return out

    correct_low = low[low["predicted_label"] == "correct"]
    incorrect_low = low[low["predicted_label"] == "incorrect"]

    per_bucket = max(1, sample_size // 2)
    sample_correct = correct_low.head(per_bucket)
    sample_incorrect = incorrect_low.head(per_bucket)
    merged = pd.concat([sample_correct, sample_incorrect], ignore_index=True)

    if len(merged) < sample_size:
        remaining = low[~low["word"].isin(set(merged["word"].tolist()))].head(sample_size - len(merged))
        merged = pd.concat([merged, remaining], ignore_index=True)

    merged = merged.head(sample_size).copy()
    merged["manual_label"] = ""
    merged["manual_notes"] = ""
    return merged


def proxy_manual_label(row: pd.Series) -> str:
    if row["reference_lexicon_hit"] == 1 or row["in_domain_frequency"] >= 2:
        return "correct"
    if row["nearest_similarity_score"] != "" and float(row["nearest_similarity_score"]) >= 90:
        return "incorrect"
    if row["category"] in {"roman_or_mixed_script", "tokenization_artifact", "likely_typo_near_lexicon"}:
        return "incorrect"
    return "correct"


def analyze_low_confidence_review(
    classified_df: pd.DataFrame,
    review_sample_df: pd.DataFrame,
    manual_review_file: Path | None,
    allow_proxy_review: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    review_df = review_sample_df.copy()
    review_mode = ""

    if manual_review_file and manual_review_file.exists():
        external = pd.read_csv(manual_review_file)
        if "word" in external.columns and "manual_label" in external.columns:
            review_df = review_df.drop(columns=["manual_label", "manual_notes"], errors="ignore").merge(
                external[["word", "manual_label"]], on="word", how="left"
            )
            review_mode = "manual"
    elif allow_proxy_review:
        review_mode = "proxy"
    else:
        raise RuntimeError(
            "Question-3 strict mode requires manual review labels for 40-50 low-confidence words. "
            "Fill `manual_label` in low_confidence_review_sample.csv and rerun with --manual-review-file."
        )

    if "manual_label" not in review_df.columns:
        review_df["manual_label"] = ""

    if review_mode == "proxy":
        review_df["manual_label"] = review_df.apply(proxy_manual_label, axis=1)
        review_df["manual_notes"] = "proxy_label_from_rule_based_adjudicator"

    reviewed = review_df[review_df["manual_label"].isin(["correct", "incorrect"])].copy()
    reviewed["is_correct_prediction"] = reviewed["predicted_label"] == reviewed["manual_label"]

    right = int(reviewed["is_correct_prediction"].sum()) if not reviewed.empty else 0
    wrong = int((~reviewed["is_correct_prediction"]).sum()) if not reviewed.empty else 0
    total = int(len(reviewed))
    accuracy = round((right / total) * 100.0, 3) if total > 0 else None

    category_breakdown: list[dict[str, Any]] = []
    if total > 0:
        grouped = reviewed.groupby("category")
        for category, frame in grouped:
            cat_total = int(len(frame))
            cat_wrong = int((~frame["is_correct_prediction"]).sum())
            err_rate = round((cat_wrong / cat_total) * 100.0, 3) if cat_total > 0 else 0.0
            category_breakdown.append(
                {
                    "category": category,
                    "reviewed": cat_total,
                    "wrong": cat_wrong,
                    "error_rate_percent": err_rate,
                }
            )
    category_breakdown = sorted(category_breakdown, key=lambda x: (x["error_rate_percent"], x["wrong"]), reverse=True)

    unreliable = [item for item in category_breakdown if item["reviewed"] >= 3][:2]
    if not unreliable:
        unreliable = category_breakdown[:2]

    unreliable_notes = []
    for item in unreliable:
        category = item["category"]
        explanation = CATEGORY_EXPLANATIONS.get(category, "This category shows unstable signals in low-confidence analysis.")
        unreliable_notes.append(
            {
                "category": category,
                "why_unreliable": explanation,
                "reviewed": item["reviewed"],
                "error_rate_percent": item["error_rate_percent"],
            }
        )

    summary = {
        "review_mode": review_mode,
        "reviewed_samples": total,
        "right_predictions": right,
        "wrong_predictions": wrong,
        "accuracy_percent": accuracy,
        "category_breakdown": category_breakdown,
        "unreliable_categories": unreliable_notes,
    }
    return summary, review_df


def build_question3_report(
    total_words: int,
    correct_count: int,
    incorrect_count: int,
    confidence_counts: dict[str, int],
    approach_summary: dict[str, Any],
    review_summary: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Question-3 Report (Word-Level Spelling Quality)")
    lines.append("")
    lines.append("## a) Correct vs incorrect spelling identification approach")
    lines.append("- Built lexical evidence from in-domain transcript word frequencies.")
    lines.append("- Added external Hindi lexical evidence from FLEURS (when enabled).")
    lines.append("- Applied rule-based diagnostics for script compliance and orthographic anomalies.")
    lines.append("- Used nearest-lexicon similarity to flag likely misspellings.")
    lines.append("")
    lines.append("### Classification totals")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total unique words processed | {total_words} |")
    lines.append(f"| Predicted correct spelling | {correct_count} |")
    lines.append(f"| Predicted incorrect spelling | {incorrect_count} |")
    lines.append("")
    lines.append("### Confidence distribution")
    lines.append("| Confidence | Count |")
    lines.append("|---|---:|")
    for level in ["high", "medium", "low"]:
        lines.append(f"| {level} | {confidence_counts.get(level, 0)} |")
    lines.append("")
    lines.append("## b) Confidence score + reason per word")
    lines.append("- Output file: `word_classification_with_confidence.csv`")
    lines.append("- Each row includes `word`, `predicted_label`, `confidence`, and `reason`.")
    lines.append("")
    lines.append("## c) Low-confidence review (40-50 words)")
    lines.append(f"- Review mode: {review_summary['review_mode']}")
    lines.append(f"- Reviewed samples: {review_summary['reviewed_samples']}")
    lines.append(f"- Right predictions: {review_summary['right_predictions']}")
    lines.append(f"- Wrong predictions: {review_summary['wrong_predictions']}")
    lines.append(f"- Accuracy (%): {review_summary['accuracy_percent']}")
    lines.append("")
    lines.append("## d) Categories where system is unreliable")
    unreliable = review_summary.get("unreliable_categories", [])
    if unreliable:
        for item in unreliable:
            lines.append(
                f"- {item['category']}: error_rate={item['error_rate_percent']}% over {item['reviewed']} reviewed words. {item['why_unreliable']}"
            )
    else:
        lines.append("- No unstable categories identified from available reviewed samples.")
    lines.append("")
    lines.append("## Deliverables generated")
    lines.append("- `google_sheet_ready_word_labels.csv` (2 columns: word, spelling_label)")
    lines.append("- `word_classification_with_confidence.csv` (full scoring output)")
    lines.append("- `low_confidence_review_sample.csv` (40-50 words for audit)")
    lines.append("- `low_confidence_review_analysis.json`")
    lines.append("- `summary.json`")
    lines.append("")
    return "\n".join(lines)


def run_pipeline(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_csv = args.manifest_csv if args.manifest_csv else default_manifest_path(repo_root)
    work_dir = ensure_dir(args.work_dir.resolve())

    use_fleurs_lexicon = bool(args.use_fleurs_lexicon and not args.disable_fleurs_lexicon)
    if not (40 <= int(args.review_sample_size) <= 50):
        raise ValueError("Question-3 requires --review-sample-size between 40 and 50.")

    print("Step 1) Build in-domain transcript evidence...")
    segments_df, preprocess_summary = preprocess_dataset(
        manifest_csv=manifest_csv,
        work_dir=work_dir,
        min_segment_sec=args.min_segment_sec,
        max_segment_sec=args.max_segment_sec,
        max_recordings=args.max_recordings,
        force_rebuild=args.force_rebuild,
    )
    domain_freq = build_in_domain_frequency(segments_df)

    print("Step 2) Load target word list...")
    if args.wordlist_file and args.wordlist_file.exists():
        words = load_word_list(args.wordlist_file.resolve(), args.word_column)
        word_source = args.wordlist_file.resolve().as_posix()
    else:
        words = sorted(domain_freq.keys())
        word_source = "derived_from_transcript_corpus"

    if not words:
        raise RuntimeError(
            "No words available for Q3 classification. Check `--wordlist-file` contents or corpus preprocessing output."
        )

    print("Step 3) Build lexical evidence sources...")
    domain_lexicon, fleurs_lexicon, reference_lexicon = build_reference_lexicons(
        domain_freq=domain_freq,
        use_fleurs_lexicon=use_fleurs_lexicon,
        max_fleurs_per_split=args.max_fleurs_per_split,
    )

    print("Step 4) Classify words with confidence + reason...")
    classified_df = classify_all_words(
        words=words,
        domain_freq=domain_freq,
        domain_lexicon=domain_lexicon,
        fleurs_lexicon=fleurs_lexicon,
        reference_lexicon=reference_lexicon,
    )
    classified_df = classified_df.sort_values(["word"]).reset_index(drop=True)

    classification_path = work_dir / "word_classification_with_confidence.csv"
    classified_df.to_csv(classification_path, index=False)

    google_sheet_df = classified_df[["word", "predicted_label"]].rename(columns={"predicted_label": "spelling_label"})
    google_sheet_path = work_dir / "google_sheet_ready_word_labels.csv"
    google_sheet_df.to_csv(google_sheet_path, index=False)

    print("Step 5) Prepare 40-50 low-confidence review sample...")
    review_sample_df = build_low_confidence_review_sample(classified_df, args.review_sample_size)
    if len(review_sample_df) < 40:
        raise RuntimeError(
            f"Question-3 requires reviewing 40-50 low-confidence words, but only {len(review_sample_df)} were available. "
            "Increase input word coverage or adjust classification sensitivity."
        )
    review_sample_path = work_dir / "low_confidence_review_sample.csv"
    review_sample_df.to_csv(review_sample_path, index=False)
    if not args.allow_proxy_review and (not args.manual_review_file or not args.manual_review_file.exists()):
        raise RuntimeError(
            f"Low-confidence review sample generated at {review_sample_path}. "
            "Fill `manual_label` (correct/incorrect) for 40-50 rows, then rerun with "
            f"--manual-review-file \"{review_sample_path}\"."
        )

    print("Step 6) Analyze low-confidence review outcomes...")
    review_summary, review_full_df = analyze_low_confidence_review(
        classified_df=classified_df,
        review_sample_df=review_sample_df,
        manual_review_file=args.manual_review_file.resolve() if args.manual_review_file else None,
        allow_proxy_review=bool(args.allow_proxy_review),
    )
    if review_summary["review_mode"] == "manual" and int(review_summary["reviewed_samples"]) < 40:
        raise RuntimeError(
            f"Question-3 requires 40-50 manually reviewed low-confidence words, but only "
            f"{review_summary['reviewed_samples']} labeled rows were found."
        )
    review_full_df.to_csv(work_dir / "low_confidence_review_with_labels.csv", index=False)
    (work_dir / "low_confidence_review_analysis.json").write_text(
        json.dumps(review_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_words = int(len(classified_df))
    correct_count = int((classified_df["predicted_label"] == "correct").sum())
    incorrect_count = int((classified_df["predicted_label"] == "incorrect").sum())
    confidence_counts = classified_df["confidence"].value_counts().to_dict()

    approach_summary = {
        "word_source": word_source,
        "use_fleurs_lexicon": use_fleurs_lexicon,
        "domain_lexicon_size": len(domain_lexicon),
        "fleurs_lexicon_size": len(fleurs_lexicon),
        "reference_lexicon_size": len(reference_lexicon),
        "recordings_requested": preprocess_summary["recordings_requested"],
        "segments_kept": preprocess_summary["segments_kept"],
    }

    summary_payload = {
        "total_unique_words_processed": total_words,
        "predicted_correct_spelling_count": correct_count,
        "predicted_incorrect_spelling_count": incorrect_count,
        "confidence_distribution": confidence_counts,
        "approach_summary": approach_summary,
        "review_summary": review_summary,
    }
    (work_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report = build_question3_report(
        total_words=total_words,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        confidence_counts=confidence_counts,
        approach_summary=approach_summary,
        review_summary=review_summary,
    )
    report_path = work_dir / "question3_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("\nCompleted Question-3 pipeline.")
    print(f"Artifacts written to: {work_dir}")
    print(f"Main report: {report_path}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
