import argparse
import json
import re
from pathlib import Path
from typing import Any

import librosa
import pandas as pd
from jiwer import wer as jiwer_wer
from tqdm import tqdm

from .question1 import (
    default_manifest_path,
    ensure_dir,
    generate_text,
    get_device,
    load_inference_bundle,
    normalize_text,
    preprocess_dataset,
)


DEVANAGARI_PUNCT = ".,!?;:()[]{}\"'`"

TOKEN_CANONICAL_OVERRIDES = {
    "हज़ार": "हजार",
    "करोड़": "करोड",
    "पन्द्रह": "पंद्रह",
    "चौतीस": "चौंतीस",
    "अठ्ठावन": "अट्ठावन",
    "दोनो": "दोनों",
}

SMALL_NUMBER_WORDS = {"एक", "दो", "तीन", "चार", "पांच", "पाँच", "छह", "सात", "आठ", "नौ"}
RANGE_CONTEXT_WORDS = {"बात", "बाते", "बातें", "चीज", "चीजें", "लोग", "दिन", "बार", "दफा", "कदम"}

HINDI_NUM_WORDS_0_99 = {
    "शून्य": 0,
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पांच": 5,
    "पाँच": 5,
    "छह": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
    "ग्यारह": 11,
    "बारह": 12,
    "तेरह": 13,
    "चौदह": 14,
    "पंद्रह": 15,
    "सोलह": 16,
    "सत्रह": 17,
    "अठारह": 18,
    "उन्नीस": 19,
    "बीस": 20,
    "इक्कीस": 21,
    "बाईस": 22,
    "तेईस": 23,
    "चौबीस": 24,
    "पच्चीस": 25,
    "छब्बीस": 26,
    "सत्ताईस": 27,
    "अट्ठाईस": 28,
    "उनतीस": 29,
    "तीस": 30,
    "इकतीस": 31,
    "बत्तीस": 32,
    "तैंतीस": 33,
    "चौंतीस": 34,
    "पैंतीस": 35,
    "छत्तीस": 36,
    "सैंतीस": 37,
    "अड़तीस": 38,
    "उनतालीस": 39,
    "चालीस": 40,
    "इकतालीस": 41,
    "बयालीस": 42,
    "तैंतालीस": 43,
    "चवालीस": 44,
    "पैंतालीस": 45,
    "छियालीस": 46,
    "सैंतालीस": 47,
    "अड़तालीस": 48,
    "उनचास": 49,
    "पचास": 50,
    "इक्यावन": 51,
    "बावन": 52,
    "तिरपन": 53,
    "चौवन": 54,
    "पचपन": 55,
    "छप्पन": 56,
    "सत्तावन": 57,
    "अट्ठावन": 58,
    "उनसठ": 59,
    "साठ": 60,
    "इकसठ": 61,
    "बासठ": 62,
    "तिरसठ": 63,
    "चौंसठ": 64,
    "पैंसठ": 65,
    "छियासठ": 66,
    "सड़सठ": 67,
    "अड़सठ": 68,
    "उनहत्तर": 69,
    "सत्तर": 70,
    "इकहत्तर": 71,
    "बहत्तर": 72,
    "तिहत्तर": 73,
    "चौहत्तर": 74,
    "पचहत्तर": 75,
    "छिहत्तर": 76,
    "सतहत्तर": 77,
    "अठहत्तर": 78,
    "उन्नासी": 79,
    "अस्सी": 80,
    "इक्यासी": 81,
    "बयासी": 82,
    "तिरासी": 83,
    "चौरासी": 84,
    "पचासी": 85,
    "छियासी": 86,
    "सतासी": 87,
    "अट्ठासी": 88,
    "नवासी": 89,
    "नब्बे": 90,
    "इक्यानवे": 91,
    "बानवे": 92,
    "तिरानवे": 93,
    "चौरानवे": 94,
    "पंचानवे": 95,
    "छियानवे": 96,
    "सत्तानवे": 97,
    "अट्ठानवे": 98,
    "निन्यानवे": 99,
}

NUMBER_SCALE_WORDS = {
    "सौ": 100,
    "हजार": 1_000,
    "लाख": 100_000,
    "करोड": 10_000_000,
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

EDGE_REASON_EXPLANATIONS = {
    "hyphenated_range": "Kept as-is because this pattern usually expresses an approximate range (e.g., दो-चार) rather than a strict numeric value.",
    "range_like_small_numbers": "Kept as-is because adjacent small numbers often indicate a loose range (e.g., दो तीन), not a single exact number.",
    "idiomatic_small_number_context": "Kept as-is because the number likely appears in an idiomatic phrase where digit conversion can distort meaning.",
    "vague_quantifier_context": "Kept as-is because quantifiers like 'कुछ/कोई' + small number are often non-literal approximations.",
    "parse_failed": "Skipped conversion because the token span could not be parsed confidently into a valid numeric expression.",
    "conversion_increased_error_against_reference": "Marked as tricky because numeric conversion did not align better with the human reference in this sample.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Question-2 ASR cleanup pipeline.")
    parser.add_argument("--manifest-csv", type=Path, default=None, help="Path to FT Data CSV.")
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/q2"), help="Output directory for Q2 artifacts.")
    parser.add_argument("--model-name", type=str, default="openai/whisper-small", help="Pretrained model used for raw ASR.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device.")
    parser.add_argument("--max-recordings", type=int, default=0, help="Debug limiter for recordings. 0 means all.")
    parser.add_argument("--max-segments", type=int, default=0, help="Debug limiter for segments. 0 means all.")
    parser.add_argument("--min-segment-sec", type=float, default=1.0, help="Drop segments shorter than this.")
    parser.add_argument("--max-segment-sec", type=float, default=30.0, help="Drop segments longer than this.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild preprocessed segment manifest.")
    parser.add_argument("--force-asr", action="store_true", help="Regenerate raw ASR even if cached file exists.")
    return parser.parse_args()


def canonicalize_token(token: str) -> str:
    token = token.strip()
    token = token.strip(DEVANAGARI_PUNCT)
    token = token.replace("।", "")
    token = TOKEN_CANONICAL_OVERRIDES.get(token, token)
    return token


def is_number_token(token: str) -> bool:
    if not token:
        return False
    if token.isdigit():
        return True
    if token in HINDI_NUM_WORDS_0_99:
        return True
    if token in NUMBER_SCALE_WORDS:
        return True
    if token == "और":
        return True
    return False


def parse_hindi_number_phrase(tokens: list[str]) -> int | None:
    if not tokens:
        return None

    total = 0
    current = 0
    seen = False

    for token in tokens:
        if token == "और":
            continue
        if token.isdigit():
            current += int(token)
            seen = True
            continue
        if token in HINDI_NUM_WORDS_0_99:
            current += HINDI_NUM_WORDS_0_99[token]
            seen = True
            continue
        if token == "सौ":
            current = (current if current > 0 else 1) * 100
            seen = True
            continue
        if token in {"हजार", "लाख", "करोड"}:
            scale = NUMBER_SCALE_WORDS[token]
            current = (current if current > 0 else 1) * scale
            total += current
            current = 0
            seen = True
            continue
        return None

    if not seen:
        return None
    return total + current


def should_skip_number_phrase(
    phrase_tokens: list[str],
    source_tokens: list[str],
    start_idx: int,
    end_idx: int,
) -> tuple[bool, str]:
    if not phrase_tokens:
        return True, "empty_phrase"

    if len(phrase_tokens) == 2 and phrase_tokens[0] in SMALL_NUMBER_WORDS and phrase_tokens[1] in SMALL_NUMBER_WORDS:
        return True, "range_like_small_numbers"

    next_token = canonicalize_token(source_tokens[end_idx]) if end_idx < len(source_tokens) else ""
    if len(phrase_tokens) == 1 and phrase_tokens[0] in SMALL_NUMBER_WORDS and next_token in RANGE_CONTEXT_WORDS:
        return True, "idiomatic_small_number_context"

    prev_token = canonicalize_token(source_tokens[start_idx - 1]) if start_idx > 0 else ""
    if prev_token in {"कोई", "कुछ"} and len(phrase_tokens) == 1 and phrase_tokens[0] in SMALL_NUMBER_WORDS:
        return True, "vague_quantifier_context"

    return False, ""


def normalize_numbers_with_trace(text: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = normalize_text(text)
    source_tokens = normalized.split()
    output_tokens: list[str] = []
    conversions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    i = 0
    while i < len(source_tokens):
        original_token = source_tokens[i]
        canonical = canonicalize_token(original_token)

        if "-" in canonical:
            parts = [canonicalize_token(part) for part in canonical.split("-")]
            if parts and all(part in SMALL_NUMBER_WORDS or part in HINDI_NUM_WORDS_0_99 for part in parts):
                output_tokens.append(original_token)
                skipped.append(
                    {
                        "span_text": original_token,
                        "reason": "hyphenated_range",
                        "start_token_index": i,
                    }
                )
                i += 1
                continue

        if not is_number_token(canonical):
            output_tokens.append(original_token)
            i += 1
            continue

        j = i
        phrase_raw: list[str] = []
        phrase_canonical: list[str] = []
        while j < len(source_tokens):
            token_j = source_tokens[j]
            canonical_j = canonicalize_token(token_j)
            if not is_number_token(canonical_j):
                break
            phrase_raw.append(token_j)
            phrase_canonical.append(canonical_j)
            j += 1

        phrase_without_connector = [token for token in phrase_canonical if token != "और"]
        skip, reason = should_skip_number_phrase(phrase_without_connector, source_tokens, i, j)
        if skip:
            output_tokens.extend(phrase_raw)
            skipped.append(
                {
                    "span_text": " ".join(phrase_raw),
                    "reason": reason,
                    "start_token_index": i,
                }
            )
            i = j
            continue

        value = parse_hindi_number_phrase(phrase_canonical)
        if value is None:
            output_tokens.extend(phrase_raw)
            skipped.append(
                {
                    "span_text": " ".join(phrase_raw),
                    "reason": "parse_failed",
                    "start_token_index": i,
                }
            )
            i = j
            continue

        output_tokens.append(str(value))
        conversions.append(
            {
                "source_span": " ".join(phrase_raw),
                "normalized_value": str(value),
                "start_token_index": i,
                "end_token_index": j - 1,
            }
        )
        i = j

    return " ".join(output_tokens), conversions, skipped


def is_english_word(token: str) -> tuple[bool, str]:
    if re.search(r"[A-Za-z]", token):
        return True, "roman_script"

    canonical = canonicalize_token(token)
    if canonical in DEVANAGARI_ENGLISH_LEXICON:
        return True, "devanagari_loanword_lexicon"

    return False, ""


def tag_english_words(text: str) -> tuple[str, list[dict[str, Any]]]:
    tokens = normalize_text(text).split()
    tagged_tokens: list[str] = []
    detections: list[dict[str, Any]] = []

    for idx, token in enumerate(tokens):
        is_english, reason = is_english_word(token)
        if is_english:
            tagged_tokens.append(f"[EN]{token}[/EN]")
            detections.append(
                {
                    "token": token,
                    "token_index": idx,
                    "reason": reason,
                }
            )
        else:
            tagged_tokens.append(token)
    return " ".join(tagged_tokens), detections


def transcribe_raw_asr(
    segments_df: pd.DataFrame,
    model_name: str,
    device_arg: str,
    work_dir: Path,
    max_segments: int,
    force_asr: bool,
) -> pd.DataFrame:
    raw_pairs_path = work_dir / "raw_asr_pairs.csv"

    if raw_pairs_path.exists() and not force_asr:
        raw_df = pd.read_csv(raw_pairs_path)
        if max_segments > 0:
            raw_df = raw_df.head(max_segments).copy()
        return raw_df

    segments = segments_df.copy()
    if max_segments > 0:
        segments = segments.head(max_segments).copy()

    device = get_device(device_arg)
    processor, model = load_inference_bundle(model_name, device)
    device_state: dict[str, Any] = {"device": device}

    rows: list[dict[str, Any]] = []
    for row in tqdm(segments.to_dict(orient="records"), desc=f"Generating raw ASR ({model_name})"):
        audio_array, sr = librosa.load(row["audio_path"], sr=16000, mono=True)
        raw_asr = generate_text(processor, model, audio_array, sr, device_state)
        rows.append(
            {
                "recording_id": int(row["recording_id"]),
                "segment_id": int(row["segment_id"]),
                "duration_sec": float(row["duration_sec"]),
                "audio_path": row["audio_path"],
                "reference_text": row["text"],
                "raw_asr_text": raw_asr,
            }
        )

    raw_df = pd.DataFrame(rows).sort_values(["recording_id", "segment_id"]).reset_index(drop=True)
    raw_df.to_csv(raw_pairs_path, index=False)
    return raw_df


def build_number_examples(processed_df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_conversions = processed_df[processed_df["num_conversion_count"] > 0].copy()
    all_conversions["improvement"] = all_conversions["per_sample_wer_before"] - all_conversions["per_sample_wer_after"]
    improved_conversions = all_conversions[all_conversions["improvement"] > 0].copy()
    improved_conversions = improved_conversions.sort_values("improvement", ascending=False)
    all_conversions = all_conversions.sort_values("improvement", ascending=False)

    def parse_json_list(raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, str) and raw.strip():
            try:
                value = json.loads(raw)
                if isinstance(value, list):
                    return value
            except Exception:
                return []
        return []

    good_examples: list[dict[str, Any]] = []
    used_keys: set[tuple[int, int]] = set()

    for _, row in improved_conversions.iterrows():
        if len(good_examples) >= 5:
            break
        key = (int(row["recording_id"]), int(row["segment_id"]))
        used_keys.add(key)
        good_examples.append(
            {
                "recording_id": int(row["recording_id"]),
                "segment_id": int(row["segment_id"]),
                "before": row["raw_asr_norm"],
                "after": row["number_normalized_text"],
                "reference": row["reference_norm"],
                "conversions": parse_json_list(row["num_conversions_json"]),
                "selection_reason": "wer_improved_after_conversion",
            }
        )

    # Guarantee 4-5 examples when possible, even if some are not clear WER improvements.
    if len(good_examples) < 4:
        for _, row in all_conversions.iterrows():
            if len(good_examples) >= 5:
                break
            key = (int(row["recording_id"]), int(row["segment_id"]))
            if key in used_keys:
                continue
            used_keys.add(key)
            good_examples.append(
                {
                    "recording_id": int(row["recording_id"]),
                    "segment_id": int(row["segment_id"]),
                    "before": row["raw_asr_norm"],
                    "after": row["number_normalized_text"],
                    "reference": row["reference_norm"],
                    "conversions": parse_json_list(row["num_conversions_json"]),
                    "selection_reason": "fallback_conversion_example",
                }
            )

    edge_examples: list[dict[str, Any]] = []
    skipped_candidates = processed_df[processed_df["num_skips_json"] != "[]"]
    for _, row in skipped_candidates.head(3).iterrows():
        decisions = parse_json_list(row["num_skips_json"])
        for decision in decisions:
            reason = decision.get("reason", "")
            decision["reasoning"] = EDGE_REASON_EXPLANATIONS.get(reason, "Context-sensitive case; kept conservative to avoid semantic errors.")
        edge_examples.append(
            {
                "recording_id": int(row["recording_id"]),
                "segment_id": int(row["segment_id"]),
                "before": row["raw_asr_norm"],
                "after": row["number_normalized_text"],
                "reference": row["reference_norm"],
                "edge_case_decision": decisions,
            }
        )

    if len(edge_examples) < 2:
        worsened = processed_df[
            (processed_df["num_conversion_count"] > 0) & (processed_df["per_sample_wer_after"] > processed_df["per_sample_wer_before"])
        ].copy()
        worsened = worsened.sort_values(
            "per_sample_wer_after",
            ascending=False,
        )
        for _, row in worsened.iterrows():
            if len(edge_examples) >= 3:
                break
            edge_examples.append(
                {
                    "recording_id": int(row["recording_id"]),
                    "segment_id": int(row["segment_id"]),
                    "before": row["raw_asr_norm"],
                    "after": row["number_normalized_text"],
                    "reference": row["reference_norm"],
                    "edge_case_decision": [
                        {
                            "reason": "conversion_increased_error_against_reference",
                            "reasoning": EDGE_REASON_EXPLANATIONS["conversion_increased_error_against_reference"],
                            "conversions": parse_json_list(row["num_conversions_json"]),
                        }
                    ],
                }
            )

    return good_examples[:5], edge_examples[:3]


def build_english_examples(processed_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    positive_examples: list[dict[str, Any]] = []
    ambiguous_examples: list[dict[str, Any]] = []

    candidates = processed_df[processed_df["english_token_count"] > 0]
    for _, row in candidates.iterrows():
        detections = json.loads(row["english_tokens_json"])
        ref_tokens = set(normalize_text(row["reference_norm"]).split())

        matched = []
        unmatched = []
        for detection in detections:
            token = detection.get("token", "")
            if canonicalize_token(token) in ref_tokens:
                matched.append(detection)
            else:
                unmatched.append(detection)

        record = {
            "recording_id": int(row["recording_id"]),
            "segment_id": int(row["segment_id"]),
            "input": row["number_normalized_text"],
            "tagged_output": row["english_tagged_text"],
            "matched_tokens_proxy": matched,
            "unmatched_tokens_proxy": unmatched,
        }

        if matched and len(positive_examples) < 5:
            positive_examples.append(record)
        if unmatched and len(ambiguous_examples) < 3:
            ambiguous_examples.append(record)

        if len(positive_examples) >= 5 and len(ambiguous_examples) >= 3:
            break

    return {
        "positive_examples": positive_examples,
        "ambiguous_examples": ambiguous_examples,
    }


def process_cleanup_pipeline(raw_df: pd.DataFrame, work_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    ref_all: list[str] = []
    before_all: list[str] = []
    after_all: list[str] = []

    english_match_proxy_total = 0
    english_unmatched_proxy_total = 0

    for row in tqdm(raw_df.to_dict(orient="records"), desc="Applying Q2 cleanup pipeline"):
        reference_norm = normalize_text(row["reference_text"])
        raw_asr_norm = normalize_text(row["raw_asr_text"])

        number_normalized_text, conversions, skips = normalize_numbers_with_trace(raw_asr_norm)
        english_tagged_text, english_tokens = tag_english_words(number_normalized_text)

        ref_token_set = set(reference_norm.split())
        english_match_proxy = sum(
            1 for item in english_tokens if canonicalize_token(item.get("token", "")) in ref_token_set
        )
        english_unmatched_proxy = len(english_tokens) - english_match_proxy
        english_match_proxy_total += english_match_proxy
        english_unmatched_proxy_total += english_unmatched_proxy

        before_wer = float(jiwer_wer([reference_norm], [raw_asr_norm]))
        after_wer = float(jiwer_wer([reference_norm], [number_normalized_text]))

        ref_all.append(reference_norm)
        before_all.append(raw_asr_norm)
        after_all.append(number_normalized_text)

        rows.append(
            {
                "recording_id": int(row["recording_id"]),
                "segment_id": int(row["segment_id"]),
                "duration_sec": float(row["duration_sec"]),
                "reference_text": row["reference_text"],
                "raw_asr_text": row["raw_asr_text"],
                "reference_norm": reference_norm,
                "raw_asr_norm": raw_asr_norm,
                "number_normalized_text": number_normalized_text,
                "english_tagged_text": english_tagged_text,
                "num_conversion_count": len(conversions),
                "num_skip_count": len(skips),
                "english_token_count": len(english_tokens),
                "english_match_proxy_count": english_match_proxy,
                "english_unmatched_proxy_count": english_unmatched_proxy,
                "num_conversions_json": json.dumps(conversions, ensure_ascii=False),
                "num_skips_json": json.dumps(skips, ensure_ascii=False),
                "english_tokens_json": json.dumps(english_tokens, ensure_ascii=False),
                "per_sample_wer_before": before_wer,
                "per_sample_wer_after": after_wer,
            }
        )

    processed_df = pd.DataFrame(rows)
    processed_df.to_csv(work_dir / "q2_processed_transcripts.csv", index=False)

    before_corpus_wer = 100.0 * jiwer_wer(ref_all, before_all) if ref_all else 0.0
    after_corpus_wer = 100.0 * jiwer_wer(ref_all, after_all) if ref_all else 0.0

    improved = int((processed_df["per_sample_wer_after"] < processed_df["per_sample_wer_before"]).sum())
    worsened = int((processed_df["per_sample_wer_after"] > processed_df["per_sample_wer_before"]).sum())
    unchanged = int((processed_df["per_sample_wer_after"] == processed_df["per_sample_wer_before"]).sum())

    summary = {
        "segments_processed": int(len(processed_df)),
        "number_conversion_segments": int((processed_df["num_conversion_count"] > 0).sum()),
        "total_number_conversions": int(processed_df["num_conversion_count"].sum()),
        "total_number_skips": int(processed_df["num_skip_count"].sum()),
        "segments_with_english_tags": int((processed_df["english_token_count"] > 0).sum()),
        "total_english_tokens_tagged": int(processed_df["english_token_count"].sum()),
        "english_tokens_matched_reference_proxy": int(english_match_proxy_total),
        "english_tokens_unmatched_reference_proxy": int(english_unmatched_proxy_total),
        "wer_before_number_normalization_percent": round(before_corpus_wer, 3),
        "wer_after_number_normalization_percent": round(after_corpus_wer, 3),
        "wer_delta_percent": round(after_corpus_wer - before_corpus_wer, 3),
        "samples_improved_by_number_normalization": improved,
        "samples_worsened_by_number_normalization": worsened,
        "samples_unchanged_by_number_normalization": unchanged,
    }
    return processed_df, summary


def build_report(
    preprocess_summary: dict[str, Any],
    summary: dict[str, Any],
    good_examples: list[dict[str, Any]],
    edge_examples: list[dict[str, Any]],
    english_examples: dict[str, list[dict[str, Any]]],
) -> str:
    lines: list[str] = []
    lines.append("# Question-2 Report (ASR Cleanup Pipeline)")
    lines.append("")
    lines.append("## Data and raw ASR generation")
    lines.append("- Used the same Josh Talks Hindi dataset and segment-level references from transcription JSON.")
    lines.append("- Generated raw ASR using pretrained `openai/whisper-small` (before fine-tuning).")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Recordings requested | {preprocess_summary['recordings_requested']} |")
    lines.append(f"| Segments kept after preprocessing | {preprocess_summary['segments_kept']} |")
    lines.append(f"| Hours kept | {preprocess_summary['hours_kept']} |")
    lines.append(f"| Segments processed in Q2 | {summary['segments_processed']} |")
    lines.append("")
    lines.append("## a) Number normalization")
    lines.append("- Pipeline converts Hindi number words to digits for simple and compound forms.")
    lines.append("- Guardrails skip likely idiomatic/range uses such as hyphenated ranges and vague quantifier contexts.")
    lines.append("")
    lines.append("| Number Normalization Impact | Value |")
    lines.append("|---|---:|")
    lines.append(
        f"| WER before (%) | {summary['wer_before_number_normalization_percent']} |"
    )
    lines.append(
        f"| WER after (%) | {summary['wer_after_number_normalization_percent']} |"
    )
    lines.append(f"| WER delta (%) | {summary['wer_delta_percent']} |")
    lines.append(f"| Samples improved | {summary['samples_improved_by_number_normalization']} |")
    lines.append(f"| Samples worsened | {summary['samples_worsened_by_number_normalization']} |")
    lines.append(f"| Samples unchanged | {summary['samples_unchanged_by_number_normalization']} |")
    lines.append("")
    lines.append("### 4-5 correct conversion examples from actual data")
    for example in good_examples:
        lines.append(
            f"- rec={example['recording_id']}, seg={example['segment_id']} | before: `{example['before']}` | after: `{example['after']}` | ref: `{example['reference']}` | note: `{example.get('selection_reason', 'converted_example')}`"
        )
    if not good_examples:
        lines.append("- No clear improvement examples found in this run; inspect `q2_processed_transcripts.csv` for manual review.")
    lines.append("")
    lines.append("### 2-3 edge cases and judgment calls")
    for example in edge_examples:
        decisions = example["edge_case_decision"]
        reasons = "; ".join(
            f"{item.get('reason', 'unknown')}: {item.get('reasoning', '')}" for item in decisions
        )
        lines.append(
            f"- rec={example['recording_id']}, seg={example['segment_id']} | before: `{example['before']}` | after: `{example['after']}` | decision: `{reasons}`"
        )
    if not edge_examples:
        lines.append("- No edge-case skips were captured; inspect `num_skips_json` column in `q2_processed_transcripts.csv`.")
    lines.append("")
    lines.append("## b) English word detection")
    lines.append("- Tagged English words with `[EN]...[/EN]` using Roman-script detection and Devanagari loanword lexicon.")
    lines.append("- This keeps Devanagari spellings valid while still marking code-switched English words.")
    lines.append("")
    lines.append("| English Detection Stats | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Segments with English tags | {summary['segments_with_english_tags']} |")
    lines.append(f"| Total English tokens tagged | {summary['total_english_tokens_tagged']} |")
    lines.append(f"| Matched reference (proxy) | {summary['english_tokens_matched_reference_proxy']} |")
    lines.append(f"| Unmatched reference (proxy) | {summary['english_tokens_unmatched_reference_proxy']} |")
    lines.append("")
    lines.append("### Tagged transcript examples (likely helpful)")
    for example in english_examples.get("positive_examples", []):
        lines.append(
            f"- rec={example['recording_id']}, seg={example['segment_id']} | input: `{example['input']}` | output: `{example['tagged_output']}`"
        )
    if not english_examples.get("positive_examples"):
        lines.append("- No English tokens detected with current rules; extend lexicon and rerun.")
    lines.append("")
    lines.append("### Tagged transcript examples (potentially harmful/ambiguous)")
    for example in english_examples.get("ambiguous_examples", []):
        lines.append(
            f"- rec={example['recording_id']}, seg={example['segment_id']} | input: `{example['input']}` | output: `{example['tagged_output']}`"
        )
    if not english_examples.get("ambiguous_examples"):
        lines.append("- No ambiguous English detections found with current proxy.")
    lines.append("")
    return "\n".join(lines)


def run_pipeline(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_csv = args.manifest_csv if args.manifest_csv else default_manifest_path(repo_root)
    work_dir = ensure_dir(args.work_dir.resolve())

    print("Step 1) Preprocess and segment dataset...")
    segments_df, preprocess_summary = preprocess_dataset(
        manifest_csv=manifest_csv,
        work_dir=work_dir,
        min_segment_sec=args.min_segment_sec,
        max_segment_sec=args.max_segment_sec,
        max_recordings=args.max_recordings,
        force_rebuild=args.force_rebuild,
    )
    if args.max_segments > 0:
        segments_df = segments_df.head(args.max_segments).copy()

    print("Step 2) Generate raw ASR using pretrained Whisper-small...")
    raw_df = transcribe_raw_asr(
        segments_df=segments_df,
        model_name=args.model_name,
        device_arg=args.device,
        work_dir=work_dir,
        max_segments=args.max_segments,
        force_asr=args.force_asr,
    )

    print("Step 3) Apply number normalization and English-word tagging...")
    processed_df, summary = process_cleanup_pipeline(raw_df, work_dir)

    print("Step 4) Build required examples and final report...")
    good_examples, edge_examples = build_number_examples(processed_df)
    english_examples = build_english_examples(processed_df)

    (work_dir / "number_normalization_examples.json").write_text(
        json.dumps({"correct_conversion_examples": good_examples, "edge_case_examples": edge_examples}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (work_dir / "english_detection_examples.json").write_text(
        json.dumps(english_examples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report = build_report(preprocess_summary, summary, good_examples, edge_examples, english_examples)
    report_path = work_dir / "question2_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("\nCompleted Question-2 pipeline.")
    print(f"Artifacts written to: {work_dir}")
    print(f"Main report: {report_path}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
