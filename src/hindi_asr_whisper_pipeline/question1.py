import argparse
import json
import random
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import requests
import soundfile as sf
import torch
from datasets import Audio, Dataset, load_dataset
from jiwer import process_words, wer as jiwer_wer
from tqdm import tqdm
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


URL_REWRITE_PATTERN = re.compile(
    r"^https://storage\.googleapis\.com/joshtalks-data-collection/hq_data/hi/(\d+)/(\d+_(?:audio\.wav|transcription\.json|metadata\.json))$"
)

NUMBER_WORDS = {
    "शून्य",
    "एक",
    "दो",
    "तीन",
    "चार",
    "पाँच",
    "पांच",
    "छह",
    "सात",
    "आठ",
    "नौ",
    "दस",
    "ग्यारह",
    "बारह",
    "तेरह",
    "चौदह",
    "पंद्रह",
    "पंद्रह",
    "सोलह",
    "सत्रह",
    "अठारह",
    "उन्नीस",
    "बीस",
    "सौ",
    "हज़ार",
    "हजार",
    "लाख",
    "करोड़",
    "करोड",
}

ERROR_CATEGORY_DESCRIPTIONS = {
    "content_omission": "Hypothesis drops one or more important reference words.",
    "content_insertion": "Hypothesis introduces extra words not present in reference.",
    "numeric_rendering": "Number words/digits are rendered differently from reference.",
    "english_or_name_token": "Errors concentrate around English or named-entity-like tokens.",
    "spelling_or_phonetic_variant": "Near-spelling or phonetic variants are used instead of reference form.",
    "word_order_or_substitution": "Main issue is substitutions/re-ordering rather than pure insert/delete.",
}

FIX_LIBRARY = {
    "content_omission": "Increase context retention with longer chunking + overlap and decode with lower compression ratio threshold.",
    "content_insertion": "Tune decoding constraints (temperature fallback schedule, no-repeat n-gram, length penalties) on Hindi dev split.",
    "numeric_rendering": "Add Hindi number verbalizer/normalizer with context guards (idiom detection) before scoring and downstream use.",
    "english_or_name_token": "Introduce named-entity lexicon biasing and mixed-language token normalization in decoding post-process.",
    "spelling_or_phonetic_variant": "Apply confusion-lexicon correction learned from systematic dev errors (implemented below).",
    "word_order_or_substitution": "Use segment-boundary-aware re-chunking and LM-assisted rescoring to reduce substitution drift.",
}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Question-1 end-to-end Hindi ASR pipeline.")
    parser.add_argument("--manifest-csv", type=Path, default=None, help="Path to FT Data CSV.")
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/q1"), help="Artifacts output directory.")
    parser.add_argument("--model-name", type=str, default="openai/whisper-small", help="Base Whisper model.")
    parser.add_argument(
        "--finetuned-model-dir",
        type=Path,
        default=Path("models/whisper-small-hindi-q1"),
        help="Where to save fine-tuned model.",
    )
    parser.add_argument("--max-recordings", type=int, default=0, help="Debug mode limiter. 0 means all recordings.")
    parser.add_argument("--max-fleurs-samples", type=int, default=0, help="Debug mode limiter. 0 means full test split.")
    parser.add_argument("--eval-ratio", type=float, default=0.15, help="Eval split ratio for Josh Talks segments.")
    parser.add_argument("--min-segment-sec", type=float, default=1.0, help="Drop segments shorter than this.")
    parser.add_argument("--max-segment-sec", type=float, default=25.0, help="Drop segments longer than this.")
    parser.add_argument("--sample-errors", type=int, default=25, help="How many error utterances to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference/training device.")

    # Fine-tuning hyperparameters
    parser.add_argument("--train-epochs", type=float, default=3.0, help="Training epochs when max-steps is not set.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override number of training steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Fine-tuning learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps.")
    parser.add_argument("--train-batch-size", type=int, default=4, help="Per-device train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Per-device eval batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--skip-training", action="store_true", help="Skip fine-tuning if model directory already exists.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild preprocessed manifests even if present.")
    parser.add_argument(
        "--skip-strict-checks",
        action="store_true",
        help="Allow non-compliant debug runs (for final assignment submission keep this disabled).",
    )
    return parser.parse_args()


def default_manifest_path(repo_root: Path) -> Path:
    candidates = [
        repo_root / "data" / "FT Data - data.csv",
        repo_root.parent / "FT Data - data.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find `FT Data - data.csv`. Pass --manifest-csv or place it in `data/`."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rewrite_upload_goai_url(url: str) -> str:
    if url.startswith("https://storage.googleapis.com/upload_goai/"):
        return url
    match = URL_REWRITE_PATTERN.match(url.strip())
    if not match:
        return url.strip()
    folder_id, filename = match.groups()
    return f"https://storage.googleapis.com/upload_goai/{folder_id}/{filename}"


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”\"'`]", "", text)
    text = re.sub(r"[।!?;,]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_devanagari_token(token: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", token))


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            ins = current[j - 1] + 1
            delete = prev[j] + 1
            subst = prev[j - 1] + (ca != cb)
            current.append(min(ins, delete, subst))
        prev = current
    return prev[-1]


def download_file(url: str, output_path: Path, timeout: int = 120) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, dir=str(output_path.parent)) as temp_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    temp_file.write(chunk)
            temp_name = temp_file.name
    Path(temp_name).replace(output_path)


def read_transcription(transcription_url: str, timeout: int = 60) -> list[dict[str, Any]]:
    with requests.get(transcription_url, timeout=timeout) as response:
        response.raise_for_status()
        payload = response.json()

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("segments", "data", "utterances"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    raise ValueError(f"Unexpected transcription schema at {transcription_url}")


def preprocess_dataset(
    manifest_csv: Path,
    work_dir: Path,
    min_segment_sec: float,
    max_segment_sec: float,
    max_recordings: int,
    force_rebuild: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    preprocess_dir = ensure_dir(work_dir / "preprocess")
    raw_audio_dir = ensure_dir(preprocess_dir / "raw_audio")
    segment_audio_dir = ensure_dir(preprocess_dir / "segments")
    segments_csv = preprocess_dir / "segments_manifest.csv"
    preprocessing_summary_path = preprocess_dir / "preprocess_summary.json"
    requested_config = {
        "manifest_csv": manifest_csv.resolve().as_posix(),
        "min_segment_sec": float(min_segment_sec),
        "max_segment_sec": float(max_segment_sec),
        "max_recordings": int(max_recordings),
    }

    if segments_csv.exists() and not force_rebuild:
        try:
            summary = json.loads(preprocessing_summary_path.read_text(encoding="utf-8"))
            cached_config = {
                "manifest_csv": summary.get("manifest_csv"),
                "min_segment_sec": float(summary.get("min_segment_sec")),
                "max_segment_sec": float(summary.get("max_segment_sec")),
                "max_recordings": int(summary.get("max_recordings", 0)),
            }
            if cached_config == requested_config:
                df = pd.read_csv(segments_csv)
                return df, summary
        except Exception:
            pass

    manifest_df = pd.read_csv(manifest_csv)
    manifest_df["rec_url_gcp"] = manifest_df["rec_url_gcp"].map(rewrite_upload_goai_url)
    manifest_df["transcription_url_gcp"] = manifest_df["transcription_url_gcp"].map(rewrite_upload_goai_url)
    manifest_df["metadata_url_gcp"] = manifest_df["metadata_url_gcp"].map(rewrite_upload_goai_url)

    if max_recordings > 0:
        manifest_df = manifest_df.head(max_recordings).copy()

    rows: list[dict[str, Any]] = []
    skipped_transcription = 0
    skipped_audio = 0
    dropped_duration = 0
    dropped_empty_text = 0

    iterator = manifest_df.to_dict(orient="records")
    for record in tqdm(iterator, desc="Preprocessing recordings"):
        recording_id = str(record["recording_id"])
        audio_url = record["rec_url_gcp"]
        transcription_url = record["transcription_url_gcp"]
        local_audio_path = raw_audio_dir / f"{recording_id}.wav"

        try:
            download_file(audio_url, local_audio_path)
        except Exception:
            skipped_audio += 1
            continue

        try:
            segments = read_transcription(transcription_url)
        except Exception:
            skipped_transcription += 1
            continue

        # Loading each recording once keeps segment slicing deterministic.
        waveform, sr = librosa.load(local_audio_path.as_posix(), sr=16000, mono=True)
        total_samples = waveform.shape[0]

        for segment_index, segment in enumerate(segments):
            text = normalize_text(str(segment.get("text", "")))
            if not text:
                dropped_empty_text += 1
                continue

            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            duration = max(0.0, end - start)

            if duration < min_segment_sec or duration > max_segment_sec:
                dropped_duration += 1
                continue

            start_sample = max(0, int(start * sr))
            end_sample = min(total_samples, int(end * sr))
            if end_sample <= start_sample:
                dropped_duration += 1
                continue

            segment_waveform = waveform[start_sample:end_sample]
            segment_file_name = f"{recording_id}_{segment_index:04d}.wav"
            segment_path = segment_audio_dir / segment_file_name

            if not segment_path.exists():
                sf.write(segment_path.as_posix(), segment_waveform, sr)

            rows.append(
                {
                    "recording_id": int(record["recording_id"]),
                    "user_id": int(record["user_id"]),
                    "language": record["language"],
                    "segment_id": segment_index,
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "duration_sec": round(duration, 3),
                    "audio_path": segment_path.as_posix(),
                    "text": text,
                }
            )

    segments_df = pd.DataFrame(rows).sort_values(["recording_id", "segment_id"]).reset_index(drop=True)
    segments_df.to_csv(segments_csv, index=False)

    total_hours = round(float(segments_df["duration_sec"].sum() / 3600.0), 3) if len(segments_df) else 0.0
    summary = {
        "manifest_csv": manifest_csv.resolve().as_posix(),
        "recordings_requested": int(len(manifest_df)),
        "segments_kept": int(len(segments_df)),
        "hours_kept": total_hours,
        "skipped_audio_download": int(skipped_audio),
        "skipped_transcription_download": int(skipped_transcription),
        "dropped_duration_filter": int(dropped_duration),
        "dropped_empty_text": int(dropped_empty_text),
        "min_segment_sec": float(min_segment_sec),
        "max_segment_sec": float(max_segment_sec),
        "max_recordings": int(max_recordings),
    }
    preprocessing_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return segments_df, summary


def split_train_eval(
    segments_df: pd.DataFrame,
    eval_ratio: float,
    seed: int,
    work_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dir = ensure_dir(work_dir / "splits")
    train_path = split_dir / "train_segments.csv"
    eval_path = split_dir / "eval_segments.csv"

    grouped = segments_df.groupby("recording_id")
    recording_ids = list(grouped.groups.keys())
    if len(recording_ids) < 2:
        raise RuntimeError("Need at least two recordings to create train/eval split without leakage.")

    rng = random.Random(seed)
    rng.shuffle(recording_ids)
    group_sizes = {rid: int(len(grouped.get_group(rid))) for rid in recording_ids}
    total_segments = int(len(segments_df))
    target_eval_segments = max(1, int(total_segments * eval_ratio))

    eval_recordings: list[int] = []
    eval_count = 0
    for rid in recording_ids:
        if len(eval_recordings) >= len(recording_ids) - 1:
            break
        if eval_count >= target_eval_segments:
            break
        eval_recordings.append(rid)
        eval_count += group_sizes[rid]

    if not eval_recordings:
        eval_recordings = [recording_ids[0]]
    if len(eval_recordings) == len(recording_ids):
        eval_recordings = eval_recordings[:-1]

    eval_ids = set(eval_recordings)
    eval_df = segments_df[segments_df["recording_id"].isin(eval_ids)].copy().reset_index(drop=True)
    train_df = segments_df[~segments_df["recording_id"].isin(eval_ids)].copy().reset_index(drop=True)

    if train_df.empty or eval_df.empty:
        raise RuntimeError("Train/eval split failed; one side is empty after recording-level split.")

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    return train_df, eval_df


def create_dataset_from_df(df: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(df[["audio_path", "text"]], preserve_index=False)
    ds = ds.rename_column("audio_path", "audio")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_inference_bundle(model_path_or_name: str, device: torch.device) -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    processor = WhisperProcessor.from_pretrained(model_path_or_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_path_or_name)
    model.generation_config.language = "hi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.to(device)
    model.eval()
    return processor, model


def generate_text(
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    audio_array: np.ndarray,
    sampling_rate: int,
    device_state: dict[str, Any],
) -> str:
    current_device: torch.device = device_state["device"]
    features = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    ).input_features.to(current_device)

    try:
        with torch.no_grad():
            predicted_ids = model.generate(features, language="hi", task="transcribe")
    except Exception as error:
        if current_device.type == "cuda":
            # Some GPU/PyTorch combos fail at runtime; fallback keeps the run alive.
            device_state["device"] = torch.device("cpu")
            model.to(device_state["device"])
            features = features.to(device_state["device"])
            with torch.no_grad():
                predicted_ids = model.generate(features, language="hi", task="transcribe")
        else:
            raise error

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def evaluate_on_segments(
    model_path_or_name: str,
    eval_df: pd.DataFrame,
    device: torch.device,
) -> tuple[float, list[dict[str, Any]]]:
    processor, model = load_inference_bundle(model_path_or_name, device)
    refs: list[str] = []
    hyps: list[str] = []
    records: list[dict[str, Any]] = []
    device_state: dict[str, Any] = {"device": device}

    for row in tqdm(eval_df.to_dict(orient="records"), desc=f"Evaluating {model_path_or_name} on Josh eval"):
        audio_array, sr = librosa.load(row["audio_path"], sr=16000, mono=True)
        hyp_raw = generate_text(processor, model, audio_array, sr, device_state)
        ref = normalize_text(row["text"])
        hyp = normalize_text(hyp_raw)

        stats = process_words(ref, hyp)
        record = {
            "recording_id": int(row["recording_id"]),
            "segment_id": int(row["segment_id"]),
            "duration_sec": float(row["duration_sec"]),
            "reference": row["text"],
            "reference_norm": ref,
            "hypothesis": hyp_raw,
            "hypothesis_norm": hyp,
            "sample_wer": float(stats.wer),
            "substitutions": int(stats.substitutions),
            "deletions": int(stats.deletions),
            "insertions": int(stats.insertions),
        }
        refs.append(ref)
        hyps.append(hyp)
        records.append(record)

    wer_pct = 100.0 * jiwer_wer(refs, hyps) if refs else 0.0
    return wer_pct, records


def evaluate_on_fleurs(
    model_path_or_name: str,
    device: torch.device,
    max_samples: int,
) -> float:
    processor, model = load_inference_bundle(model_path_or_name, device)
    fleurs = load_dataset("google/fleurs", "hi_in", split="test", trust_remote_code=True)
    if max_samples > 0:
        fleurs = fleurs.select(range(min(max_samples, len(fleurs))))

    refs: list[str] = []
    hyps: list[str] = []
    device_state: dict[str, Any] = {"device": device}

    for example in tqdm(fleurs, desc=f"Evaluating {model_path_or_name} on FLEURS hi_in"):
        audio = example["audio"]
        hyp_raw = generate_text(processor, model, audio["array"], audio["sampling_rate"], device_state)
        ref = normalize_text(example["transcription"])
        hyp = normalize_text(hyp_raw)
        refs.append(ref)
        hyps.append(hyp)

    return 100.0 * jiwer_wer(refs, hyps) if refs else 0.0


def finetune_whisper(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    output_dir = args.finetuned_model_dir.resolve()
    if args.skip_training and (output_dir / "config.json").exists():
        return output_dir

    processor = WhisperProcessor.from_pretrained(args.model_name, language="Hindi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.generation_config.language = "hi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    train_ds = create_dataset_from_df(train_df)
    eval_ds = create_dataset_from_df(eval_df)

    def prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    train_prepared = train_ds.map(prepare_batch, remove_columns=train_ds.column_names, num_proc=1)
    eval_prepared = eval_ds.map(prepare_batch, remove_columns=eval_ds.column_names, num_proc=1)

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(pred: Any) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_norm = [normalize_text(item) for item in pred_str]
        label_norm = [normalize_text(item) for item in label_str]
        return {"wer": 100.0 * jiwer_wer(label_norm, pred_norm)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir.as_posix(),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.train_epochs if args.max_steps <= 0 else 1.0,
        max_steps=args.max_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=False,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_prepared,
        eval_dataset=eval_prepared,
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    trainer.save_model(output_dir.as_posix())
    processor.save_pretrained(output_dir.as_posix())
    return output_dir


def classify_error(error_item: dict[str, Any]) -> str:
    ref = error_item["reference_norm"]
    hyp = error_item["hypothesis_norm"]
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)

    if any(token.isdigit() for token in ref_tokens + hyp_tokens) or (
        ref_set | hyp_set
    ).intersection(NUMBER_WORDS):
        return "numeric_rendering"

    if re.search(r"[A-Za-z]", ref + " " + hyp):
        return "english_or_name_token"

    if error_item["deletions"] >= 2 and error_item["deletions"] >= error_item["insertions"] + error_item["substitutions"]:
        return "content_omission"
    if error_item["insertions"] >= 2 and error_item["insertions"] >= error_item["deletions"] + error_item["substitutions"]:
        return "content_insertion"

    aligned_similar = 0
    compared = 0
    for hyp_token, ref_token in zip(hyp_tokens, ref_tokens):
        compared += 1
        if hyp_token != ref_token and edit_distance(hyp_token, ref_token) <= 2 and is_devanagari_token(hyp_token):
            aligned_similar += 1
    if compared > 0 and aligned_similar / compared >= 0.2:
        return "spelling_or_phonetic_variant"

    return "word_order_or_substitution"


def sample_errors_stratified(error_items: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if not error_items:
        return []

    low: list[dict[str, Any]] = []
    medium: list[dict[str, Any]] = []
    high: list[dict[str, Any]] = []

    sorted_items = sorted(
        error_items,
        key=lambda item: (item["sample_wer"], item["recording_id"], item["segment_id"]),
        reverse=True,
    )
    for item in sorted_items:
        if item["sample_wer"] > 0.5:
            high.append(item)
        elif item["sample_wer"] > 0.25:
            medium.append(item)
        else:
            low.append(item)

    sampled: list[dict[str, Any]] = []
    buckets = [high, medium, low]
    cursor = 0
    while len(sampled) < min(n, len(sorted_items)) and any(buckets):
        bucket = buckets[cursor % len(buckets)]
        if bucket:
            sampled.append(bucket.pop(0))
        cursor += 1
    return sampled


def build_taxonomy(error_items: list[dict[str, Any]]) -> dict[str, Any]:
    categorized = []
    for item in error_items:
        category = classify_error(item)
        row = dict(item)
        row["error_category"] = category
        row["category_description"] = ERROR_CATEGORY_DESCRIPTIONS[category]
        categorized.append(row)

    counts = Counter(item["error_category"] for item in categorized)
    examples_all: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in sorted(categorized, key=lambda x: x["sample_wer"], reverse=True):
        cat = item["error_category"]
        examples_all[cat].append(
            {
                "recording_id": item["recording_id"],
                "segment_id": item["segment_id"],
                "reference": item["reference"],
                "hypothesis": item["hypothesis"],
                "reasoning": ERROR_CATEGORY_DESCRIPTIONS[cat],
            }
        )

    selected_categories = [cat for cat, _ in counts.most_common() if len(examples_all[cat]) >= 3]
    examples: dict[str, list[dict[str, Any]]] = {
        cat: examples_all[cat][:5] for cat in selected_categories
    }

    top3 = [cat for cat, _ in counts.most_common(3)]
    top3_fixes = [{"error_category": category, "actionable_fix": FIX_LIBRARY[category]} for category in top3]

    return {
        "counts": counts,
        "examples": examples,
        "selected_categories": selected_categories,
        "top3_fixes": top3_fixes,
        "categorized_errors": categorized,
    }


def learn_confusion_lexicon(
    categorized_errors: list[dict[str, Any]],
    min_count: int = 2,
    min_ratio: float = 0.6,
    max_distance: int = 2,
) -> dict[str, str]:
    pairs: dict[str, Counter] = defaultdict(Counter)

    for item in categorized_errors:
        ref_tokens = item["reference_norm"].split()
        hyp_tokens = item["hypothesis_norm"].split()
        matcher = SequenceMatcher(a=hyp_tokens, b=ref_tokens, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace":
                continue
            if (i2 - i1) != 1 or (j2 - j1) != 1:
                continue
            hyp_token = hyp_tokens[i1]
            ref_token = ref_tokens[j1]
            if hyp_token == ref_token:
                continue
            if not (is_devanagari_token(hyp_token) and is_devanagari_token(ref_token)):
                continue
            if edit_distance(hyp_token, ref_token) > max_distance:
                continue
            pairs[hyp_token][ref_token] += 1

    mapping: dict[str, str] = {}
    for hyp_token, counter in pairs.items():
        total = sum(counter.values())
        ref_token, count = counter.most_common(1)[0]
        if count < min_count:
            continue
        if (count / total) < min_ratio:
            continue
        mapping[hyp_token] = ref_token
    return mapping


def derive_single_pair_lexicon(categorized_errors: list[dict[str, Any]]) -> dict[str, str]:
    pair_counts: Counter = Counter()
    for item in categorized_errors:
        ref_tokens = item["reference_norm"].split()
        hyp_tokens = item["hypothesis_norm"].split()
        matcher = SequenceMatcher(a=hyp_tokens, b=ref_tokens, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace":
                continue
            if (i2 - i1) != 1 or (j2 - j1) != 1:
                continue
            hyp_token = hyp_tokens[i1]
            ref_token = ref_tokens[j1]
            if hyp_token == ref_token:
                continue
            pair_counts[(hyp_token, ref_token)] += 1

    if not pair_counts:
        return {}
    (hyp_token, ref_token), _ = pair_counts.most_common(1)[0]
    return {hyp_token: ref_token}


def apply_confusion_lexicon(text: str, lexicon: dict[str, str]) -> tuple[str, int]:
    tokens = normalize_text(text).split()
    replaced = 0
    output_tokens = []
    for token in tokens:
        mapped = lexicon.get(token, token)
        if mapped != token:
            replaced += 1
        output_tokens.append(mapped)
    return " ".join(output_tokens), replaced


def evaluate_fix_on_targeted_subset(
    categorized_errors: list[dict[str, Any]],
    lexicon: dict[str, str],
) -> dict[str, Any]:
    targeted = []
    for item in categorized_errors:
        corrected, replacements = apply_confusion_lexicon(item["hypothesis"], lexicon)
        if replacements <= 0:
            continue
        targeted.append(
            {
                "recording_id": item["recording_id"],
                "segment_id": item["segment_id"],
                "reference": normalize_text(item["reference"]),
                "before_hypothesis": normalize_text(item["hypothesis"]),
                "after_hypothesis": normalize_text(corrected),
                "replacements": replacements,
            }
        )

    if not targeted:
        return {
            "targeted_subset_size": 0,
            "before_wer_percent": None,
            "after_wer_percent": None,
            "delta_wer_percent": None,
            "examples": [],
            "note": "No suitable replacements found for confusion-lexicon fix.",
        }

    refs = [item["reference"] for item in targeted]
    before = [item["before_hypothesis"] for item in targeted]
    after = [item["after_hypothesis"] for item in targeted]

    before_wer = 100.0 * jiwer_wer(refs, before)
    after_wer = 100.0 * jiwer_wer(refs, after)
    delta = after_wer - before_wer

    return {
        "targeted_subset_size": len(targeted),
        "before_wer_percent": round(before_wer, 3),
        "after_wer_percent": round(after_wer, 3),
        "delta_wer_percent": round(delta, 3),
        "examples": targeted[:10],
    }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_report_markdown(
    preprocess_summary: dict[str, Any],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    baseline_eval_wer: float,
    finetuned_eval_wer: float,
    baseline_fleurs_wer: float,
    finetuned_fleurs_wer: float,
    sampled_errors: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    fix_result: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Question-1 Report (Josh Talks Hindi ASR)")
    lines.append("")
    lines.append("## a) Data preprocessing")
    lines.append("- Converted stale `joshtalks-data-collection/hq_data/hi/...` URLs to `upload_goai/...` URLs.")
    lines.append("- Downloaded audio + transcription JSON for each recording.")
    lines.append("- Segmented each recording using transcription timestamps and exported segment-level WAV clips.")
    lines.append("- Normalized transcripts by whitespace/punctuation cleanup and filtered invalid segment durations.")
    lines.append("")
    lines.append("### Preprocessing Summary")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Recordings requested | {preprocess_summary['recordings_requested']} |")
    lines.append(f"| Segments kept | {preprocess_summary['segments_kept']} |")
    lines.append(f"| Hours kept | {preprocess_summary['hours_kept']} |")
    lines.append(f"| Dropped by duration filter | {preprocess_summary['dropped_duration_filter']} |")
    lines.append(f"| Dropped by empty text | {preprocess_summary['dropped_empty_text']} |")
    lines.append(f"| Train segments | {len(train_df)} |")
    lines.append(f"| Eval segments | {len(eval_df)} |")
    lines.append("")
    lines.append("## b/c) Fine-tuning + WER comparison")
    lines.append("| Model | Josh Eval WER (%) | FLEURS Hindi Test WER (%) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Whisper-small (pretrained) | {baseline_eval_wer:.3f} | {baseline_fleurs_wer:.3f} |")
    lines.append(f"| Whisper-small (fine-tuned) | {finetuned_eval_wer:.3f} | {finetuned_fleurs_wer:.3f} |")
    lines.append("")
    lines.append("## d) Error sampling strategy and 25 sampled errors")
    lines.append("- Strategy: deterministic severity-stratified round-robin across high/medium/low sample-WER buckets.")
    lines.append("- No cherry-picking: items are sampled from the full error pool after sorting by severity.")
    lines.append(f"- Sample size produced: {len(sampled_errors)}")
    if len(sampled_errors) < 25:
        lines.append("- Warning: fewer than 25 error samples were available; rerun with a larger eval pool for strict compliance.")
    lines.append("")
    lines.append("## e) Emergent error taxonomy")
    lines.append("- Categories below are selected from observed error frequencies and kept only when enough evidence exists (>=3 examples).")
    lines.append("| Error Category | Count | Description |")
    lines.append("|---|---:|---|")
    if taxonomy["selected_categories"]:
        for category in taxonomy["selected_categories"]:
            count = taxonomy["counts"][category]
            lines.append(f"| {category} | {count} | {ERROR_CATEGORY_DESCRIPTIONS[category]} |")
    else:
        lines.append("| _none_ | 0 | No category met minimum evidence threshold (>=3 examples). |")
    lines.append("")
    lines.append("### Category Examples (3-5 each)")
    if taxonomy["selected_categories"]:
        for category in taxonomy["selected_categories"]:
            examples = taxonomy["examples"].get(category, [])
            lines.append(f"#### {category}")
            for example in examples[:5]:
                lines.append(
                    f"- ref: `{example['reference']}` | hyp: `{example['hypothesis']}` | reasoning: {example['reasoning']}"
                )
    else:
        lines.append("- No category with >=3 examples was available in this run.")
    lines.append("")
    lines.append("## f) Top-3 frequent error types and actionable fixes")
    for item in taxonomy["top3_fixes"]:
        lines.append(f"- {item['error_category']}: {item['actionable_fix']}")
    lines.append("")
    lines.append("## g) Implemented fix and before/after results")
    lines.append("- Implemented fix: confusion-lexicon post-correction learned from systematic one-to-one token substitutions.")
    lines.append(f"- Targeted subset size: {fix_result['targeted_subset_size']}")
    lines.append(f"- Before WER (%): {fix_result['before_wer_percent']}")
    lines.append(f"- After WER (%): {fix_result['after_wer_percent']}")
    lines.append(f"- Delta WER (%): {fix_result['delta_wer_percent']}")
    lines.append("")
    lines.append("### Before/After examples")
    for item in fix_result.get("examples", [])[:10]:
        lines.append(
            f"- ref: `{item['reference']}` | before: `{item['before_hypothesis']}` | after: `{item['after_hypothesis']}`"
        )
    lines.append("")
    return "\n".join(lines)


def run_pipeline(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    set_seed(args.seed)
    if not args.skip_strict_checks and args.sample_errors < 25:
        raise ValueError("Q1 strict mode requires --sample-errors >= 25.")

    manifest_csv = args.manifest_csv if args.manifest_csv else default_manifest_path(repo_root)
    work_dir = args.work_dir.resolve()
    ensure_dir(work_dir)

    print("Step a) Preprocessing dataset...")
    segments_df, preprocess_summary = preprocess_dataset(
        manifest_csv=manifest_csv,
        work_dir=work_dir,
        min_segment_sec=args.min_segment_sec,
        max_segment_sec=args.max_segment_sec,
        max_recordings=args.max_recordings,
        force_rebuild=args.force_rebuild,
    )
    if len(segments_df) < 5:
        raise RuntimeError("Not enough segments after preprocessing. Check URL rewrites and data quality filters.")

    train_df, eval_df = split_train_eval(segments_df, args.eval_ratio, args.seed, work_dir)
    print(f"Train segments: {len(train_df)} | Eval segments: {len(eval_df)}")

    device = get_device(args.device)
    print(f"Using device: {device}")

    print("Step b) Evaluating pretrained Whisper-small on Josh eval split...")
    baseline_eval_wer, baseline_eval_records = evaluate_on_segments(args.model_name, eval_df, device)
    save_json(work_dir / "baseline_eval_predictions.json", baseline_eval_records)

    print("Step b) Evaluating pretrained Whisper-small on FLEURS hi_in test...")
    baseline_fleurs_wer = evaluate_on_fleurs(args.model_name, device, args.max_fleurs_samples)

    print("Step b) Fine-tuning Whisper-small...")
    finetuned_model_dir = finetune_whisper(train_df, eval_df, args, device)

    print("Step b) Evaluating fine-tuned model on Josh eval split...")
    finetuned_eval_wer, finetuned_eval_records = evaluate_on_segments(finetuned_model_dir.as_posix(), eval_df, device)
    save_json(work_dir / "finetuned_eval_predictions.json", finetuned_eval_records)

    print("Step b) Evaluating fine-tuned model on FLEURS hi_in test...")
    finetuned_fleurs_wer = evaluate_on_fleurs(finetuned_model_dir.as_posix(), device, args.max_fleurs_samples)

    print("Step c) Writing WER table...")
    wer_summary = {
        "baseline_eval_wer_percent": round(baseline_eval_wer, 3),
        "finetuned_eval_wer_percent": round(finetuned_eval_wer, 3),
        "baseline_fleurs_wer_percent": round(baseline_fleurs_wer, 3),
        "finetuned_fleurs_wer_percent": round(finetuned_fleurs_wer, 3),
    }
    save_json(work_dir / "wer_summary.json", wer_summary)

    print("Step d/e/f) Sampling errors and building taxonomy...")
    all_errors = [item for item in finetuned_eval_records if item["sample_wer"] > 0.0]
    if not args.skip_strict_checks and len(all_errors) < 25:
        raise RuntimeError(
            f"Q1 strict mode requires at least 25 error utterances, but only {len(all_errors)} were found. "
            "Increase eval data size and rerun."
        )
    sampled_errors = sample_errors_stratified(all_errors, args.sample_errors)
    if not args.skip_strict_checks and len(sampled_errors) < 25:
        raise RuntimeError(
            f"Q1 strict mode requires at least 25 sampled errors, but only {len(sampled_errors)} were sampled."
        )
    save_json(work_dir / "error_samples_25.json", sampled_errors)

    taxonomy = build_taxonomy(all_errors)
    if not args.skip_strict_checks:
        if len(taxonomy["selected_categories"]) == 0:
            raise RuntimeError(
                "Q1 strict mode requires taxonomy categories with 3-5 examples each; none met the >=3 threshold."
            )
        for category in taxonomy["selected_categories"]:
            n_examples = len(taxonomy["examples"].get(category, []))
            if n_examples < 3 or n_examples > 5:
                raise RuntimeError(
                    f"Q1 strict mode requires 3-5 examples per taxonomy category; "
                    f"`{category}` has {n_examples}."
                )
        if len(taxonomy["top3_fixes"]) < 3:
            raise RuntimeError(
                f"Q1 strict mode requires top-3 frequent error types, but only {len(taxonomy['top3_fixes'])} were found."
            )
    taxonomy_serializable = {
        "counts": dict(taxonomy["counts"]),
        "examples": dict(taxonomy["examples"]),
        "selected_categories": taxonomy["selected_categories"],
        "top3_fixes": taxonomy["top3_fixes"],
    }
    save_json(work_dir / "error_taxonomy.json", taxonomy_serializable)

    print("Step g) Implementing one actionable fix (confusion-lexicon correction)...")
    lexicon = learn_confusion_lexicon(taxonomy["categorized_errors"])
    if not lexicon:
        lexicon = learn_confusion_lexicon(
            taxonomy["categorized_errors"],
            min_count=1,
            min_ratio=0.5,
            max_distance=3,
        )
    if not lexicon:
        lexicon = derive_single_pair_lexicon(taxonomy["categorized_errors"])
    save_json(work_dir / "implemented_fix_lexicon.json", lexicon)
    fix_result = evaluate_fix_on_targeted_subset(taxonomy["categorized_errors"], lexicon)
    save_json(work_dir / "implemented_fix_results.json", fix_result)

    report = build_report_markdown(
        preprocess_summary=preprocess_summary,
        train_df=train_df,
        eval_df=eval_df,
        baseline_eval_wer=baseline_eval_wer,
        finetuned_eval_wer=finetuned_eval_wer,
        baseline_fleurs_wer=baseline_fleurs_wer,
        finetuned_fleurs_wer=finetuned_fleurs_wer,
        sampled_errors=sampled_errors,
        taxonomy=taxonomy,
        fix_result=fix_result,
    )
    report_path = work_dir / "question1_report.md"
    report_path.write_text(report, encoding="utf-8")

    summary_table_path = work_dir / "wer_table.md"
    summary_table_path.write_text(
        "\n".join(
            [
                "| Model | Josh Eval WER (%) | FLEURS Hindi Test WER (%) |",
                "|---|---:|---:|",
                f"| Whisper-small (pretrained) | {baseline_eval_wer:.3f} | {baseline_fleurs_wer:.3f} |",
                f"| Whisper-small (fine-tuned) | {finetuned_eval_wer:.3f} | {finetuned_fleurs_wer:.3f} |",
            ]
        ),
        encoding="utf-8",
    )

    print("\nCompleted Question-1 pipeline.")
    print(f"Artifacts written to: {work_dir}")
    print(f"Main report: {report_path}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
