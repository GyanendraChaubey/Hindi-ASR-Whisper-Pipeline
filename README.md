# Hindi-ASR-Whisper-Pipeline

End-to-end Hindi ASR optimization with Whisper-small, including:
- dataset preprocessing from Josh Talks manifest,
- Whisper-small baseline + fine-tuning,
- evaluation on Josh held-out split and Hindi FLEURS test split,
- structured WER table generation,
- systematic 25-sample error analysis and emergent taxonomy,
- top-3 actionable fixes, and
- one implemented fix with before/after targeted-subset results.

## Repository layout
- `data/FT Data - data.csv`: source manifest from assignment.
- `New_Task Assignment _ AI Researcher Intern- Speech & Audio _ Josh Talks .pdf`: assignment prompt.
- `scripts/notebook3635a60622.ipynb`: original experimental notebook.
- `src/hindi_asr_whisper_pipeline/question1.py`: reproducible Q1 pipeline implementation.
- `src/hindi_asr_whisper_pipeline/question2.py`: reproducible Q2 cleanup pipeline implementation.
- `src/hindi_asr_whisper_pipeline/question3.py`: reproducible Q3 spelling-quality pipeline implementation.
- `src/hindi_asr_whisper_pipeline/question4.py`: reproducible Q4 lattice-evaluation implementation.
- `scripts/run_question1.py`: CLI entrypoint.
- `scripts/run_question2.py`: CLI entrypoint.
- `scripts/run_question3.py`: CLI entrypoint.
- `scripts/run_question4.py`: CLI entrypoint.
- `RUN_KAGGLE.md`: step-by-step Kaggle execution guide.

## Running on Kaggle
- Recommended: use separate notebooks for Q1, Q2, Q3, and Q4.
- Reason: easier runtime management, easier reruns after failures, and Q3 needs a two-pass manual-review flow.
- Full guide: `RUN_KAGGLE.md`.

## How to run on your GPU server

### 1) Clone and enter repo
```bash
git clone https://github.com/GyanendraChaubey/Hindi-ASR-Whisper-Pipeline.git
cd Hindi-ASR-Whisper-Pipeline
```

### 2) Confirm required files are present
You should have:
- `data/FT Data - data.csv`
- `New_Task Assignment _ AI Researcher Intern- Speech & Audio _ Josh Talks .pdf`
- `scripts/notebook3635a60622.ipynb`

### 3) Create environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4) Debug run first (recommended)
This validates the pipeline before full 10-hour training.
```bash
python scripts/run_question1.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q1_debug \
  --finetuned-model-dir models/whisper-small-hindi-q1-debug \
  --max-recordings 4 \
  --max-fleurs-samples 30 \
  --train-epochs 1 \
  --skip-strict-checks \
  --device cuda
```
Use `--skip-strict-checks` only for quick debugging. For final Q1 submission, keep strict checks enabled.

### 5) Full Question-1 run
```bash
python scripts/run_question1.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q1 \
  --finetuned-model-dir models/whisper-small-hindi-q1 \
  --device cuda
```

### 6) If your GPU has compatibility issues
Use CPU fallback:
```bash
python scripts/run_question1.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q1 \
  --finetuned-model-dir models/whisper-small-hindi-q1 \
  --device cpu
```

## Main outputs
After running, check:
- `artifacts/q1/question1_report.md` (final assignment narrative for Q1 a-g)
- `artifacts/q1/wer_table.md` (structured WER table)
- `artifacts/q1/wer_summary.json`
- `artifacts/q1/error_samples_25.json`
- `artifacts/q1/error_taxonomy.json`
- `artifacts/q1/implemented_fix_lexicon.json`
- `artifacts/q1/implemented_fix_results.json`

## What each output answers in Q1
- **a)** preprocessing details: `artifacts/q1/preprocess/preprocess_summary.json`
- **b/c)** baseline vs fine-tuned metrics: `artifacts/q1/wer_table.md`, `artifacts/q1/wer_summary.json`
- **d)** 25 systematic error samples: `artifacts/q1/error_samples_25.json`
- **e)** emergent taxonomy + 3-5 examples/category: `artifacts/q1/error_taxonomy.json`, `artifacts/q1/question1_report.md`
- **f)** top-3 frequent error types + fixes: `artifacts/q1/question1_report.md`
- **g)** implemented fix with before/after: `artifacts/q1/implemented_fix_results.json`

Quick execution checklist: see `RUN_Q1.md`.

## Question-2 (ASR cleanup pipeline)
Run checklist: see `RUN_Q2.md`.

### Debug run
```bash
python scripts/run_question2.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q2_debug \
  --max-recordings 4 \
  --max-segments 120 \
  --device cuda
```

### Full run
```bash
python scripts/run_question2.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q2 \
  --device cuda
```

Main Q2 outputs:
- `artifacts/q2/raw_asr_pairs.csv`
- `artifacts/q2/q2_processed_transcripts.csv`
- `artifacts/q2/number_normalization_examples.json`
- `artifacts/q2/english_detection_examples.json`
- `artifacts/q2/summary.json`
- `artifacts/q2/question2_report.md`

## Question-3 (word-level spelling quality)
Run checklist: see `RUN_Q3.md`.

### Debug run
```bash
python scripts/run_question3.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q3_debug \
  --wordlist-file "data/unique_words.csv" \
  --max-recordings 4 \
  --max-fleurs-per-split 200 \
  --review-sample-size 40
```

### Full run
```bash
python scripts/run_question3.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q3 \
  --wordlist-file "data/unique_words.csv" \
  --review-sample-size 50
```

Strict Q3 flow requires manual review:
1) Run once to generate `artifacts/q3/low_confidence_review_sample.csv` (40-50 low-confidence words).  
2) Fill `manual_label` (`correct`/`incorrect`) in that file.  
3) Rerun with:
```bash
python scripts/run_question3.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q3 \
  --wordlist-file "data/unique_words.csv" \
  --manual-review-file "artifacts/q3/low_confidence_review_sample.csv"
```

Non-compliant debug fallback is available with `--allow-proxy-review`.

If `data/unique_words.csv` is not available yet, omit `--wordlist-file` and the pipeline will classify unique words derived from transcript corpus.

Main Q3 outputs:
- `artifacts/q3/word_classification_with_confidence.csv`
- `artifacts/q3/google_sheet_ready_word_labels.csv`
- `artifacts/q3/low_confidence_review_sample.csv`
- `artifacts/q3/low_confidence_review_analysis.json`
- `artifacts/q3/summary.json`
- `artifacts/q3/question3_report.md`

## Question-4 (lattice-based ASR evaluation)
Run checklist: see `RUN_Q4.md`.

### Input format
Prepare a CSV such as `data/question4_transcripts.csv` with:
- `utterance_id`
- `reference`
- five model output columns (example: `model_1` ... `model_5`)

The assignment text explicitly states five ASR models for Question-4, and the Q4 script enforces this by default.

### Run
```bash
python scripts/run_question4.py \
  --input-file "data/question4_transcripts.csv" \
  --work-dir artifacts/q4 \
  --reference-col "reference" \
  --id-col "utterance_id" \
  --expected-model-count 5
```

If model column names are custom, pass them explicitly:
```bash
python scripts/run_question4.py \
  --input-file "data/question4_transcripts.csv" \
  --work-dir artifacts/q4 \
  --reference-col "reference" \
  --id-col "utterance_id" \
  --model-cols "asr_a,asr_b,asr_c,asr_d,asr_e" \
  --expected-model-count 5
```

Main Q4 outputs:
- `artifacts/q4/lattice_bins_per_utterance.jsonl`
- `artifacts/q4/per_utterance_model_scores.csv`
- `artifacts/q4/model_summary.csv`
- `artifacts/q4/lattice_theory_pseudocode.md`
- `artifacts/q4/question4_report.md`
- `artifacts/q4/summary.json`

Q4 summaries now include improved / unchanged / worsened pair counts so you can verify fairness behavior directly.
Q4 defaults now include all model-proposed alternatives for reference and insertion bins (`--alternative-support-threshold 1`, `--insertion-support-threshold 1`).

## Notes
- The pipeline auto-rewrites stale `joshtalks-data-collection/hq_data/hi/...` URLs into working `upload_goai/...` URLs based on assignment instructions.
- If CUDA setup is incompatible, use `--device cpu` to avoid runtime GPU kernel mismatch errors.
