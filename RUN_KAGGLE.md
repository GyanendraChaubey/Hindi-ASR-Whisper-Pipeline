# Run on Kaggle (Q1-Q4)

This guide explains how to run this repo on Kaggle and what notebook structure is best.

## Recommended structure
Use **separate notebooks** for each question:
- Notebook 1: Q1 (`run_question1.py`)
- Notebook 2: Q2 (`run_question2.py`)
- Notebook 3: Q3 (`run_question3.py`)
- Notebook 4: Q4 (`run_question4.py`)

Pre-created Kaggle notebook files in this repo:
- `scripts/kaggle_q1.ipynb`
- `scripts/kaggle_q2.ipynb`
- `scripts/kaggle_q3.ipynb`
- `scripts/kaggle_q4.ipynb`

Why separate notebooks:
- Q1 is long-running and can consume most of a Kaggle session.
- Q2/Q3/Q4 are independent and easier to rerun separately.
- Q3 requires manual review and a second run with reviewed labels.

You *can* run everything in one notebook, but it is less reliable for Kaggle time limits.

## 1) Kaggle notebook settings
Before running:
- Accelerator: `GPU` (recommended for Q1/Q2; CPU also works but slower)
- Internet: `ON` (required to download audio/transcription URLs and FLEURS)

## 2) Bootstrap cell (all notebooks)
```bash
!git clone https://github.com/GyanendraChaubey/Hindi-ASR-Whisper-Pipeline.git
%cd Hindi-ASR-Whisper-Pipeline
!pip install -r requirements.txt
```

## 3) Question-1 notebook

### Debug run (recommended first)
```bash
!python scripts/run_question1.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q1_debug \
  --finetuned-model-dir models/whisper-small-hindi-q1-debug \
  --max-recordings 4 \
  --max-fleurs-samples 30 \
  --train-epochs 1 \
  --skip-strict-checks \
  --device cuda
```

### Full run (strict)
```bash
!python scripts/run_question1.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q1 \
  --finetuned-model-dir models/whisper-small-hindi-q1 \
  --device cuda
```

## 4) Question-2 notebook
```bash
!python scripts/run_question2.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q2 \
  --device cuda
```

## 5) Question-3 notebook (two-pass flow)

### Pass-1: generate low-confidence review sample
Run this first:
```bash
!python scripts/run_question3.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q3 \
  --wordlist-file "data/unique_words.csv" \
  --review-sample-size 50
```

In strict mode, the run intentionally stops after generating:
- `artifacts/q3/low_confidence_review_sample.csv`

Now review that CSV manually and fill `manual_label` (`correct`/`incorrect`) for 40-50 rows.

### Pass-2: rerun with reviewed file
If reviewed file is in the same working directory:
```bash
!python scripts/run_question3.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q3 \
  --wordlist-file "data/unique_words.csv" \
  --manual-review-file "artifacts/q3/low_confidence_review_sample.csv"
```

If you uploaded reviewed file via Kaggle input dataset, use that path:
```bash
!python scripts/run_question3.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q3 \
  --wordlist-file "data/unique_words.csv" \
  --manual-review-file "/kaggle/input/<your-dataset>/low_confidence_review_sample.csv"
```

## 6) Question-4 notebook
Prepare `data/question4_transcripts.csv` (with `utterance_id`, `reference`, and five model columns).

If file is uploaded as Kaggle input:
```bash
!cp /kaggle/input/<your-dataset>/question4_transcripts.csv data/question4_transcripts.csv
```

Run:
```bash
!python scripts/run_question4.py \
  --input-file "data/question4_transcripts.csv" \
  --work-dir artifacts/q4 \
  --reference-col "reference" \
  --id-col "utterance_id" \
  --expected-model-count 5
```

## 7) Verify outputs quickly
```bash
!ls -lah artifacts/q1 || true
!ls -lah artifacts/q2 || true
!ls -lah artifacts/q3 || true
!ls -lah artifacts/q4 || true
```

## 8) Persist and download outputs from Kaggle
- Use `Save Version` so outputs are stored with notebook version.
- Optional archive:
```bash
!tar -czf asr_artifacts.tar.gz artifacts
```
Then download `asr_artifacts.tar.gz` from Kaggle output files.
