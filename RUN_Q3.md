# Run Checklist: Question 3

Use this on your GPU/compute server to run Question-3 end-to-end.

## 1) Setup
```bash
git clone https://github.com/GyanendraChaubey/Hindi-ASR-Whisper-Pipeline.git
cd Hindi-ASR-Whisper-Pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Input options
You can run in either mode:
- **Preferred (assignment mode):** provide the given unique-word list file (`--wordlist-file`).
- **Fallback mode:** omit `--wordlist-file` and the pipeline derives unique words from transcript corpus.

Supported `--wordlist-file` formats: `.csv`, `.txt`, `.json`

## 3) Debug run (recommended first)
```bash
python scripts/run_question3.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q3_debug \
  --wordlist-file "data/unique_words.csv" \
  --max-recordings 4 \
  --max-fleurs-per-split 200 \
  --review-sample-size 40
```

If you do not have the unique-word file yet, run without `--wordlist-file`.

## 4) Full run
```bash
python scripts/run_question3.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q3 \
  --wordlist-file "data/unique_words.csv" \
  --review-sample-size 50
```

## 5) Mandatory manual low-confidence review (strict assignment compliance)
In strict mode, the first run creates the sample and stops with instructions.

The pipeline creates:
- `artifacts/q3/low_confidence_review_sample.csv`

Fill `manual_label` (`correct`/`incorrect`) for 40-50 rows, then rerun:
```bash
python scripts/run_question3.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q3 \
  --wordlist-file "data/unique_words.csv" \
  --manual-review-file "artifacts/q3/low_confidence_review_sample.csv"
```

If you need a non-compliant debug fallback, add `--allow-proxy-review`.

## 6) Verify outputs (must exist)
```bash
ls -lah artifacts/q3/word_classification_with_confidence.csv \
       artifacts/q3/google_sheet_ready_word_labels.csv \
       artifacts/q3/low_confidence_review_sample.csv \
       artifacts/q3/low_confidence_review_analysis.json \
       artifacts/q3/summary.json \
       artifacts/q3/question3_report.md
```

## 7) Map to Question-3 requirements
- **a)** correct vs incorrect approach + final count:
  - `artifacts/q3/question3_report.md`
  - `artifacts/q3/summary.json`
- **b)** confidence score + reason for every word:
  - `artifacts/q3/word_classification_with_confidence.csv`
- **c)** review 40-50 low-confidence words and right/wrong:
  - `artifacts/q3/low_confidence_review_sample.csv`
  - `artifacts/q3/low_confidence_review_analysis.json`
- **d)** unreliable categories:
  - `artifacts/q3/question3_report.md`
  - `artifacts/q3/low_confidence_review_analysis.json`
- **Deliverable (Google Sheet 2 columns):**
  - `artifacts/q3/google_sheet_ready_word_labels.csv`

## 8) Final completion check
Question-3 is complete when:
- Full run finishes without crash.
- All files in step 6 exist.
- `low_confidence_review_analysis.json` reports `review_mode` as `manual`.
- `low_confidence_review_analysis.json` reports `reviewed_samples` between 40 and 50.
- `question3_report.md` contains reviewed low-confidence outcome and unreliable categories.
