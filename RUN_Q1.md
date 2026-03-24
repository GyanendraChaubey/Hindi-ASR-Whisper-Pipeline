# Run Checklist: Question 1

Use this on your GPU server to run Question-1 end-to-end and verify outputs.

## 1) Setup
```bash
git clone https://github.com/GyanendraChaubey/Hindi-ASR-Whisper-Pipeline.git
cd Hindi-ASR-Whisper-Pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Quick debug run (recommended first)
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
`--skip-strict-checks` is only for quick debug. Keep strict checks enabled for final assignment run.

## 3) Full run
```bash
python scripts/run_question1.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q1 \
  --finetuned-model-dir models/whisper-small-hindi-q1 \
  --device cuda
```

If CUDA is not usable:
```bash
python scripts/run_question1.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q1 \
  --finetuned-model-dir models/whisper-small-hindi-q1 \
  --device cpu
```

## 4) Verify outputs (must exist)
```bash
ls -lah artifacts/q1/question1_report.md \
       artifacts/q1/wer_table.md \
       artifacts/q1/wer_summary.json \
       artifacts/q1/error_samples_25.json \
       artifacts/q1/error_taxonomy.json \
       artifacts/q1/implemented_fix_lexicon.json \
       artifacts/q1/implemented_fix_results.json \
       artifacts/q1/preprocess/preprocess_summary.json
```

## 5) Map to Question-1 parts
- a) preprocessing: `artifacts/q1/preprocess/preprocess_summary.json`
- b/c) baseline vs fine-tuned + WER table: `artifacts/q1/wer_table.md`, `artifacts/q1/wer_summary.json`
- d) 25 sampled errors: `artifacts/q1/error_samples_25.json`
- e) taxonomy + category examples: `artifacts/q1/error_taxonomy.json`, `artifacts/q1/question1_report.md`
- f) top-3 frequent error types + fixes: `artifacts/q1/question1_report.md`
- g) implemented fix + before/after: `artifacts/q1/implemented_fix_results.json`

## 6) Final completion check
Question-1 is considered fully complete when:
- Full run finishes without crash.
- All files in step 4 exist.
- `error_samples_25.json` contains at least 25 sampled errors.
- `error_taxonomy.json` categories each contain 3-5 examples.
- `question1_report.md` contains filled metrics and error analysis content.
