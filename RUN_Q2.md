# Run Checklist: Question 2

Use this on your GPU server to run Question-2 end-to-end and verify outputs.

## 1) Setup
```bash
git clone https://github.com/GyanendraChaubey/Hindi-ASR-Whisper-Pipeline.git
cd Hindi-ASR-Whisper-Pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Debug run (recommended first)
```bash
python scripts/run_question2.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q2_debug \
  --max-recordings 4 \
  --max-segments 120 \
  --device cuda
```

## 3) Full run
```bash
python scripts/run_question2.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q2 \
  --device cuda
```

If CUDA is unavailable:
```bash
python scripts/run_question2.py \
  --manifest-csv "data/FT Data - data.csv" \
  --work-dir artifacts/q2 \
  --device cpu
```

## 4) Verify outputs (must exist)
```bash
ls -lah artifacts/q2/raw_asr_pairs.csv \
       artifacts/q2/q2_processed_transcripts.csv \
       artifacts/q2/number_normalization_examples.json \
       artifacts/q2/english_detection_examples.json \
       artifacts/q2/summary.json \
       artifacts/q2/question2_report.md
```

## 5) Map to Question-2 requirements
- Raw ASR + reference pairing: `artifacts/q2/raw_asr_pairs.csv`
- Number normalization output: `artifacts/q2/q2_processed_transcripts.csv` (`number_normalized_text`)
- Tagged English words output: `artifacts/q2/q2_processed_transcripts.csv` (`english_tagged_text`)
- 4-5 correct conversions + 2-3 edge cases: `artifacts/q2/number_normalization_examples.json`
- Tagged transcript examples: `artifacts/q2/english_detection_examples.json`
- Final narrative write-up: `artifacts/q2/question2_report.md`

## 6) Final completion check
Question-2 is complete when:
- Full run finishes without crash.
- All files in step 4 exist.
- `question2_report.md` includes normalization impact table, conversion examples, edge cases, and English tagging examples.

