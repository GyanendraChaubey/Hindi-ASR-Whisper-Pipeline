# Run Checklist: Question 4

Use this on your server to run lattice-based evaluation for Question-4.

## 1) Setup
```bash
git clone https://github.com/GyanendraChaubey/Hindi-ASR-Whisper-Pipeline.git
cd Hindi-ASR-Whisper-Pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Prepare input CSV
Create `data/question4_transcripts.csv` with columns like:
- `utterance_id`
- `reference` (human reference; may contain errors)
- `model_1`, `model_2`, `model_3`, `model_4`, `model_5` (ASR outputs)

Question-4 in the assignment explicitly uses **five ASR models**, and this pipeline now enforces that count by default.

If your model columns have different names, pass them using `--model-cols`.

## 3) Run
```bash
python scripts/run_question4.py \
  --input-file "data/question4_transcripts.csv" \
  --work-dir artifacts/q4 \
  --reference-col "reference" \
  --id-col "utterance_id" \
  --expected-model-count 5
```

Example with explicit model columns:
```bash
python scripts/run_question4.py \
  --input-file "data/question4_transcripts.csv" \
  --work-dir artifacts/q4 \
  --reference-col "reference" \
  --id-col "utterance_id" \
  --model-cols "asr_a,asr_b,asr_c,asr_d,asr_e" \
  --expected-model-count 5
```

## 4) Verify outputs (must exist)
```bash
ls -lah artifacts/q4/lattice_bins_per_utterance.jsonl \
       artifacts/q4/per_utterance_model_scores.csv \
       artifacts/q4/model_summary.csv \
       artifacts/q4/lattice_theory_pseudocode.md \
       artifacts/q4/question4_report.md \
       artifacts/q4/summary.json
```

## 5) Map to Question-4 requirements
- Lattice construction approach:
  - `artifacts/q4/lattice_theory_pseudocode.md`
  - `artifacts/q4/lattice_bins_per_utterance.jsonl`
- Handling insertions/deletions/substitutions:
  - implemented in DP scoring and bin trust rules
  - `artifacts/q4/lattice_theory_pseudocode.md`
- Trust model agreement over reference:
  - recorded as trust events in `lattice_bins_per_utterance.jsonl`
- WER per model (rigid vs lattice):
  - `artifacts/q4/model_summary.csv`
  - `artifacts/q4/per_utterance_model_scores.csv`
  - includes improved / unchanged / worsened tracking
- Final narrative:
  - `artifacts/q4/question4_report.md`

Notes on alternatives:
- Reference-position alternatives include all model-proposed tokens by default (`--alternative-support-threshold 1`).
- Insertion alternatives also include all model-proposed tokens by default (`--insertion-support-threshold 1`).

## 6) Final completion check
Question-4 is complete when:
- Run finishes without crash.
- All files in step 4 exist.
- `summary.json` reports `"models": 5`.
- `model_summary.csv` and report show lattice WER improvements for unfairly penalized cases and unchanged (or near-unchanged) outcomes where reference already aligns.
- `summary.json` contains `total_worsened_pairs`; this should ideally be `0` or very small.
