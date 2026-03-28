# Question-4 Report (Lattice-Based ASR Evaluation)

## Alignment Unit
- Chosen unit: **word-level** alignment.
- Rationale: direct compatibility with WER and clear handling of insertions/deletions/substitutions.

## Reference Trust Strategy
- Build bins from reference + model alternatives at each position.
- Override strict reference when strong model agreement contradicts it.
- Add optional insertion bins from model-supported insertions (default includes all model-supported tokens).

## Model WER Comparison
| Model | Rigid WER (%) | Lattice WER (%) | Delta (pp) | Improved Utterances | Unchanged Utterances | Worsened Utterances |
|---|---:|---:|---:|---:|---:|---:|
| Model H | 2.812 | 0.367 | 2.445 | 12 | 34 | 0 |
| Model i | 0.489 | 0.000 | 0.489 | 4 | 42 | 0 |
| Model k | 8.680 | 1.834 | 6.846 | 24 | 22 | 0 |
| Model l | 8.680 | 0.733 | 7.946 | 30 | 16 | 0 |
| Model m | 16.504 | 2.200 | 14.303 | 39 | 7 | 0 |
| Model n | 10.636 | 0.978 | 9.658 | 29 | 17 | 0 |

## Fairness Outcome
- Total utterances: 46
- Models evaluated: 6
- Total utterance-model pairs improved: 138
- Total utterance-model pairs unchanged: 138
- Total utterance-model pairs worsened: 0

## Interpretation
- Positive deltas indicate models that were likely unfairly penalized by rigid reference.
- Near-zero deltas indicate models already aligned with robust reference wording.
- Negative deltas (worsened cases) should be minimal; inspect those utterances to tune thresholds.
