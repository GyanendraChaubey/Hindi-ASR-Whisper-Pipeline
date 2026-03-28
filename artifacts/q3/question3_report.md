# Question-3 Report (Word-Level Spelling Quality)

## a) Correct vs incorrect spelling identification approach
- Built lexical evidence from in-domain transcript word frequencies.
- Added external Hindi lexical evidence from FLEURS (when enabled).
- Applied rule-based diagnostics for script compliance and orthographic anomalies.
- Used nearest-lexicon similarity to flag likely misspellings.

### Classification totals
| Metric | Value |
|---|---:|
| Total unique words processed | 161327 |
| Predicted correct spelling | 122250 |
| Predicted incorrect spelling | 39077 |

### Confidence distribution
| Confidence | Count |
|---|---:|
| high | 7609 |
| medium | 5749 |
| low | 147969 |

## b) Confidence score + reason per word
- Output file: `word_classification_with_confidence.csv`
- Each row includes `word`, `predicted_label`, `confidence`, and `reason`.

## c) Low-confidence review (40-50 words)
- Review mode: manual
- Reviewed samples: 42
- Right predictions: 30
- Wrong predictions: 12
- Accuracy (%): 71.429

## d) Categories where system is unreliable
- rare_or_name_ambiguous: error_rate=70.588% over 17 reviewed words. Rare word with weak evidence; may be valid proper noun or may be misspelling.
- suspicious_form: error_rate=0.0% over 25 reviewed words. Word has suspicious orthographic pattern (matra/halant/repetition) without strong supporting evidence.

## Deliverables generated
- `google_sheet_ready_word_labels.csv` (2 columns: word, spelling_label)
- `word_classification_with_confidence.csv` (full scoring output)
- `low_confidence_review_sample.csv` (40-50 words for audit)
- `low_confidence_review_analysis.json`
- `summary.json`
