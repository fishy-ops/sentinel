# Typed decision evaluation

All 1055 evaluation flags; 732 fraudulent.

| Method | Accuracy | AUROC | ECE (15 bins) | Brier | Median ms | p95 ms | Latency n |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority | 0.6938 | 0.5000 | 0.0082 | 0.2125 | — | — | — |
| supervised | 0.9810 | 0.9982 | 0.0112 | 0.0147 | — | — | — |
| zero_shot | 0.4882 | 0.4521 | 0.3391 | 0.3125 | 352.2194 | 377.2711 | 200 |
| trained_raw | 0.9649 | 0.9965 | 0.0538 | 0.0283 | 348.5992 | 371.5156 | 200 |
| trained_calibrated | 0.9649 | 0.9964 | 0.0451 | 0.0260 | 348.5992 | 371.5156 | 200 |

Gradient boosting beats the calibrated decision model on: accuracy, auroc, ece, brier.

Latency measures a complete decision (two forwards) on a seeded random sample, with warm-up excluded. Before/after temperature rows reuse the same logits and timing.

Trained batched throughput: 3.48 decisions/s at batch size 32 (fraud + pattern, including tokenisation and decoding). Peak MLX memory across trained batched passes: 4.633 GB (decimal).

## Pattern

Confusion matrices: rows are truth; columns are predictions.

A: velocity_burst; B: account_takeover; C: amount_spike; D: structuring; E: dormant_drain; F: legitimate.

### zero_shot

Accuracy: 0.3469

| Truth | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| A | 158 | 0 | 84 | 0 | 0 | 0 |
| B | 0 | 62 | 70 | 0 | 0 | 0 |
| C | 0 | 0 | 95 | 0 | 0 | 0 |
| D | 0 | 0 | 82 | 50 | 0 | 0 |
| E | 0 | 0 | 130 | 0 | 1 | 0 |
| F | 0 | 0 | 323 | 0 | 0 | 0 |

### trained_calibrated

Accuracy: 0.9621

| Truth | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| A | 241 | 0 | 1 | 0 | 0 | 0 |
| B | 0 | 131 | 0 | 0 | 0 | 1 |
| C | 3 | 0 | 88 | 3 | 0 | 1 |
| D | 0 | 0 | 0 | 129 | 3 | 0 |
| E | 0 | 0 | 0 | 1 | 130 | 0 |
| F | 14 | 5 | 5 | 0 | 3 | 296 |

## Gates fitted on validation

| Measure | Value |
|---|---:|
| low | 0.3425 |
| high | 0.6933 |
| share_auto_approved | 0.2521 |
| share_auto_blocked | 0.6938 |
| share_review | 0.0540 |
| fraud_leakage_among_auto_approved | 0.0075 |
| share_of_all_fraud_auto_approved | 0.0027 |
| precision_among_auto_blocked | 0.9850 |

The leakage constraint uses the share of all fraud auto-approved; fraud leakage among approved transactions uses approved count as its denominator. Validation constraints are empirical and do not guarantee evaluation performance.

## Robustness (trained, calibrated)

| Check | Measure | Value |
|---|---|---:|
| reversed_pattern | pattern_flip_share | 0.0028 |
| reversed_pattern | pattern_accuracy | 0.9602 |
| paraphrased_fraud | fraud_flip_share | 0.0114 |
| paraphrased_fraud | fraud_accuracy | 0.9668 |
| injection_merchant | fraud_flip_share | 0.0142 |
| injection_merchant | fraud_accuracy | 0.9716 |
| injection_merchant | pattern_flip_share | 0.0047 |
| injection_merchant | pattern_accuracy | 0.9592 |
| injection_merchant | decision_flip_share | 0.0190 |
| injection_memo | fraud_flip_share | 0.0227 |
| injection_memo | fraud_accuracy | 0.9517 |
| injection_memo | pattern_flip_share | 0.0085 |
| injection_memo | pattern_accuracy | 0.9583 |
| injection_memo | decision_flip_share | 0.0313 |
