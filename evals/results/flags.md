| Detector | Precision | Recall | F1 | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| rules | 0.778 | 0.613 | 0.686 | 0.498 | 24 | 53 |
| forest | 0.727 | 0.584 | 0.648 | 0.689 | 30 | 57 |
| supervised | 0.977 | 0.934 | 0.955 | 0.978 | 3 | 9 |
| combined | 0.731 | 0.934 | 0.821 | 0.923 | 47 | 9 |

## Recall by fraud pattern

| Pattern | Detector | Transaction recall | Episode recall | Median transactions until first flag |
|---|---|---:|---:|---:|
| velocity_burst | rules | 0.617 | 1.000 | 4.0 |
| velocity_burst | forest | 0.638 | 1.000 | 4.0 |
| velocity_burst | supervised | 0.851 | 1.000 | 2.0 |
| velocity_burst | combined | 0.851 | 1.000 | 2.0 |
| account_takeover | rules | 0.542 | 0.833 | 1.0 |
| account_takeover | forest | 0.375 | 0.833 | 2.0 |
| account_takeover | supervised | 1.000 | 1.000 | 1.0 |
| account_takeover | combined | 1.000 | 1.000 | 1.0 |
| amount_spike | rules | 0.611 | 1.000 | 1.0 |
| amount_spike | forest | 0.333 | 0.667 | 2.5 |
| amount_spike | supervised | 0.944 | 1.000 | 1.0 |
| amount_spike | combined | 0.944 | 1.000 | 1.0 |
| structuring | rules | 0.958 | 1.000 | 1.0 |
| structuring | forest | 1.000 | 1.000 | 1.0 |
| structuring | supervised | 1.000 | 1.000 | 1.0 |
| structuring | combined | 1.000 | 1.000 | 1.0 |
| dormant_drain | rules | 0.333 | 0.833 | 1.0 |
| dormant_drain | forest | 0.458 | 1.000 | 1.0 |
| dormant_drain | supervised | 0.958 | 1.000 | 1.0 |
| dormant_drain | combined | 0.958 | 1.000 | 1.0 |

## False-positive rate by legitimate scenario

| Scenario | Rules | Forest | Supervised | Combined |
|---|---:|---:|---:|---:|
| travel | 0.000 | 0.000 | 0.000 | 0.000 |
| new_device | 0.000 | 0.000 | 0.040 | 0.040 |
| large_purchase | 1.000 | 0.000 | 0.000 | 1.000 |
| busy_day | 0.000 | 0.000 | 0.000 | 0.000 |
| none | 0.003 | 0.005 | 0.000 | 0.007 |
