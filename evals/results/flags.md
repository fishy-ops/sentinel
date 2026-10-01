| Detector | Precision | Recall | F1 | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| rules | 0.736 | 0.628 | 0.678 | 0.498 | 171 | 282 |
| forest | 0.641 | 0.612 | 0.626 | 0.641 | 260 | 294 |
| supervised | 0.962 | 0.963 | 0.962 | 0.986 | 29 | 28 |
| combined | 0.694 | 0.966 | 0.808 | 0.964 | 323 | 26 |

## Recall by fraud pattern

| Pattern | Detector | Transaction recall | Episode recall | Median transactions until first flag |
|---|---|---:|---:|---:|
| velocity_burst | rules | 0.627 | 1.000 | 4.0 |
| velocity_burst | forest | 0.624 | 1.000 | 4.0 |
| velocity_burst | supervised | 0.920 | 1.000 | 2.0 |
| velocity_burst | combined | 0.920 | 1.000 | 2.0 |
| account_takeover | rules | 0.470 | 0.909 | 1.0 |
| account_takeover | forest | 0.333 | 0.818 | 2.0 |
| account_takeover | supervised | 0.992 | 1.000 | 1.0 |
| account_takeover | combined | 1.000 | 1.000 | 1.0 |
| amount_spike | rules | 0.596 | 0.939 | 1.0 |
| amount_spike | forest | 0.374 | 0.727 | 2.0 |
| amount_spike | supervised | 0.949 | 1.000 | 1.0 |
| amount_spike | combined | 0.960 | 1.000 | 1.0 |
| structuring | rules | 0.977 | 1.000 | 1.0 |
| structuring | forest | 1.000 | 1.000 | 1.0 |
| structuring | supervised | 1.000 | 1.000 | 1.0 |
| structuring | combined | 1.000 | 1.000 | 1.0 |
| dormant_drain | rules | 0.462 | 0.939 | 1.0 |
| dormant_drain | forest | 0.659 | 1.000 | 1.0 |
| dormant_drain | supervised | 0.992 | 1.000 | 1.0 |
| dormant_drain | combined | 0.992 | 1.000 | 1.0 |

## False-positive rate by legitimate scenario

| Scenario | Rules | Forest | Supervised | Combined |
|---|---:|---:|---:|---:|
| travel | 0.000 | 0.000 | 0.000 | 0.000 |
| new_device | 0.000 | 0.000 | 0.053 | 0.053 |
| large_purchase | 1.000 | 0.000 | 0.000 | 1.000 |
| busy_day | 0.000 | 0.000 | 0.000 | 0.000 |
| none | 0.005 | 0.008 | 0.001 | 0.010 |
