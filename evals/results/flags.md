| Detector | Precision | Recall | F1 | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| rules | 0.657 | 0.336 | 0.444 | 0.264 | 24 | 91 |
| forest | 0.759 | 0.438 | 0.556 | 0.612 | 19 | 77 |
| combined | 0.679 | 0.555 | 0.610 | 0.558 | 36 | 61 |

## Recall by fraud pattern

| Pattern | Rules | Forest | Combined |
|---|---:|---:|---:|
| velocity_burst | 0.000 | 0.489 | 0.489 |
| account_takeover | 0.208 | 0.000 | 0.208 |
| amount_spike | 0.556 | 0.167 | 0.611 |
| structuring | 0.958 | 1.000 | 1.000 |
| dormant_drain | 0.333 | 0.417 | 0.542 |

## False-positive rate by legitimate scenario

| Scenario | Rules | Forest | Combined |
|---|---:|---:|---:|
| travel | 0.000 | 0.000 | 0.000 |
| new_device | 0.000 | 0.000 | 0.000 |
| large_purchase | 1.000 | 0.000 | 1.000 |
| busy_day | 0.000 | 0.000 | 0.000 |
| none | 0.003 | 0.003 | 0.005 |
