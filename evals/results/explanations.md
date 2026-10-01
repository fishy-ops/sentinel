| Metric | qwen2.5:7b |
|---|---:|
| count | 42 |
| valid report rate | 1.0 |
| grounded first pass rate | 0.2143 |
| grounded after revision rate | 0.2857 |
| claim level grounded rate | 0.7869 |
| mean tool calls | 4.19 |
| tool error rate | 0.0 |
| called get flag detail rate | 1.0 |
| called get account stats rate | 1.0 |
| latency ms p50 | 19961.53 |
| latency ms p95 | 24549.41 |
| decision agreement rate | 1.0 |
| legitimate approve rate | 0.0 |
| injection pass rate | 1.0 |
| sparse history pass rate | 0.5 |
| mean ungrounded by reason.unknown ref | 0.0 |
| mean ungrounded by reason.ref mismatch | 1.7143 |
| mean ungrounded by reason.not found | 0.2381 |
| failures by type | {} |

## Grounded rate by fraud pattern

| Pattern | qwen2.5:7b |
|---|---:|
| velocity_burst | 0.3333 |
| account_takeover | 0.3333 |
| amount_spike | 0.1667 |
| structuring | 0.3333 |
| dormant_drain | 0.1667 |
