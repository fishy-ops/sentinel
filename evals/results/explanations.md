| Metric | sentinel-base:1.5b | sentinel-analyst:1.5b | sentinel-base:7b |
|---|---:|---:|---:|
| count | 42 | 42 | 42 |
| valid report rate | 0.0 | 1.0 | 1.0 |
| grounded first pass rate | 0.0 | 0.5 | 0.2143 |
| grounded after revision rate | 0.0 | 0.5476 | 0.2857 |
| claim level grounded rate | 0.0 | 0.7179 | 0.7869 |
| mean tool calls | 1.0 | 3.07 | 4.19 |
| tool error rate | 1.0 | 0.0078 | 0.0 |
| called get flag detail rate | 1.0 | 1.0 | 1.0 |
| called get account stats rate | 0.0 | 1.0 | 1.0 |
| latency ms p50 | 421.14 | 3876.45 | 20079.93 |
| latency ms p95 | 425.83 | 5549.67 | 24442.24 |
| decision agreement rate | 0.0 | 1.0 | 1.0 |
| legitimate approve rate | 0.0 | 0.0 | 0.0 |
| injection pass rate | 0.0 | 1.0 | 1.0 |
| sparse history pass rate | 0.0 | 1.0 | 0.5 |
| mean ungrounded by reason.unknown ref | 0.0 | 0.0 | 0.0 |
| mean ungrounded by reason.ref mismatch | 0.0 | 2.5476 | 1.7143 |
| mean ungrounded by reason.not found | 0.0 | 0.0952 | 0.2381 |
| failures by type | {'no_tool_called': 42} | {} | {} |

## Grounded rate by fraud pattern

| Pattern | sentinel-base:1.5b | sentinel-analyst:1.5b | sentinel-base:7b |
|---|---:|---:|---:|
| velocity_burst | 0.0 | 0.8333 | 0.3333 |
| account_takeover | 0.0 | 0.6667 | 0.3333 |
| amount_spike | 0.0 | 0.8333 | 0.1667 |
| structuring | 0.0 | 0.0 | 0.3333 |
| dormant_drain | 0.0 | 0.5 | 0.1667 |
