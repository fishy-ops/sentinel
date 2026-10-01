| Metric | sentinel-base:1.5b | sentinel-analyst:1.5b | sentinel-base:7b |
|---|---:|---:|---:|
| count | 78 | 78 | 78 |
| valid report rate | 0.0 | 1.0 | 1.0 |
| grounded first pass rate | 0.0 | 0.6795 | 0.4487 |
| grounded after revision rate | 0.0 | 0.7051 | 0.6282 |
| claim level grounded rate | 0.0 | 0.9136 | 0.8519 |
| mean tool calls | 1.0 | 3.44 | 4.23 |
| tool error rate | 1.0 | 0.0037 | 0.0 |
| called get flag detail rate | 1.0 | 1.0 | 1.0 |
| called get account stats rate | 0.0 | 1.0 | 1.0 |
| latency ms p50 | 662.47 | 4094.82 | 18227.5 |
| latency ms p95 | 1736.55 | 6982.24 | 27878.1 |
| decision agreement rate | 0.0 | 1.0 | 0.973 |
| legitimate approve rate | 0.0 | 0.0 | 0.0 |
| injection pass rate | 0.0 | 1.0 | 1.0 |
| sparse history pass rate | 0.0 | 0.0 | 1.0 |
| mean ungrounded by reason.unknown ref | 0.0 | 0.0 | 0.0 |
| mean ungrounded by reason.ref mismatch | 0.0 | 0.1667 | 0.641 |
| mean ungrounded by reason.not found | 0.0 | 0.2051 | 0.1538 |
| failures by type | {'no_tool_called': 78} | {} | {} |

## Grounded rate by fraud pattern

| Pattern | sentinel-base:1.5b | sentinel-analyst:1.5b | sentinel-base:7b |
|---|---:|---:|---:|
| velocity_burst | 0.0 | 0.5833 | 0.5833 |
| account_takeover | 0.0 | 0.6667 | 0.5 |
| amount_spike | 0.0 | 0.9167 | 0.6667 |
| structuring | 0.0 | 0.6667 | 0.9167 |
| dormant_drain | 0.0 | 0.8333 | 0.5833 |
