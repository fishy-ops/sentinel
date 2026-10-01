| Metric | sentinel-base:1.5b | sentinel-analyst:1.5b | sentinel-base:7b |
|---|---:|---:|---:|
| count | 42 | 42 | 42 |
| valid report rate | 0.0 | 1.0 | 1.0 |
| grounded first pass rate | 0.0 | 0.7619 | 0.4048 |
| grounded after revision rate | 0.0 | 0.8571 | 0.5714 |
| claim level grounded rate | 0.0 | 0.9406 | 0.8197 |
| mean tool calls | 1.0 | 3.57 | 4.26 |
| tool error rate | 1.0 | 0.0467 | 0.0 |
| called get flag detail rate | 1.0 | 1.0 | 1.0 |
| called get account stats rate | 0.0 | 1.0 | 1.0 |
| latency ms p50 | 417.67 | 3621.34 | 20214.22 |
| latency ms p95 | 427.55 | 6029.32 | 26909.97 |
| decision agreement rate | 0.0 | 0.9474 | 1.0 |
| legitimate approve rate | 0.0 | 0.0 | 0.0 |
| injection pass rate | 0.0 | 1.0 | 1.0 |
| sparse history pass rate | 0.0 | 1.0 | 1.0 |
| mean ungrounded by reason.unknown ref | 0.0 | 0.0 | 0.0 |
| mean ungrounded by reason.ref mismatch | 0.0 | 0.1429 | 0.881 |
| mean ungrounded by reason.not found | 0.0 | 0.0476 | 0.1905 |
| failures by type | {'no_tool_called': 42} | {} | {} |

## Grounded rate by fraud pattern

| Pattern | sentinel-base:1.5b | sentinel-analyst:1.5b | sentinel-base:7b |
|---|---:|---:|---:|
| velocity_burst | 0.0 | 1.0 | 0.6667 |
| account_takeover | 0.0 | 0.6667 | 0.5 |
| amount_spike | 0.0 | 1.0 | 0.3333 |
| structuring | 0.0 | 0.8333 | 0.8333 |
| dormant_drain | 0.0 | 0.8333 | 0.5 |
