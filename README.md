# Sentinel

Sentinel is a transaction fraud triage service. This first milestone provides a reproducible synthetic transaction dataset with separate fraud labels for training and evaluation.

## Quickstart

```sh
uv sync
make data
uv run pytest -q
```

Generated train, val, eval, and finetune files are written to `data/generated/`.
