# Sentinel

Sentinel is a transaction fraud triage service with reproducible synthetic data and an authenticated transaction API.

## Quickstart

```sh
uv sync
make data
uv run python -m sentinel.api.keys create --scopes ingest,read --name local
make run
uv run pytest -q
```

Generated train, val, eval, and finetune files are written to `data/generated/`.
Set `SENTINEL_DATABASE_URL` to choose a SQLite database; the default is `sqlite:///./sentinel.db`. Send the printed key in `X-API-Key` when calling `/v1` endpoints. Set `SENTINEL_RATE_PER_SECOND`, `SENTINEL_RATE_BURST`, and `SENTINEL_MAX_BODY_BYTES` to adjust request limits.
