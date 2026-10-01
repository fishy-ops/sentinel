import json
from pathlib import Path
from typing import Any

from sentinel.detect.features import features_for_rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def load_split(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, float]], list[dict[str, Any]]]:
    rows = read_jsonl(root / "transactions.jsonl")
    features = features_for_rows(rows)
    labels_by_id = {row["transaction_id"]: row for row in read_jsonl(root / "labels.jsonl")}
    labels = [labels_by_id[row["transaction_id"]] for row in rows]
    return rows, features, labels
