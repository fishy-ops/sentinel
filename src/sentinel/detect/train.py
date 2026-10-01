import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import precision_recall_curve

from sentinel.detect.data import load_split
from sentinel.detect.features import FEATURE_NAMES
from sentinel.detect.iforest import matrix


def train(data_root: Path, artifact: Path, seed: int = 42) -> dict[str, object]:
    _, train_features, train_labels = load_split(data_root / "train")
    _, val_features, val_labels = load_split(data_root / "val")
    train_matrix = matrix(train_features)
    model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=1)
    model.fit(train_matrix)
    val_scores = -model.decision_function(matrix(val_features))
    truth = np.asarray([int(row["is_fraud"]) for row in val_labels])
    precision, recall, thresholds = precision_recall_curve(truth, val_scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    threshold = float(thresholds[int(np.argmax(f1))])
    metadata: dict[str, object] = {
        "version": "iforest-1",
        "feature_list": list(FEATURE_NAMES),
        "threshold": threshold,
        "training_data_hash": hashlib.sha256(
            (data_root / "train" / "transactions.jsonl").read_bytes()
        ).hexdigest(),
        "sklearn_version": sklearn.__version__,
        "seed": seed,
        "feature_medians": np.median(train_matrix, axis=0).tolist(),
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact)
    artifact.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    supervised = HistGradientBoostingClassifier(random_state=seed)
    supervised.fit(train_matrix, [int(row["is_fraud"]) for row in train_labels])
    supervised_scores = supervised.predict_proba(matrix(val_features))[:, 1]
    precision, recall, thresholds = precision_recall_curve(truth, supervised_scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    supervised_artifact = artifact.with_name("supervised.joblib")
    supervised_metadata = metadata | {
        "version": "supervised-1",
        "threshold": float(thresholds[int(np.argmax(f1))]),
    }
    joblib.dump(supervised, supervised_artifact)
    supervised_artifact.with_suffix(".json").write_text(
        json.dumps(supervised_metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/generated"))
    parser.add_argument("--artifact", type=Path, default=Path("models/iforest.joblib"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = train(args.data, args.artifact, args.seed)
    print(f"trained {metadata['version']} with validation threshold {metadata['threshold']:.6f}")


if __name__ == "__main__":
    main()
