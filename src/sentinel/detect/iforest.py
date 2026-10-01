import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from sentinel.detect.features import FEATURE_NAMES


def matrix(rows: list[Mapping[str, float]]) -> np.ndarray:
    return np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)


@dataclass
class ForestModel:
    estimator: IsolationForest
    threshold: float
    version: str
    medians: list[float]

    def anomaly_scores(self, rows: list[Mapping[str, float]]) -> np.ndarray:
        return -self.estimator.decision_function(matrix(rows))

    def top_features(self, row: Mapping[str, float], count: int = 3) -> list[str]:
        original = matrix([row])
        baseline = float(-self.estimator.decision_function(original)[0])
        contributions = []
        for index, name in enumerate(FEATURE_NAMES):
            changed = original.copy()
            changed[0, index] = self.medians[index]
            delta = baseline + float(self.estimator.decision_function(changed)[0])
            contributions.append((delta, name))
        ranked = sorted(contributions, reverse=True)[:count]
        return [f"{name}={row[name]:.2f}" for delta, name in ranked if delta > 0]

    @classmethod
    def load(cls, path: str | Path) -> "ForestModel":
        artifact = Path(path)
        metadata: dict[str, Any] = json.loads(artifact.with_suffix(".json").read_text())
        if metadata["feature_list"] != list(FEATURE_NAMES):
            raise ValueError("model feature list does not match")
        return cls(
            estimator=joblib.load(artifact),
            threshold=float(metadata["threshold"]),
            version=str(metadata["version"]),
            medians=[float(value) for value in metadata["feature_medians"]],
        )
