import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from sentinel.detect.features import FEATURE_NAMES
from sentinel.detect.iforest import matrix

FEATURE_WORDS = {name: name.replace("_", " ") for name in FEATURE_NAMES}


@dataclass
class SupervisedModel:
    estimator: HistGradientBoostingClassifier
    threshold: float
    version: str
    medians: list[float]

    def probabilities(self, rows: list[Mapping[str, float]]) -> np.ndarray:
        return self.estimator.predict_proba(matrix(rows))[:, 1]

    def top_features(self, row: Mapping[str, float], count: int = 3) -> list[str]:
        original = matrix([row])
        baseline = float(self.estimator.predict_proba(original)[0, 1])
        contributions = []
        for index, name in enumerate(FEATURE_NAMES):
            changed = original.copy()
            changed[0, index] = self.medians[index]
            delta = baseline - float(self.estimator.predict_proba(changed)[0, 1])
            contributions.append((delta, name))
        return [
            f"{FEATURE_WORDS[name]} {row[name]:.2f}"
            for delta, name in sorted(contributions, reverse=True)[:count]
            if delta > 0
        ]

    @classmethod
    def load(cls, path: str | Path) -> "SupervisedModel":
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
