from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from sentinel.detect.features import transaction_features
from sentinel.detect.iforest import ForestModel
from sentinel.detect.rules import evaluate_rules
from sentinel.detect.supervised import SupervisedModel
from sentinel.store.models import Flag, Transaction


@dataclass(frozen=True)
class Detection:
    flagged: bool
    score: float
    reasons: list[str]
    model_version: str


class CombinedScorer:
    def __init__(
        self, forest: ForestModel | None = None, supervised: SupervisedModel | None = None
    ) -> None:
        self.forest = forest
        self.supervised = supervised

    @classmethod
    def from_artifact(cls, artifact: str | Path | None) -> "CombinedScorer":
        if not artifact:
            return cls()
        path = Path(artifact)
        supervised = path.with_name("supervised.joblib")
        return cls(
            ForestModel.load(path),
            SupervisedModel.load(supervised) if supervised.exists() else None,
        )

    def detect(
        self,
        features: Mapping[str, float],
        mode: Literal["rules", "forest", "supervised", "combined"] = "combined",
        forest_anomaly: float | None = None,
        supervised_probability: float | None = None,
        explain_features: bool = True,
    ) -> Detection:
        hits = evaluate_rules(features) if mode in ("rules", "combined") else []
        reasons = [hit.reason for hit in hits]
        forest_flag = False
        forest_score = 0.0
        if self.forest is not None and mode in ("forest", "combined"):
            anomaly = (
                forest_anomaly
                if forest_anomaly is not None
                else float(self.forest.anomaly_scores([features])[0])
            )
            forest_flag = anomaly >= self.forest.threshold
            forest_score = min(1.0, max(0.0, 0.5 + 2 * (anomaly - self.forest.threshold)))
            if forest_flag:
                contributing = self.forest.top_features(features) if explain_features else []
                reasons.append("forest model: " + ", ".join(contributing or ["multiple features"]))
        supervised_flag = False
        supervised_score = 0.0
        if self.supervised is not None and mode in ("supervised", "combined"):
            supervised_score = (
                supervised_probability
                if supervised_probability is not None
                else float(self.supervised.probabilities([features])[0])
            )
            supervised_flag = supervised_score >= self.supervised.threshold
            if supervised_flag:
                contributing = self.supervised.top_features(features) if explain_features else []
                reasons.append(
                    "supervised model: " + ", ".join(contributing or ["transaction profile"])
                )
        flagged = bool(hits) or forest_flag or supervised_flag
        if mode == "rules":
            score = min(1.0, 0.7 + 0.1 * (len(hits) - 1)) if hits else 0.0
        else:
            score = max(
                supervised_score,
                forest_score,
                min(1.0, 0.7 + 0.1 * (len(hits) - 1)) if hits else 0.0,
            )
        version = "+".join(
            value
            for value in (
                "rules-1" if hits else "",
                self.forest.version if self.forest and mode in ("forest", "combined") else "",
                self.supervised.version
                if self.supervised and mode in ("supervised", "combined")
                else "",
            )
            if value
        )
        return Detection(flagged, score, reasons, version)

    def score(self, transaction: Transaction, history: list[Transaction]) -> Flag | None:
        result = self.detect(transaction_features(transaction, history))
        if not result.flagged:
            return None
        return Flag(
            transaction_id=transaction.id,
            score=result.score,
            reasons=result.reasons,
            model_version=result.model_version,
            created_at=datetime.now(UTC),
        )
