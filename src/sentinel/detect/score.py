from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from sentinel.detect.features import transaction_features
from sentinel.detect.iforest import ForestModel
from sentinel.detect.rules import evaluate_rules
from sentinel.store.models import Flag, Transaction


@dataclass(frozen=True)
class Detection:
    flagged: bool
    score: float
    reasons: list[str]
    model_version: str


class CombinedScorer:
    def __init__(self, forest: ForestModel | None = None) -> None:
        self.forest = forest

    @classmethod
    def from_artifact(cls, artifact: str | Path | None) -> "CombinedScorer":
        return cls(ForestModel.load(artifact) if artifact else None)

    def detect(
        self,
        features: Mapping[str, float],
        mode: Literal["rules", "forest", "combined"] = "combined",
    ) -> Detection:
        hits = evaluate_rules(features) if mode != "forest" else []
        reasons = [hit.reason for hit in hits]
        forest_flag = False
        forest_score = 0.0
        if self.forest is not None and mode != "rules":
            anomaly = float(self.forest.anomaly_scores([features])[0])
            forest_flag = anomaly >= self.forest.threshold
            forest_score = min(1.0, max(0.0, 0.5 + 2 * (anomaly - self.forest.threshold)))
            if forest_flag and not hits:
                contributing = self.forest.top_features(features)
                reasons.append(
                    "unusual transaction profile: "
                    + ", ".join(contributing or ["multiple features"])
                )
        flagged = bool(hits) or forest_flag
        if mode == "rules":
            score = min(1.0, 0.7 + 0.1 * (len(hits) - 1)) if hits else 0.0
        else:
            score = max(forest_score, min(1.0, 0.7 + 0.1 * (len(hits) - 1)) if hits else 0.0)
        version = self.forest.version if self.forest and mode != "rules" else "rules-1"
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
