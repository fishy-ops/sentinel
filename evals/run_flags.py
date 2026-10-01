import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from sentinel.detect.data import load_split
from sentinel.detect.score import CombinedScorer, Detection

PATTERNS = ("velocity_burst", "account_takeover", "amount_spike", "structuring", "dormant_drain")
SCENARIOS = ("travel", "new_device", "large_purchase", "busy_day", "none")
MODES = ("rules", "forest", "combined")


def evaluate(
    data_root: Path, artifact: Path, results: Path, failures: Path, curve: Path
) -> dict[str, Any]:
    rows, features, labels = load_split(data_root / "eval")
    scorer = CombinedScorer.from_artifact(artifact)
    truth = [bool(label["is_fraud"]) for label in labels]
    outputs: dict[str, dict[str, Any]] = {}
    failed: list[dict[str, Any]] = []
    plt.figure(figsize=(6, 4))
    for mode in MODES:
        detections: list[Detection] = [scorer.detect(item, mode) for item in features]
        predicted = [item.flagged for item in detections]
        scores = [item.score for item in detections]
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, average="binary", zero_division=0
        )
        pattern_recall = {
            pattern: sum(
                predicted[index]
                for index, label in enumerate(labels)
                if label["pattern"] == pattern
            )
            / max(1, sum(label["pattern"] == pattern for label in labels))
            for pattern in PATTERNS
        }
        scenario_fpr = {
            scenario: sum(
                predicted[index]
                for index, label in enumerate(labels)
                if not label["is_fraud"] and (label["scenario"] or "none") == scenario
            )
            / max(
                1,
                sum(
                    not label["is_fraud"] and (label["scenario"] or "none") == scenario
                    for label in labels
                ),
            )
            for scenario in SCENARIOS
        }
        pr_auc = float(average_precision_score(truth, scores))
        outputs[mode] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": pr_auc,
            "recall_per_pattern": pattern_recall,
            "false_positive_rate_per_scenario": scenario_fpr,
            "counts": {
                "true_positive": sum(t and p for t, p in zip(truth, predicted, strict=True)),
                "false_positive": sum(not t and p for t, p in zip(truth, predicted, strict=True)),
                "false_negative": sum(t and not p for t, p in zip(truth, predicted, strict=True)),
            },
        }
        curve_precision, curve_recall, _ = precision_recall_curve(truth, scores)
        plt.plot(curve_recall, curve_precision, label=f"{mode} (AP={pr_auc:.3f})")
        for index, (actual, item) in enumerate(zip(truth, detections, strict=True)):
            if actual != item.flagged:
                failed.append(
                    {
                        "detector": mode,
                        "failure": "false_negative" if actual else "false_positive",
                        "transaction_id": rows[index]["transaction_id"],
                        "pattern": labels[index]["pattern"],
                        "scenario": labels[index]["scenario"],
                        "score": item.score,
                        "features": features[index],
                        "reasons": item.reasons,
                    }
                )
    results.mkdir(parents=True, exist_ok=True)
    failures.parent.mkdir(parents=True, exist_ok=True)
    curve.parent.mkdir(parents=True, exist_ok=True)
    (results / "flags.json").write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n")
    headers = (
        "| Detector | Precision | Recall | F1 | PR-AUC | FP | FN |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [headers]
    for mode in MODES:
        item = outputs[mode]
        lines.append(
            f"| {mode} | {item['precision']:.3f} | {item['recall']:.3f} | "
            f"{item['f1']:.3f} | {item['pr_auc']:.3f} | {item['counts']['false_positive']} | "
            f"{item['counts']['false_negative']} |\n"
        )
    lines.append(
        "\n## Recall by fraud pattern\n\n"
        "| Pattern | Rules | Forest | Combined |\n|---|---:|---:|---:|\n"
    )
    for pattern in PATTERNS:
        lines.append(
            "| "
            + pattern
            + " | "
            + " | ".join(f"{outputs[mode]['recall_per_pattern'][pattern]:.3f}" for mode in MODES)
            + " |\n"
        )
    lines.append(
        "\n## False-positive rate by legitimate scenario\n\n"
        "| Scenario | Rules | Forest | Combined |\n|---|---:|---:|---:|\n"
    )
    for scenario in SCENARIOS:
        lines.append(
            "| "
            + scenario
            + " | "
            + " | ".join(
                f"{outputs[mode]['false_positive_rate_per_scenario'][scenario]:.3f}"
                for mode in MODES
            )
            + " |\n"
        )
    (results / "flags.md").write_text("".join(lines))
    with failures.open("w", encoding="utf-8") as target:
        for row in failed:
            target.write(json.dumps(row, sort_keys=True) + "\n")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Evaluation split precision–recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve, dpi=160)
    plt.close()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/generated"))
    parser.add_argument("--artifact", type=Path, default=Path("models/iforest.joblib"))
    args = parser.parse_args()
    outputs = evaluate(
        args.data,
        args.artifact,
        Path("evals/results"),
        Path("evals/failures/flags.jsonl"),
        Path("docs/results/pr_curve.png"),
    )
    for mode in MODES:
        print(
            f"{mode}: precision={outputs[mode]['precision']:.3f} "
            f"recall={outputs[mode]['recall']:.3f} f1={outputs[mode]['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
