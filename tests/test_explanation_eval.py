from typing import Any

from evals.run_explanations import aggregate, sample_flags
from sentinel.store.models import Flag


def result(
    *,
    grounded: bool,
    first: bool,
    action: str = "review",
    issues: list[dict[str, Any]] | None = None,
    failure: str | None = None,
    calls: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "explanation": None if failure else {"recommended_action": action},
        "grounding": None
        if failure
        else {
            "ungrounded_items": issues or [],
            "verdicts": [
                {"claim_index": 0, "grounded": True},
                {"claim_index": 1, "grounded": grounded},
            ],
        },
        "grounded": grounded,
        "first_pass_grounded": first,
        "failure": failure,
        "latency_ms": 100,
        "tool_trace": [{"name": name, "error": None} for name in (calls or [])],
    }


def test_metric_aggregation() -> None:
    rows = [
        {
            "group": "amount_spike",
            "result": result(
                grounded=True, first=True, calls=["get_flag_detail", "get_account_stats"]
            ),
        },
        {
            "group": "travel",
            "result": result(
                grounded=False,
                first=False,
                action="approve",
                issues=[{"reason": "ref_mismatch"}],
                calls=["get_flag_detail"],
            ),
        },
        {"group": "none", "result": result(grounded=False, first=False, failure="timeout")},
        {"group": "injection_merchant", "result": result(grounded=True, first=False)},
        {"group": "sparse_zero", "result": result(grounded=True, first=True)},
    ]
    metrics = aggregate(rows)
    assert metrics["count"] == 5
    assert metrics["valid_report_rate"] == 0.8
    assert metrics["grounded_first_pass_rate"] == 0.4
    assert metrics["grounded_after_revision_rate"] == 0.6
    assert metrics["claim_level_grounded_rate"] == 0.75
    assert metrics["mean_ungrounded_by_reason"]["ref_mismatch"] == 0.25
    assert metrics["mean_tool_calls"] == 0.6
    assert metrics["called_get_flag_detail_rate"] == 0.4
    assert metrics["decision_agreement_rate"] == 0.6667
    assert metrics["legitimate_approve_rate"] == 0.5
    assert metrics["injection_pass_rate"] == 1.0
    assert metrics["sparse_history_pass_rate"] == 1.0
    assert metrics["failures_by_type"] == {"timeout": 1}


def test_seeded_stratified_sampling() -> None:
    flags = [Flag(id=index, transaction_id=f"tx{index}") for index in range(1, 11)]
    labels = {
        f"tx{index}": {
            "is_fraud": index <= 5,
            "pattern": "amount_spike" if index <= 5 else None,
            "scenario": "travel" if index <= 8 else None,
        }
        for index in range(1, 11)
    }
    first = sample_flags(flags, labels, 2, 7)
    assert first == sample_flags(flags, labels, 2, 7)
    assert len(first) == 6
    assert {group for _, group in first} == {"amount_spike", "travel", "none"}
    assert len({flag_id for flag_id, _ in first}) == 6
