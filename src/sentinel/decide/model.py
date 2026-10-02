"""Label-only probability readouts and dependency-free calibration maths."""

import json
import math
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

from sentinel.decide.questions import FRAUD, PATTERN, QUESTIONS, Question

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_BATCH_SIZE = 512
Action = Literal["approve", "review", "block_and_contact"]


@dataclass(frozen=True)
class Decision:
    fraud_probability: float
    pattern_probabilities: dict[str, float]
    latency_ms: float | None  # None for batched decisions; only single reads are timed.
    model: str


class DecisionModel(Protocol):
    def decide(self, digest: str) -> Decision: ...


def _softmax(values: Sequence[float]) -> list[float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("logits must be nonempty and finite")
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def renormalise(logits: Sequence[float], allowed_ids: Sequence[int]) -> list[float]:
    if not allowed_ids or len(set(allowed_ids)) != len(allowed_ids):
        raise ValueError("allowed ids must be nonempty and distinct")
    if any(type(index) is not int or not 0 <= index < len(logits) for index in allowed_ids):
        raise ValueError("allowed token id outside vocabulary")
    return _softmax([logits[index] for index in allowed_ids])


def apply_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    return _softmax([value / temperature for value in logits])


def fit_temperature(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Minimise multiclass NLL in log-temperature on [0.01, 100]."""
    if not logits or len(logits) != len(labels):
        raise ValueError("logits and labels must have the same nonzero length")
    width = len(logits[0])
    for row, label in zip(logits, labels, strict=True):
        _softmax(row)
        if len(row) != width or type(label) is not int or not 0 <= label < width:
            raise ValueError("invalid label or inconsistent logit width")

    def objective(log_t: float) -> float:
        temperature = math.exp(log_t)
        loss = 0.0
        for row, label in zip(logits, labels, strict=True):
            scaled = [value / temperature for value in row]
            peak = max(scaled)
            loss += peak + math.log(sum(math.exp(value - peak) for value in scaled)) - scaled[label]
        return loss / len(labels)

    left, right = math.log(0.01), math.log(100)
    ratio = (math.sqrt(5) - 1) / 2
    a, b = right - ratio * (right - left), left + ratio * (right - left)
    fa, fb = objective(a), objective(b)
    for _ in range(80):
        if fa < fb:
            right, b, fb = b, a, fa
            a = right - ratio * (right - left)
            fa = objective(a)
        else:
            left, a, fa = a, b, fb
            b = left + ratio * (right - left)
            fb = objective(b)
    candidates = (math.log(0.01), (left + right) / 2, math.log(100), 0.0)
    return math.exp(min(candidates, key=objective))


def _binary(probabilities: Sequence[float], labels: Sequence[int]) -> None:
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same nonzero length")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("binary labels must be 0 or 1")


def reliability_bins(
    probabilities: Sequence[float], labels: Sequence[int], bins: int = 15
) -> list[dict[str, float | int]]:
    _binary(probabilities, labels)
    if type(bins) is not int or bins <= 0:
        raise ValueError("bins must be a positive integer")
    groups: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for p, y in zip(probabilities, labels, strict=True):
        groups[min(int(p * bins), bins - 1)].append((p, y))
    return [
        {
            "count": len(group),
            "probability": sum(p for p, _ in group) / len(group),
            "frequency": sum(y for _, y in group) / len(group),
        }
        for group in groups
        if group
    ]


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int], bins: int = 15
) -> float:
    return sum(
        item["count"] * abs(item["probability"] - item["frequency"])
        for item in reliability_bins(probabilities, labels, bins)
    ) / len(labels)


def brier(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    _binary(probabilities, labels)
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels, strict=True)) / len(labels)


def auroc(probabilities: Sequence[float], labels: Sequence[int]) -> float | None:
    _binary(probabilities, labels)
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(probabilities, labels, strict=True))
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        rank_sum += sum(y for _, y in ordered[start:end]) * (start + 1 + end) / 2
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def gate(p_fraud: float, low: float, high: float) -> Action:
    if not all(math.isfinite(value) for value in (p_fraud, low, high)):
        raise ValueError("probability and thresholds must be finite")
    if not 0 <= p_fraud <= 1 or not 0 <= low <= high <= 1:
        raise ValueError("require 0 <= low <= high <= 1 and probability in [0, 1]")
    if p_fraud < low:
        return "approve"
    if p_fraud > high:
        return "block_and_contact"
    return "review"


def fit_gates(probabilities: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    """Choose validation boundaries, prioritising leakage when action regions overlap."""
    _binary(probabilities, labels)
    fraud_total = sum(labels)
    if not fraud_total or fraud_total == len(labels):
        return {"low": 0.0, "high": 1.0}
    ordered = sorted(zip(probabilities, labels, strict=True))
    scores = [p for p, _ in ordered]
    cumulative = [0]
    for _, y in ordered:
        cumulative.append(cumulative[-1] + y)
    candidates = sorted({0.0, 1.0, *scores})
    low = max(p for p in candidates if cumulative[bisect_left(scores, p)] / fraud_total <= 0.01)
    # A representable boundary just below the next score can admit that whole score
    # group to blocking while leaving the approval set identical.
    if low > 0:
        below = math.nextafter(low, 0.0)
        if bisect_left(scores, below) == bisect_left(scores, low):
            low = below
    block_candidates = {0.0, 1.0, low, *(math.nextafter(p, 0.0) for p in scores if p > 0)}
    valid_highs = []
    for high in sorted(block_candidates):
        if high < low:
            continue
        index = bisect_right(scores, high)
        count = len(scores) - index
        fraud = fraud_total - cumulative[index]
        if count and fraud / count >= 0.98:
            valid_highs.append(high)
    return {"low": low, "high": min(valid_highs, default=1.0)}


def answer_logits(network: Any, tokens: Any, positions: Any, ids: Any, mx: Any) -> Any:
    """Project only answer positions; do not materialise sequence-by-vocabulary logits."""
    hidden = network.model(tokens)
    hidden = hidden[mx.arange(tokens.shape[0]), positions]
    if network.args.tie_word_embeddings:
        logits = network.model.embed_tokens.as_linear(hidden)
    else:
        logits = network.lm_head(hidden)
    return logits[:, ids].astype(mx.float32)


def plan_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch size must be a positive integer")
    if any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("token lengths must be positive integers")
    ordered = sorted(range(len(lengths)), key=lengths.__getitem__)
    return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def pad_tokens(tokens: list[list[int]], pad: int) -> tuple[list[list[int]], list[int]]:
    """Right-pad rows and return their true answer positions before padding."""
    if not tokens:
        return [], []
    if any(not row for row in tokens):
        raise ValueError("token rows must be nonempty")
    width = max(map(len, tokens))
    return (
        [row + [pad] * (width - len(row)) for row in tokens],
        [len(row) - 1 for row in tokens],
    )


class MlxDecisionModel:
    def __init__(
        self,
        base_model: str = BASE_MODEL,
        adapter: Path | None = None,
        metadata: Path | None = None,
        calibrated: bool = True,
    ) -> None:
        import mlx.core as mx
        from mlx_lm import load

        if not mx.metal.is_available():
            raise RuntimeError("Metal is unavailable; run on an Apple-silicon host with MLX")
        # Cap the buffer cache; prompts vary in length and the cache is keyed by shape.
        mx.set_cache_limit(1024**3)
        self.mx = mx
        self.network, self.tokenizer = load(
            base_model, adapter_path=str(adapter) if adapter else None
        )
        self.network.eval()
        mx.eval(self.network.parameters())
        self.model = base_model + ("+decision-adapter" if adapter else " (zero-shot)")
        self.temperatures = {question.name: 1.0 for question in QUESTIONS}
        if metadata is not None:
            sidecar = json.loads(metadata.read_text())
            if sidecar["base_model"] != base_model:
                raise ValueError("sidecar base model does not match")
            for question in QUESTIONS:
                _, ids = question.token_labels(self.tokenizer)
                if list(ids) != sidecar["label_token_ids"][question.name]:
                    raise ValueError("sidecar label token ids do not match tokenizer")
            if calibrated:
                self.temperatures.update(sidecar["temperatures"])
        for question in QUESTIONS:
            question.token_labels(self.tokenizer)

    def read(self, digest: str, question: Question) -> tuple[list[float], float]:
        start = perf_counter()
        prompt = question.render(digest, self.tokenizer)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        _, ids = question.token_labels(self.tokenizer)
        logits = answer_logits(
            self.network,
            self.mx.array([tokens]),
            self.mx.array([len(tokens) - 1]),
            self.mx.array(ids),
            self.mx,
        )[0]
        self.mx.eval(logits)
        return logits.tolist(), (perf_counter() - start) * 1000

    def read_batch(
        self,
        digests: list[str],
        question: Question,
        batch_size: int = 32,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        # Keep inference within the memory budget for 270–430 token prompts.
        if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch size must be between 1 and {MAX_BATCH_SIZE}")
        tokens = [
            self.tokenizer.encode(question.render(digest, self.tokenizer), add_special_tokens=False)
            for digest in digests
        ]
        batches = plan_batches(list(map(len, tokens)), batch_size)
        if not batches:
            return []
        pad = self.tokenizer.pad_token_id
        if pad is None:
            raise ValueError("tokenizer must define a pad token")
        _, ids = question.token_labels(self.tokenizer)
        results: list[list[float]] = [[] for _ in digests]
        done = 0
        for count, indices in enumerate(batches, 1):
            padded, positions = pad_tokens([tokens[index] for index in indices], pad)
            logits = answer_logits(
                self.network,
                self.mx.array(padded),
                self.mx.array(positions),
                self.mx.array(ids),
                self.mx,
            )
            self.mx.eval(logits)
            for index, row in zip(indices, logits.tolist(), strict=True):
                results[index] = row
            del logits
            if count % 4 == 0 or count == len(batches):
                self.mx.clear_cache()
            done += len(indices)
            if progress is not None:
                progress(done, len(digests))
        return results

    def decide_batch(self, digests: list[str], batch_size: int = 32) -> list[Decision]:
        fraud = self.read_batch(digests, FRAUD, batch_size)
        pattern = self.read_batch(digests, PATTERN, batch_size)
        return [
            Decision(
                apply_temperature(fraud_row, self.temperatures["fraud"])[0],
                dict(
                    zip(
                        PATTERN.labels,
                        apply_temperature(pattern_row, self.temperatures["pattern"]),
                        strict=True,
                    )
                ),
                None,
                self.model,
            )
            for fraud_row, pattern_row in zip(fraud, pattern, strict=True)
        ]

    def decide(self, digest: str) -> Decision:
        fraud, fraud_ms = self.read(digest, FRAUD)
        pattern, pattern_ms = self.read(digest, PATTERN)
        probabilities = apply_temperature(pattern, self.temperatures["pattern"])
        return Decision(
            apply_temperature(fraud, self.temperatures["fraud"])[0],
            dict(zip(PATTERN.labels, probabilities, strict=True)),
            fraud_ms + pattern_ms,
            self.model,
        )
