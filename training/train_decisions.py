"""LoRA training on answer-position label logits, without a language modelling loss."""

import argparse
import importlib.metadata
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sentinel.decide.model import (
    BASE_MODEL,
    answer_logits,
    apply_temperature,
    fit_gates,
    fit_temperature,
)
from sentinel.decide.questions import QUESTIONS, Question
from sentinel.detect.data import read_jsonl
from training.build_decisions import data_hash

MEMORY_LIMIT_GB = 12


@dataclass(frozen=True)
class Example:
    tokens: list[int]
    label: int
    question: str
    allowed_ids: tuple[int, ...]


def encode_examples(rows: list[dict[str, Any]], tokenizer: Any) -> list[Example]:
    questions = {question.name: question for question in QUESTIONS}
    examples = []
    for row in rows:
        question = questions[row["question"]]
        if row["allowed_labels"] != list(question.labels):
            raise ValueError("dataset allowed labels do not match declared question")
        _, ids = question.token_labels(tokenizer)
        tokens = tokenizer.encode(
            question.render(row["digest"], tokenizer), add_special_tokens=False
        )
        examples.append(Example(tokens, question.labels.index(row["answer"]), question.name, ids))
    return examples


def batches(
    examples: list[Example], size: int, rng: random.Random | None = None
) -> list[list[Example]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    groups = []
    for question in QUESTIONS:
        ordered = sorted(
            (example for example in examples if example.question == question.name),
            key=lambda example: len(example.tokens),
        )
        if rng is not None:
            # Shuffle within local length buckets, then shuffle whole batches.
            for start in range(0, len(ordered), size * 16):
                bucket = ordered[start : start + size * 16]
                rng.shuffle(bucket)
                ordered[start : start + size * 16] = bucket
        groups.extend(ordered[start : start + size] for start in range(0, len(ordered), size))
    if rng is not None:
        rng.shuffle(groups)
    return groups


def label_loss(logits: Any, labels: Any, mx: Any, nn: Any) -> Any:
    probabilities = mx.softmax(logits, axis=-1)
    targets = mx.arange(logits.shape[-1])[None, :] == labels[:, None]
    # Multiclass Brier is the sum across allowed classes, averaged across examples.
    brier_term = mx.mean(mx.sum((probabilities - targets) ** 2, axis=-1))
    return nn.losses.cross_entropy(logits, labels, reduction="mean") + 0.5 * brier_term


def batch_arrays(batch: list[Example], pad: int, mx: Any) -> tuple[Any, Any, Any, Any]:
    length = max(len(example.tokens) for example in batch)
    tokens = [example.tokens + [pad] * (length - len(example.tokens)) for example in batch]
    return (
        mx.array(tokens),
        mx.array([len(example.tokens) - 1 for example in batch]),
        mx.array([example.label for example in batch]),
        mx.array(batch[0].allowed_ids),
    )


def validation_metrics(
    network: Any, examples: list[Example], pad: int, mx: Any, nn: Any
) -> tuple[dict[str, Any], dict[str, list[list[float]]], dict[str, list[int]]]:
    network.eval()
    logits_by_question: dict[str, list[list[float]]] = {q.name: [] for q in QUESTIONS}
    labels_by_question: dict[str, list[int]] = {q.name: [] for q in QUESTIONS}
    losses: dict[str, float] = {q.name: 0.0 for q in QUESTIONS}
    for batch in batches(examples, 8):
        tokens, positions, labels, ids = batch_arrays(batch, pad, mx)
        logits = answer_logits(network, tokens, positions, ids, mx)
        loss = label_loss(logits, labels, mx, nn)
        mx.eval(logits, loss)
        question = batch[0].question
        losses[question] += float(loss.item()) * len(batch)
        logits_by_question[question].extend(logits.tolist())
        labels_by_question[question].extend(example.label for example in batch)
    metrics = {}
    for question in QUESTIONS:
        rows, labels = logits_by_question[question.name], labels_by_question[question.name]
        metrics[question.name] = {
            "loss": losses[question.name] / len(rows),
            "accuracy": sum(
                max(range(len(row)), key=row.__getitem__) == label
                for row, label in zip(rows, labels, strict=True)
            )
            / len(rows),
        }
    return metrics, logits_by_question, labels_by_question


def question_metadata(question: Question, tokenizer: Any) -> dict[str, Any]:
    labels, ids = question.token_labels(tokenizer)
    return {**asdict(question), "token_labels": labels, "token_ids": ids}


def train(
    data: Path, adapter: Path, sidecar: Path, base_model: str, epochs: int, learning_rate: float
) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    if not mx.metal.is_available():
        raise RuntimeError("Metal is unavailable; run on an Apple-silicon host with MLX")
    if epochs not in (1, 2) or not 5e-5 <= learning_rate <= 1e-4:
        raise ValueError("use 1-2 epochs and learning rate in [5e-5, 1e-4]")
    # MLX keeps freed buffers in a cache keyed by shape. Batches differ in length, so without
    # a cap the cache grows until the machine runs out of memory.
    mx.set_memory_limit(MEMORY_LIMIT_GB * 1024**3)
    mx.set_cache_limit(1024**3)
    mx.random.seed(7)
    rng = random.Random(7)
    network, tokenizer = load(base_model)
    network.freeze()
    lora = {
        "rank": 16,
        "scale": 16.0,
        "dropout": 0.0,
        "keys": ["self_attn.q_proj", "self_attn.v_proj"],
    }
    linear_to_lora_layers(network, 16, lora)
    mx.eval(network.parameters())
    training = encode_examples(read_jsonl(data / "train.jsonl"), tokenizer)
    valid = encode_examples(read_jsonl(data / "valid.jsonl"), tokenizer)
    if not training or not valid:
        raise ValueError("training and validation examples cannot be empty")
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)

    def loss_fn(model: Any, tokens: Any, positions: Any, labels: Any, ids: Any) -> Any:
        return label_loss(answer_logits(model, tokens, positions, ids, mx), labels, mx, nn)

    value_and_grad = nn.value_and_grad(network, loss_fn)
    history = []
    for epoch in range(epochs):
        network.train()
        total, count = 0.0, 0
        epoch_batches = batches(training, 8, rng)
        for step, batch in enumerate(epoch_batches, 1):
            arrays = batch_arrays(batch, pad, mx)
            loss, gradients = value_and_grad(network, *arrays)
            optimizer.update(network, gradients)
            mx.eval(network.trainable_parameters(), optimizer.state, loss)
            total += float(loss.item()) * len(batch)
            count += len(batch)
            if step % 10 == 0 or step == len(epoch_batches):
                print(
                    f"epoch={epoch + 1} step={step}/{len(epoch_batches)} "
                    f"train_loss={total / count:.5f}",
                    flush=True,
                )
        metrics, logits, labels = validation_metrics(network, valid, pad, mx, nn)
        record = {"epoch": epoch + 1, "train_loss": total / count, "valid": metrics}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    temperatures = {q.name: fit_temperature(logits[q.name], labels[q.name]) for q in QUESTIONS}
    probabilities = [apply_temperature(row, temperatures["fraud"])[0] for row in logits["fraud"]]
    truth = [int(label == 0) for label in labels["fraud"]]
    thresholds = fit_gates(probabilities, truth)
    adapter.mkdir(parents=True, exist_ok=True)
    config = {"fine_tune_type": "lora", "num_layers": 16, "lora_parameters": lora}
    (adapter / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    mx.save_safetensors(
        str(adapter / "adapters.safetensors"), dict(tree_flatten(network.trainable_parameters()))
    )
    metadata = {
        "base_model": base_model,
        "questions": [question_metadata(q, tokenizer) for q in QUESTIONS],
        "label_token_ids": {q.name: list(q.token_labels(tokenizer)[1]) for q in QUESTIONS},
        "temperatures": temperatures,
        "gate_thresholds": thresholds,
        "gate_semantics": "approve p < low; block p > high; equality goes to review",
        "data_hash": data_hash([data / "train.jsonl", data / "valid.jsonl"]),
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "transformers", "tokenizers")
        },
        "training": {
            "seed": 7,
            "epochs": epochs,
            "batch_size": 8,
            "learning_rate": learning_rate,
            "brier_weight": 0.5,
            "lora": config,
            "history": history,
            "majority_probability": sum(e.label == 0 for e in training if e.question == "fraud")
            / sum(e.question == "fraud" for e in training),
        },
        "validation": {
            "flags": len(truth),
            "fraud": sum(truth),
            "majority_probability": sum(truth) / len(truth),
        },
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"adapter: {adapter}; calibration: {sidecar}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("training/data/decisions"))
    parser.add_argument("--adapter", type=Path, default=Path("training/decision-adapter"))
    parser.add_argument("--sidecar", type=Path, default=Path("models/decision.json"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=7.5e-5)
    args = parser.parse_args()
    train(args.data, args.adapter, args.sidecar, args.base_model, args.epochs, args.learning_rate)


if __name__ == "__main__":
    main()
