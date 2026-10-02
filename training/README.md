# sentinel-analyst:1.5b

A LoRA fine-tune of [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) for one job: review a flagged transaction with Sentinel's five read-only tools and write a short report in which every fact cites the record it came from.

## Why fine-tune

The stock 1.5B model cannot do the task: it sends invented arguments to the first tool and never produces a report. The stock 7B model can, but slowly, and it often cites the wrong record. The fine-tuned 1.5B gets more reports right on the first attempt than the 7B, in under a quarter of the time. Numbers are in the [project README](../README.md#analyst-reports).

## Training data

`build_dataset.py` generates the data; nothing is hand-written and no real customer data is involved.

- Source: the `finetune` split of the synthetic generator (600 accounts, seed 1337). It shares no accounts with the `eval` split used for every reported number.
- Each example is a complete conversation: system prompt, tool calls, the real tool results, and a target report.
- The verdict in each target follows the strength of the evidence gathered: two or more independent signals read as likely fraud, one as needing a person's review, none as thin.
- Target reports are assembled from the tool results, so each number, date, country, device, and merchant in them is copied from a cited record. Every target is run through the same grounding checker used at inference, and rejected if it fails (150 of 1,339 were).
- 15% of examples carry instruction-like text in a merchant name or memo, with a target that reports it as data. The strings differ from the ones used in evaluation.
- Conversations are rendered to ChatML in three tool-listing formats. Chat servers serialise the tool list differently, and a model trained on a single format produced malformed tool calls when served through Ollama.
- 1,071 training and 118 validation conversations, 2,500 to 6,000 tokens each.

## Training

MLX LoRA on an Apple M4 Pro: 16 layers, 1,500 iterations, batch size 1 with 4-step gradient accumulation, learning rate 1e-4, max sequence length 8192, seed 7. Validation loss fell from 1.52 to 0.12. The adapter is fused, converted to GGUF, quantised to Q4_K_M (986 MB), and registered with Ollama.

```sh
LLAMA_CPP=/path/to/llama.cpp training/finetune.sh
```

## Limitations

- The model is trained on reports with a fixed structure and a fixed tool set. It is not a general assistant and will not transfer to other tools without retraining.
- Training targets and the evaluation metric share the same grounding checker. The checker verifies literal facts against cited records; it does not judge whether the reasoning that connects them is sound.
- Risk level and recommended action were learned from synthetic labels. It never recommended `approve` in evaluation, including for legitimate transactions, so it should be read as conservative triage, not a decision.
- On accounts with almost no history it tends to state statistics the tools did not return; both sparse-history cases in evaluation failed the check.
- Results moved by several points between training runs with different data sizes. Treat differences of that size as noise.
- All data is synthetic. Behaviour on real transaction data is untested.

## Typed decision model

The decision model is a separate second stage for **flagged transactions only**. It returns
`P(fraud)` and probabilities for six patterns rather than a report. Each declared question
uses one Qwen2.5-1.5B-Instruct forward pass, reads only the allowed answer-label logits, and
renormalises them. A complete decision uses two forwards and generates zero tokens. There is
no shared prefix cache in this implementation.

The approach draws on [TypeSafe's Jev](https://typesafe.ai/), which exposes typed decisions,
[Convai's Laya](https://github.com/NandhaKishorM/laya), an open decision engine with a
ModernBERT encoder, and the [JevLite recipe](https://arxiv.org/abs/2609.23959), which uses a
small language model's label logits and temperature calibration. This implementation uses
Qwen and a supervised answer-position loss; it does not replicate Jev's architecture or
Laya's reinforcement learning objective.

### Evidence and data

The digest calls `BoundTools.get_flag_detail` and `get_account_stats` through the existing
read-only tool dispatcher. History is strictly earlier than the flagged transaction. It
omits account/transaction identifiers, labels, and model-version metadata. Values retain the
tools' rounding. Merchant and memo are quoted on `untrusted.*` lines, stripped of control
characters, truncated to 60 characters, and escaped against chat delimiters. Less essential
trailing fields are dropped to cap text at 1,250 characters; the real tokenizer then enforces
350 tokens during training and inference. This can omit detector reasons on unusually long
digests. Untrusted delimiters are a boundary cue, not a guarantee against prompt injection.

`build_decisions.py` freshly generates the `finetune` split (600 accounts, seed 1337) in a
temporary directory and seeds a temporary database with the existing detectors. TRAIN
contains only those flags. VALID contains all flags from the existing `val` split. Account
prefixes and disjointness are checked; evaluation data is never used for training,
calibration, or gate selection. Each flag produces two examples. Labels come only from
`labels.jsonl`: fraud yes/no, and A velocity burst, B account takeover, C amount spike,
D structuring, E dormant drain, F legitimate activity. The builder reports class balance.

### Loss, calibration, and gates

The custom MLX loop tunes rank-16 LoRA adapters on the query/value projections of the top 16
layers with AdamW, learning rate 7.5e-5, two epochs, batch size eight, length bucketing, and
seed seven. It trains only at the answer position: cross-entropy over the allowed logits
plus 0.5 times multiclass Brier loss (sum over classes, mean over examples). Neither prompt
tokens nor output prose contribute to the loss. Right padding is causal and answer positions
are gathered independently for each example. Label spellings must encode to distinct single
tokens; otherwise training and inference fail.

After training, one temperature per question minimises VALID negative log likelihood with
a bounded one-dimensional search (0.01–100). VALID also selects gates using observed score
boundaries: the largest `low` allowing at most 1% of all fraudulent flags to be auto-approved,
and the smallest compatible `high` with auto-block precision at least 0.98. If the
independently eligible regions overlap, the leakage constraint takes priority and blocking
is restricted to scores outside the approval region. Boundaries just below observed scores
keep ties intact without unnecessarily disabling a block region. `p < low` approves,
`p > high` blocks and contacts, and everything else goes to review. Equal scores stay together;
unsupported auto-actions are disabled with 0/1 thresholds. Single-class validation disables
both actions. Constraints are empirical, without a statistical guarantee on new data.

The adapter stays in ignored `training/decision-adapter/`; `models/decision.json` is produced
only after successful training and records the base model, questions, token ids,
temperatures, gates, dataset hash, library versions, and actual training/validation metrics.
It should be reviewed and committed alongside any measured results. No fused or GGUF copy
is needed.

### Run

On an Apple-silicon host with Metal access, using the existing generated data and detector
artifacts:

```sh
uv sync --group train
uv run python -m training.build_decisions
uv run --group train python -m training.train_decisions
make eval-decisions
make check
node --test tests/
```

Cache locations can be set with shell environment variables before these commands. Linux CI
runs the model-free tests without MLX; the `train` dependency is restricted to macOS.

Evaluation uses **all** flags of the untouched `eval` split. It compares a constant training
prevalence/majority predictor, existing supervised gradient boosting, base Qwen zero-shot,
and the trained readout before/after temperature. It reports accuracy, AUROC, ECE (15 bins),
Brier, full-decision median/p95 latency, pattern confusion matrices, gate coverage, leakage,
and block precision. Raw/calibrated rows share the same measured logits and latency; load
and warm-up are excluded. Robustness checks reverse the pattern options, paraphrase the
fraud question, and place the explanation evaluation's injection string in merchant/memo.
The 60-character truncation applies to injected text too. Results go to
`evals/results/decisions.json`, `decisions.md`, `docs/results/decisions.png`, and ignored
`evals/failures/decisions.jsonl`. The comparison explicitly identifies metrics where gradient
boosting wins. No results or calibration values are claimed before these commands succeed.

### Limitations

All labels and evaluation accounts are synthetic. This evaluates triage conditional on a
flag; it cannot recover fraud the first-stage detectors missed. Temperature changes
confidence, not the argmax answer or fraud AUROC. Probabilities depend on the selected
labels, prompt wording, and option order; neutral keys reduce one source of bias but do not
remove it. False alarms and ambiguous pattern labels can remain even with calibration.
Gate thresholds tuned on one synthetic validation split can drift on real traffic. Latency
must be measured on the deployment machine; zero output tokens alone does not imply tens
of milliseconds. This module is available for explicit callers and does not change the
existing API, dashboard, or generative report workflow.
