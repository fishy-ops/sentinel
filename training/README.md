# sentinel-analyst:1.5b

A LoRA fine-tune of [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) for one job: review a flagged transaction with Sentinel's five read-only tools and write a short report in which every fact cites the record it came from.

## Why fine-tune

The stock 1.5B model cannot do the task: it sends invented arguments to the first tool and never produces a report. The stock 7B model can, but slowly, and it often cites the wrong record. The fine-tuned 1.5B is more accurate than the 7B at about a fifth of the latency. Numbers are in the [project README](../README.md#analyst-reports).

## Training data

`build_dataset.py` generates the data; nothing is hand-written and no real customer data is involved.

- Source: the `finetune` split of the synthetic generator (300 accounts, seed 1337). It shares no accounts with the `eval` split used for every reported number.
- Each example is a complete conversation: system prompt, tool calls, the real tool results, and a target report.
- Target reports are assembled from the tool results, so each number, date, country, device, and merchant in them is copied from a cited record. Every target is run through the same grounding checker used at inference, and rejected if it fails (54 of 592 were).
- 12% of examples carry instruction-like text in a merchant name or memo, with a target that reports it as data. The strings differ from the ones used in evaluation.
- Conversations are rendered to ChatML in three tool-listing formats. Chat servers serialise the tool list differently, and a model trained on a single format produced malformed tool calls when served through Ollama.
- 485 training and 53 validation conversations, 2,500 to 6,000 tokens each.

## Training

MLX LoRA on an Apple M4 Pro: 16 layers, 900 iterations, batch size 1 with 4-step gradient accumulation, learning rate 1e-4, max sequence length 8192, seed 7. Validation loss fell from 1.59 to 0.13. The adapter is fused, converted to GGUF, quantised to Q4_K_M (986 MB), and registered with Ollama.

```sh
LLAMA_CPP=/path/to/llama.cpp training/finetune.sh
```

## Limitations

- The model is trained on reports with a fixed structure and a fixed tool set. It is not a general assistant and will not transfer to other tools without retraining.
- Training targets and the evaluation metric share the same grounding checker. The checker verifies literal facts against cited records; it does not judge whether the reasoning that connects them is sound.
- Risk level and recommended action were learned from synthetic labels. The model never recommended `approve` in evaluation, including for legitimate transactions, so it should be read as conservative triage, not a decision.
- All data is synthetic. Behaviour on real transaction data is untested.
