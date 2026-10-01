#!/usr/bin/env bash
# Fine-tune the analyst model and register it with Ollama as sentinel-analyst:1.5b.
# Requires Apple silicon (MLX), Ollama, and a local llama.cpp checkout for GGUF conversion:
#   LLAMA_CPP=/path/to/llama.cpp training/finetune.sh
set -euo pipefail

BASE="Qwen/Qwen2.5-1.5B-Instruct"
ROOT="$(cd "$(dirname "$0")" && pwd)"
: "${LLAMA_CPP:?set LLAMA_CPP to a llama.cpp checkout}"

cd "$ROOT/.."
uv run python -m training.build_dataset --out training/data --accounts 600 --seed 1337

uvx --from mlx-lm mlx_lm.lora --model "$BASE" --train --data training/data \
  --adapter-path training/adapters --iters 1500 --batch-size 1 --grad-accumulation-steps 4 \
  --num-layers 16 --learning-rate 1e-4 --max-seq-length 8192 --seed 7 \
  --steps-per-eval 500 --val-batches 20 --save-every 500

uvx --from mlx-lm mlx_lm.fuse --model "$BASE" --adapter-path training/adapters \
  --save-path training/build/fused

# Older converters expect `additional_special_tokens` rather than a list under
# `extra_special_tokens`.
python3 - <<'PY'
import json
path = "training/build/fused/tokenizer_config.json"
config = json.load(open(path))
if isinstance(config.get("extra_special_tokens"), list):
    config["additional_special_tokens"] = config.pop("extra_special_tokens")
    json.dump(config, open(path, "w"), indent=1, ensure_ascii=False)
PY

uv run --no-project \
  --with-requirements "$LLAMA_CPP/requirements/requirements-convert_hf_to_gguf.txt" \
  python "$LLAMA_CPP/convert_hf_to_gguf.py" training/build/fused \
  --outfile training/build/sentinel-analyst-1.5b-f16.gguf --outtype f16

for size in 1.5b 7b; do
  ollama create "sentinel-base:$size" -f "training/modelfiles/base-$size.Modelfile"
done
(cd training/modelfiles && ollama create sentinel-analyst:1.5b -q q4_K_M -f analyst.Modelfile)

# The f16 GGUF and fused weights are only needed for the import.
rm -rf training/build
