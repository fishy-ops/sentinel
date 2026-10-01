.PHONY: install lint test data train eval-flags eval-explanations redteam run

install:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest -q

run:
	uv run uvicorn sentinel.api.main:app --factory --reload

data:
	@for split in train val eval finetune; do \
		uv run python -m data.synth --out data/generated/$$split --seed 42 --accounts 100 --days 60 --split $$split || exit 1; \
	done

train:
	uv run python -m sentinel.detect.train

eval-flags:
	uv run python -m evals.run_flags

eval-explanations:
	uv run python -m evals.run_explanations --models qwen2.5:7b --per-group 6 --seed 7

redteam:
	uv run python -m evals.redteam
