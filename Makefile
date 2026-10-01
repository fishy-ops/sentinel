.PHONY: install lint test data train eval-flags eval-explanations redteam run demo eval check help

help:
	@printf 'install            Install dependencies\n'
	@printf 'lint               Run lint checks\n'
	@printf 'test               Run tests\n'
	@printf 'data               Generate data splits\n'
	@printf 'train              Train detectors\n'
	@printf 'eval-flags         Evaluate flag detection\n'
	@printf 'eval-explanations  Evaluate reports with a model\n'
	@printf 'eval               Run flag and report evaluation\n'
	@printf 'redteam            Run security checks without a model\n'
	@printf 'run                Start the API with reload\n'
	@printf 'demo               Prepare data and start the console\n'
	@printf 'check              Run lint, format check, and tests\n'

install:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest -q

run:
	uv run uvicorn sentinel.api.main:app --factory --reload

demo:
	uv run python -m sentinel.demo

eval: eval-flags eval-explanations

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest -q

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
