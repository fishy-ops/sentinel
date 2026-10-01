.PHONY: install lint test data

install:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest -q

data:
	@for split in train val eval finetune; do \
		uv run python -m data.synth --out data/generated/$$split --seed 42 --accounts 100 --days 60 --split $$split || exit 1; \
	done
