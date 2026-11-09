.PHONY: install test lint check fixtures inject testset eval-fixtures clean

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

lint:
	python -m ruff check .

check: lint test

# Record real provider responses into data/fixtures (network; resumable).
fixtures:
	python pipelines/01_record_fixtures.py

# Replay every fixture offline and prove determinism (no network).
eval-fixtures:
	python pipelines/01_record_fixtures.py --verify

inject:
	python pipelines/02_inject_failures.py

testset:
	python pipelines/03_make_seed_questions.py

clean:
	rm -rf .pytest_cache .ruff_cache
