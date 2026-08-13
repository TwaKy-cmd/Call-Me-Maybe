.PHONY: install run debug clean fclean re lint lint-strict test

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	find . -not -path "./.venv/*" -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -not -path "./.venv/*" -type f -name "*.pyc" -delete

# clean + everything that `make install` and `make run` can regenerate.
fclean: clean
	rm -rf .venv
	rm -rf data/output

re: fclean install

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run pytest