# **************************************************************************** #
#                          RAG against the machine                             #
# **************************************************************************** #

PYTHON ?= uv run python
MYPY ?= uv run mypy
FLAKE8 ?= uv run flake8

.PHONY: install run debug clean fclean lint lint-strict

install:
	uv sync --all-extras

run:
	$(PYTHON) -m src --help

debug:
	$(PYTHON) -m pdb -m src

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .mypy_cache .pytest_cache .ruff_cache

fclean: clean
	rm -rf .venv
	@echo "Virtual environment removed"

lint:
	$(FLAKE8) .
	$(MYPY) . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(FLAKE8) .
	$(MYPY) . --strict
