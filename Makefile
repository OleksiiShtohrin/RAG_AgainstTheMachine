PYTHON ?= uv run python
MYPY ?= uv run mypy
FLAKE8 ?= uv run flake8

RAW_DIR ?= data/raw
INDEX_DIR ?= data/processed
DOCS_DATASET ?= data/datasets/UnansweredQuestions/dataset_docs_public.json
CODE_DATASET ?= data/datasets/UnansweredQuestions/dataset_code_public.json
DOCS_GROUND_TRUTH ?= data/datasets/AnsweredQuestions/dataset_docs_public.json
CODE_GROUND_TRUTH ?= data/datasets/AnsweredQuestions/dataset_code_public.json
SEARCH_DIR ?= data/output/search_results/UnansweredQuestions
ANSWER_DIR ?= data/output/search_results_and_answer/UnansweredQuestions
QUERY ?= How to configure OpenAI server?
K ?= 5
HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: install run help index index-semantic index-incremental docs code search search-docs search-code search-hybrid moulinette-docs moulinette-code answer answer-dataset-docs answer-dataset-code evaluate-docs evaluate-code serve api-test demo-hybrid demo-incremental test test-bonus check check-strict clean fclean lint lint-strict

install:
	uv sync --all-extras

run:
	$(PYTHON) -m src --help

help:
	@echo "RAG Against the Machine"
	@echo "Mandatory: install run index search docs code answer answer-dataset-docs answer-dataset-code evaluate-docs evaluate-code test lint"
	@echo "Bonus: index-semantic index-incremental search-hybrid serve api-test test-bonus demo-hybrid demo-incremental"
	@echo "Quality: check check-strict clean fclean lint-strict"

index:
	$(PYTHON) -m src index --max_chunk_size 2000 --raw_dir $(RAW_DIR) --output_dir $(INDEX_DIR)

index-semantic:
	$(PYTHON) -m src index_semantic --output_dir $(INDEX_DIR)

index-incremental:
	$(PYTHON) -m src index_incremental --raw_dir $(RAW_DIR) --output_dir $(INDEX_DIR)

docs:
	$(PYTHON) -m src search_dataset --dataset_path $(DOCS_DATASET) --k 10 --save_directory $(SEARCH_DIR) --index_dir $(INDEX_DIR)

code:
	$(PYTHON) -m src search_dataset --dataset_path $(CODE_DATASET) --k 10 --save_directory $(SEARCH_DIR) --index_dir $(INDEX_DIR)

search:
	$(PYTHON) -m src search "$(QUERY)" --k $(K) --index_dir $(INDEX_DIR)

search-docs:
	$(MAKE) docs

search-code:
	$(MAKE) code

search-hybrid:
	$(PYTHON) -m src search_hybrid "$(QUERY)" --k $(K) --output_dir $(INDEX_DIR)

moulinette-docs:
	./moulinette evaluate_student_search_results $(SEARCH_DIR)/dataset_docs_public.json $(DOCS_GROUND_TRUTH) --k 10 --max_context_length 2000

moulinette-code:
	./moulinette evaluate_student_search_results $(SEARCH_DIR)/dataset_code_public.json $(CODE_GROUND_TRUTH) --k 10 --max_context_length 2000

answer:
	$(PYTHON) -m src answer "$(QUERY)" --k $(K) --index_dir $(INDEX_DIR)

answer-dataset-docs:
	$(PYTHON) -m src answer_dataset --student_search_results_path $(SEARCH_DIR)/dataset_docs_public.json --save_directory $(ANSWER_DIR)

answer-dataset-code:
	$(PYTHON) -m src answer_dataset --student_search_results_path $(SEARCH_DIR)/dataset_code_public.json --save_directory $(ANSWER_DIR)

evaluate-docs:
	$(PYTHON) -m src evaluate --student_search_results_path $(SEARCH_DIR)/dataset_docs_public.json --dataset_path $(DOCS_GROUND_TRUTH)

evaluate-code:
	$(PYTHON) -m src evaluate --student_search_results_path $(SEARCH_DIR)/dataset_code_public.json --dataset_path $(CODE_GROUND_TRUTH)

serve:
	$(PYTHON) -m src serve --host $(HOST) --port $(PORT)

api-test:
	$(PYTHON) -m pytest tests/test_api.py -q

demo-hybrid:
	$(PYTHON) -m src search_hybrid "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 5 --output_dir $(INDEX_DIR)

demo-incremental:
	rm -rf /tmp/rag_incremental_demo
	mkdir -p /tmp/rag_incremental_demo/raw
	printf "print('hello')\n" > /tmp/rag_incremental_demo/raw/demo.py
	$(PYTHON) -m src index_incremental --raw_dir /tmp/rag_incremental_demo/raw --output_dir /tmp/rag_incremental_demo/processed
	$(PYTHON) -m src index_incremental --raw_dir /tmp/rag_incremental_demo/raw --output_dir /tmp/rag_incremental_demo/processed

test:
	$(PYTHON) -m pytest tests -v

test-bonus:
	$(PYTHON) -m pytest tests/test_indexing/test_semantic_index.py tests/test_indexing/test_index_cache.py tests/test_cli_cache.py tests/test_api.py -q

check:
	$(MAKE) lint
	$(MAKE) test

check-strict:
	$(MAKE) lint-strict
	$(MAKE) test

debug:
	$(PYTHON) -m pdb -m src

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
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
