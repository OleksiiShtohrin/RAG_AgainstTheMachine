*This project has been created as part of the 42 curriculum by oshtohri.*

# RAG against the machine

## Description

RAG against the machine is a Retrieval-Augmented Generation system for searching and answering questions about a codebase.

The system:
- reads Python, Markdown, and text files from a corpus;
- splits documents into chunks of at most 2000 characters;
- builds a BM25 lexical index;
- retrieves relevant sources with exact file paths and character ranges;
- generates grounded answers with Qwen/Qwen3-0.6B;
- evaluates retrieval results against reference datasets;
- provides optional semantic, hybrid, incremental, caching, and HTTP API features.

The project is implemented in Python 3.10+ and uses `uv` for dependency management.

## Requirements

- Python 3.10+
- `uv`

## Installation

```bash
uv sync --all-extras
```

The project is configured to use the CPU-only PyTorch index.

## Project Structure

```text
.
├── src/
│   ├── chunking/
│   ├── evaluation/
│   ├── generation/
│   ├── indexing/
│   ├── ingestion/
│   ├── models/
│   ├── retrieval/
│   ├── utils/
│   ├── cli.py
│   └── server.py
├── data/
├── tests/
├── Makefile
├── pyproject.toml
└── uv.lock
```

The corpus, datasets, generated indexes, and tests are not required in the final submitted source package.

## Usage

The CLI is implemented with Python Fire.

General form:

```bash
uv run python -m src <command> [options]
```

### Mandatory commands

Build the lexical index:

```bash
uv run python -m src index \
  --max_chunk_size 2000 \
  --raw_dir data/raw \
  --output_dir data/processed
```

Search one query:

```bash
uv run python -m src search \
  "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  --k 5 \
  --index_dir data/processed
```

Search a dataset:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions \
  --index_dir data/processed
```

Generate one answer:

```bash
uv run python -m src answer \
  "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  --k 5 \
  --index_dir data/processed
```

The default generation model is `Qwen/Qwen3-0.6B`.

Answer a search-results dataset:

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Evaluate retrieval results:

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

The internal `evaluate` command is independent of the Moulinette. Official retrieval evaluation is performed separately with the provided `moulinette`.

## Makefile

Useful targets:

```bash
make install
make index
make search
make docs
make code
make answer
make answer-dataset-docs
make answer-dataset-code
make evaluate-docs
make evaluate-code
make test
make lint
make check
make check-strict
```

Bonus targets:

```bash
make index-semantic
make index-incremental
make search-hybrid
make serve
make api-test
make test-bonus
make demo-hybrid
make demo-incremental
```

Run:

```bash
make help
```

to see the available targets.

## System Architecture

The main pipeline is:

```text
Raw corpus
    │
    ▼
CorpusReader
    │
    ▼
Document
    │
    ▼
ChunkerFactory
    │
    ├── PythonChunker
    └── MarkdownChunker
    │
    ▼
Chunks
    │
    ▼
BM25Index
    │
    ▼
LexicalRetriever
    │
    ▼
MinimalSource results
    │
    ▼
Qwen/Qwen3-0.6B
    │
    ▼
MinimalAnswer
```

The implementation separates ingestion, chunking, indexing, retrieval, generation, evaluation, models, and utilities. Retrievers share a common interface, allowing lexical, semantic, and hybrid strategies to be used without coupling the rest of the pipeline to one retrieval implementation.

## Chunking Strategy

The maximum chunk size is 2000 characters.

Python files are parsed with the Python AST where possible. Python-aware chunking keeps logical code structures together while respecting the maximum size.

Markdown and text content use the Markdown/text chunking strategy.

Each chunk preserves its source file path and character range so retrieved results can point back to the original corpus.

## Retrieval Method

### Lexical retrieval

The mandatory retrieval implementation uses BM25.

The tokenizer is identifier-aware and splits identifier components so that terms inside names such as `openai_compatible_server` can be matched more effectively.

Retrieval returns `MinimalSource` objects containing:
- exact `file_path`;
- `start_char`;
- `end_char`;
- relevance score.

### Semantic retrieval

The semantic bonus uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model is loaded through Hugging Face Transformers. Embeddings use mean pooling with the attention mask followed by L2 normalization. Similarity is computed using the normalized dot product.

The semantic index is persisted to disk and can be loaded for retrieval.

### Hybrid retrieval

Hybrid retrieval combines BM25 and semantic retrieval.

Candidates are retrieved from both systems and combined using Reciprocal Rank Fusion (RRF), producing one ranked result list.

Example:

```bash
uv run python -m src search_hybrid \
  "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  --k 5 \
  --output_dir data/processed
```

## Answer Generation

The answer generator uses `Qwen/Qwen3-0.6B` by default.

Retrieved sources are converted into context for the model. The generator is designed to:
- use retrieved evidence;
- stay within the available context/token budget;
- produce coherent and relevant answers;
- avoid relying on information outside the retrieved context where possible.

The answer pipeline returns the required search results together with the generated answer.

## Data Models and Output

The project uses Pydantic models for the required data structures:

- `MinimalSource`
- `UnansweredQuestion`
- `AnsweredQuestion`
- `RagDataset`
- `MinimalSearchResults`
- `MinimalAnswer`
- `StudentSearchResults`
- `StudentSearchResultsAndAnswer`

Search output uses `StudentSearchResults`.

Answer output uses `StudentSearchResultsAndAnswer`.

Source locations preserve exact file paths and character ranges.

## Evaluation

Retrieval quality is measured using Recall@5.

Official evaluation is performed with the provided Moulinette:

```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5 \
  --max_context_length 2000
```

The project also contains an internal `evaluate` command for local evaluation. It does not import or call the Moulinette.

### Verified Recall@5

Latest verified mandatory retrieval results:

```text
Documentation: 83.0%
Code:          79.8%
```

These satisfy the project targets of at least 80% for documentation and 50% for code.

## Performance

Mandatory retrieval meets the required Recall@5 thresholds.

Indexing provides progress feedback with `tqdm`. Mandatory lexical indexing is separated from semantic indexing because semantic embedding of a large corpus is substantially more expensive on CPU.

## Bonus Features

### 1. Semantic embeddings

Dense-vector retrieval using `sentence-transformers/all-MiniLM-L6-v2` through the Transformers API.

### 2. Hybrid retrieval

Combines BM25 and semantic rankings using Reciprocal Rank Fusion.

### 3. Incremental indexing

Tracks corpus files using SHA-256 hashes.

On subsequent runs:
- unchanged files are skipped;
- changed files are re-read and re-chunked;
- deleted files are removed from the current corpus representation.

The BM25 index is rebuilt from the current complete set of chunks after changed/deleted files are processed.

Example:

```bash
uv run python -m src index_incremental \
  --raw_dir data/raw \
  --output_dir data/processed
```

### 4. Caching

Two cache mechanisms are provided.

`IndexCache` keeps loaded indexes in memory so repeated loads within the same process do not repeatedly deserialize the BM25 index.

`QueryCache` stores query results on disk using deterministic SHA-256 keys, allowing repeated queries to reuse previously computed results.

### 5. Local HTTP API

A local FastAPI server exposes the RAG functionality.

Start it with:

```bash
uv run python -m src serve --host 127.0.0.1 --port 8000
```

Available endpoints:

```text
GET  /health
POST /search
POST /answer
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Design Decisions

### BM25 for mandatory retrieval

BM25 provides an efficient lexical baseline and works particularly well for code search when queries contain exact identifiers, function names, configuration keys, or endpoint paths.

### AST-aware Python chunking

Code should not be split arbitrarily when logical structures can be preserved. AST-based chunking keeps useful Python structures together while respecting the maximum chunk size.

### Separate retriever implementations

Retrieval is abstracted behind a common interface. This allows lexical, semantic, and hybrid retrieval to be used without coupling the rest of the RAG pipeline to one strategy.

### Exact source ranges

Returning file paths and character offsets makes retrieval results traceable to the original corpus and supports grounded answer generation.

### CPU-friendly semantic retrieval

The semantic bonus uses a lightweight embedding model and reuses the loaded tokenizer and model during repeated scoring.

## Challenges and Solutions

### Large corpus

The corpus contains many files and chunks. Indexing uses `tqdm` for progress feedback and keeps the mandatory lexical pipeline separate from the more expensive semantic bonus.

### Code retrieval

Generic tokenization can lose information contained in identifiers. Identifier-aware tokenization improves matching of code-specific terms.

### Semantic retrieval performance

Repeatedly loading the Transformer model would make batch retrieval impractical. The implementation reuses the loaded tokenizer and model and stores the semantic index on disk.

### Incremental updates

Re-indexing every file after every corpus change is unnecessary. SHA-256 manifests identify unchanged, changed, and deleted files so only affected files need to be reprocessed before rebuilding the current BM25 index.

### Repeated queries

Disk-backed query caching avoids repeating identical retrieval work across processes.

## Testing

The project uses pytest for unit, integration, regression, and API tests.

Latest full verification:

```text
108 passed
```

Standard and strict quality checks both pass:

```bash
make check
make check-strict
```

The project also passes the required flake8 and mypy checks, including:

```bash
mypy . --strict
```

## Example End-to-End Workflow

```bash
# Install
make install

# Build mandatory BM25 index
make index

# Search
make search QUERY="What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?"

# Generate an answer
make answer QUERY="What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?"

# Run tests and quality checks
make check
make check-strict
```

Bonus workflow:

```bash
make index-semantic
make search-hybrid
make index-incremental
make serve
```

## Resources and AI Usage

The project specification and provided datasets are the primary references for required behavior.

AI assistance was used during development for:
- discussing architecture and separation of responsibilities;
- planning incremental implementation steps;
- explaining Python, typing, testing, and concurrency concepts;
- reviewing implementation ideas and debugging errors;
- generating and refining tests and documentation.

The final implementation was tested locally and verified with the project's automated test and quality checks.
