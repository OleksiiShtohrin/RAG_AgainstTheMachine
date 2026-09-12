*This project has been created as part of the 42 curriculum by oshtohri.*

# RAG against the machine

## Description

RAG against the machine is a high-performance Retrieval-Augmented Generation (RAG) system engineered for searching, navigating, and answering complex questions about large source code repositories (specifically the `vLLM` codebase).

The system:
- Ingests Python, Markdown, and text files from a raw repository corpus;
- Splits documents into structurally sound chunks strictly bounded by 2000 characters;
- Builds an in-memory, disk-persisted BM25Okapi lexical index with subword identifier tokenization;
- Retrieves candidate source snippets with verbatim file paths and character offset spans;
- Synthesizes grounded answers from retrieved context. using local SLMs (`Qwen/Qwen3-0.6B`);
- Evaluates retrieval quality with Recall@k metrics based on character-level Intersection over Union (IoU);
- Implements 5 bonus features: semantic embeddings, hybrid search (RRF), incremental delta indexing, disk-backed query caching, and a local FastAPI REST server.

The project is developed in Python 3.10+ and uses `uv` for dependency management.

---

## Requirements

- Python 3.10+
- `uv` package manager

---

## Installation

```bash
make install
# or directly:
uv sync --all-extras
```

The project is configured to use the CPU-only PyTorch index.

## Project Structure

```text
.
├── src/
│   ├── chunking/          # Chunker strategies (AST Python, Markdown sliding window)
│   ├── evaluation/        # IoU overlap (>= 0.05) & Recall@k metrics
│   ├── generation/        # Prompt engineering & local Qwen inference
│   ├── indexing/          # Tokenization, BM25, vector embeddings & cache
│   ├── ingestion/         # File discovery and corpus reading
│   ├── models/            # Strict Pydantic data schemas
│   ├── retrieval/         # BaseRetriever, Lexical, Semantic & Hybrid rankers
│   ├── utils/             # Safe file I/O & query caching layer
│   ├── cli.py             # Python Fire CLI orchestration
│   └── server.py          # Local FastAPI HTTP REST API
├── data/                  # Raw corpus, datasets, processed index, and output artifacts
├── tests/                 # Comprehensive unit, integration, and API tests
├── Makefile               # Automated build, lint, and workflow rules
├── pyproject.toml         # Project metadata, dependencies, and linter settings
└── uv.lock                # Locked dependency tree
```

The corpus, datasets, generated indexes, and tests are not required in the final submitted source package.

## Usage

The CLI(Command-Line Interface) is implemented with Python Fire.

General form:

```bash
uv run python -m src <command> [options]
```

### Mandatory commands

1. **Build the lexical index:**

```bash
uv run python -m src index \
  --max_chunk_size 2000 \
  --raw_dir data/raw \
  --output_dir data/processed
```

2. **Search one query:**

```bash
uv run python -m src search \
  "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  --k 5 \
  --index_dir data/processed
```

3. **Search a dataset:**

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions \
  --index_dir data/processed
```

4. **Generate one answer:**

```bash
uv run python -m src answer \
  "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  --k 5 \
  --index_dir data/processed
```

The default generation model is `Qwen/Qwen3-0.6B`.

5. **Answer a search-results dataset:**

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

6. **Evaluate retrieval results (Recall@k):**

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

The internal `evaluate` command is independent of the Moulinette. Official retrieval evaluation is performed separately with the provided `moulinette`.

## Makefile

Useful targets:

```bash
make install               # Sync environment via uv
make index                 # Build mandatory BM25 index
make search                # Run search with default QUERY
make docs                  # Search documentation dataset
make code                  # Search code dataset
make answer                # Answer default QUERY
make answer-dataset-docs   # Batch generate answers for docs
make answer-dataset-code   # Batch generate answers for code
make evaluate-docs         # Run local evaluation on docs
make evaluate-code         # Run local evaluation on code
make moulinette-docs       # Run official Moulinette evaluation on docs
make moulinette-code       # Run official Moulinette evaluation on code
make lint                  # Run flake8 and mypy checks
make lint-strict           # Run flake8 and strict mypy checks
make check-strict          # Run strict linters and full test suite
make clean                 # Remove temporary cache artifacts
```

Bonus targets:

```bash
make index-semantic        # Build vector embedding index (Bonus 1)
make search-hybrid         # Run hybrid BM25 + Vector search (Bonus 2)
make index-incremental     # Run SHA-256 delta re-indexing (Bonus 3)
make serve                 # Launch local FastAPI server at port 8000 (Bonus 5)
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
CorpusReader (Ingestion)
    │
    ▼
Document
    │
    ▼
ChunkerFactory
    │
    ├── PythonChunker (AST Structural Blocks)
    └── MarkdownChunker (Heading & Paragraph Sliding Window)
    │
    ▼
Chunks (Strict <= 2000 character validation)
    │
    ▼
BM25Index / SemanticIndex
    │
    ▼
BaseRetriever (Abstraction Layer)
    ├── LexicalRetriever (BM25 Keyword Matching)
    ├── SemanticRetriever (Cosine Dense Vector Similarity)
    └── HybridRetriever (Reciprocal Rank Fusion)
    │
    ▼
MinimalSource results
    │
    ▼
Local LLM (Qwen/Qwen3-0.6B)
    │
    ▼
Structured Pydantic Output (MinimalAnswer & StudentSearchResults)
```

The implementation separates ingestion, chunking, indexing, retrieval, generation, evaluation, models, and utilities. Retrievers share a common interface, allowing lexical, semantic, and hybrid strategies to be used without coupling the rest of the pipeline to one retrieval implementation.

- **Single Responsibility (SRP):** Slicing, indexing, candidate retrieval, context formulation, and inference are segregated into independent modules.
- **Open-Closed & Dependency Inversion (OCP / DIP):** All rankers implement the abstract `BaseRetriever` interface. The generation pipeline queries the interface rather than a concrete implementation, allowing seamless switching between lexical, dense semantic, and hybrid models.

---

## Chunking Strategy

A central requirement is that **no retrieved chunk may exceed 2000 characters**. 

1. **Markdown / Text Chunking (`MarkdownChunker`):**
  - Applies an adaptive sliding window (~800 characters) with a 150-character overlap.
  - Snaps to semantic boundaries: double newlines (`\n\n`) for paragraphs and Markdown heading markers (`#`, `##`).
2. **Python Code Chunking (`PythonChunker`):**
  - Uses the Python Abstract Syntax Tree (`ast`) to extract structural boundaries of classes, functions, and async declarations.
  - Preserves leading comments and docstrings alongside their parent declarations.
  - Falls back gracefully to `ChunkAssembler` sliding windows if AST parsing encounters invalid or non-standard syntax.

---

## Retrieval Method

### 1. Lexical Retrieval (BM25Okapi)
- Uses **BM25Okapi** from `rank-bm25`.
- **Subword Code Tokenization:** A specialized regex pipeline deconstructs `camelCase` and `snake_case` identifiers (e.g. `openai_compatible_server` $\to$ `openai`, `compatible`, `server`), enabling natural language queries to match exact code identifiers.
- **Path Boosting:** Relative file paths are indexed twice alongside chunk text, prioritizing files whose path directly names the query topic (e.g. `lora.md`).

### 2. Semantic Retrieval (Bonus 1)
- Uses `sentence-transformers/all-MiniLM-L6-v2` via the Hugging Face Transformers library.
- Computes 384-dimensional dense vectors using mean pooling over active attention masks, normalized via L2 norm.
- Measures similarity using the matrix dot product (equivalent to cosine similarity).

### 3. Hybrid Retrieval with RRF (Bonus 2)
- Merges candidate lists from lexical and semantic rankers using **Reciprocal Rank Fusion (RRF)**:
  $$RRF\_score(d) = \frac{1}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{1}{60 + \text{rank}_{\text{Semantic}}(d)}$$
- Resolves score calibration mismatches between unbounded BM25 scores and bounded cosine similarities.

Example:

```bash
uv run python -m src search_hybrid \
  "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  --k 5 \
  --output_dir data/processed
```

## Answer Generation

- **Model:** `Qwen/Qwen3-0.6B` (default, compact local SLM).
- **Grounding Strategy:** Formats up to top-3 retrieved snippets into a concise context block (~3000 characters maximum).
- **Prompt Engineering:** System instructions mandate that the model respond strictly using facts from the provided sources and state inability to answer if the context is insufficient.
- **Post-Processing:** Automatically strips internal thinking tokens (`<think>...</think>`) and enforces deterministic, reproducible answers (`do_sample=False`).

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

Evaluated with the official exam **Moulinette** against the ground-truth benchmark datasets:

| Benchmark Dataset | Target Threshold | Achieved Recall@5 | Status |
| :--- | :---: | :---: | :---: |
| **Documentation Questions** | $\ge 80.0\%$ | **83.0%** | ✅ PASSED |
| **Codebase Questions** | $\ge 50.0\%$ | **79.8%** | ✅ PASSED |

- **Indexing Throughput:** 2605 files (41,915 chunks) fully ingested and indexed. The mandatory lexical indexing pipeline is designed to satisfy the 5-minute indexing requirement.
- **Retrieval Latency:** 100 questions searched in ($\le 90$ seconds for 200 questions limit).

---

## Bonus Features

### 1. Semantic embeddings

Dense-vector retrieval using `sentence-transformers/all-MiniLM-L6-v2` through the Transformers API.
```bash
uv run python -m src index_semantic
```

### 2. Hybrid retrieval

Combines BM25 and semantic rankings using Reciprocal Rank Fusion.
```bash
uv run python -m src search_hybrid "..."
```

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
110 passed
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
```
```text
osh@42% make index
uv run python -m src index --max_chunk_size 2000 --raw_dir data/raw --output_dir data/processed
Chunking: 100%|███████████████████████████| 2605/2605 [00:07<00:00, 356.14file/s]
Tokenizing: 100%|████████████████████| 41915/41915 [00:02<00:00, 15727.10chunk/s]
Ingestion complete! Indexed 41915 chunks under data/processed/
```

```bash
# Search
make search QUERY="What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?"
```
```text
osh@42% make search QUERY="What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?"
uv run python -m src search "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 5 --index_dir data/processed
data/raw/vllm-0.10.1/docs/features/lora.md [4595:5208]
data/raw/vllm-0.10.1/docs/features/lora.md [5969:6726]
data/raw/vllm-0.10.1/vllm/plugins/lora_resolvers/README.md [0:799]
data/raw/vllm-0.10.1/docs/features/lora.md [3963:4745]
data/raw/vllm-0.10.1/docs/features/lora.md [3318:4113]
```

```bash
# Generate an answer
make answer QUERY="What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?"
```
```text
osh@42% make answer QUERY="What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?"
uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 5 --index_dir data/processed
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████| 311/311 [02:16<00:00,  2.28it/s]

--- Retrieved Sources ---
- data/raw/vllm-0.10.1/docs/features/lora.md [4595:5208]
- data/raw/vllm-0.10.1/docs/features/lora.md [5969:6726]
- data/raw/vllm-0.10.1/vllm/plugins/lora_resolvers/README.md [0:799]
- data/raw/vllm-0.10.1/docs/features/lora.md [3963:4745]
- data/raw/vllm-0.10.1/docs/features/lora.md [3318:4113]

--- Answer ---
The HTTP endpoint used to dynamically load a LoRA adapter in vLLM is http://localhost:8000/v1/load_lora_adapter.
```

```bash
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

### Core Academic References & Technical Topics

1. **Retrieval-Augmented Generation (RAG):**
  - *Reference:* 
  - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).
  - [Wiki: Retrieval-augmented generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation).
  - *Key Insight:* Parametric model memory is static and prone to hallucinations. RAG decouples knowledge storage from generation by dynamically conditioning an LLM on external, verifiable ground-truth evidence.

2. **The BM25 Retrieval Function:**
  - *Key Insight:* Unlike raw TF-IDF, BM25 introduces asymptotic term frequency saturation ($k_1$) and document length normalization ($b$), preventing long source files from unfairly dominating relevance ranking. [Okapi BM25 Ranking](https://en.wikipedia.org/wiki/Okapi_BM25).

3. **Reciprocal Rank Fusion (RRF):**
  - *Key Insight:* Combining dense cosine similarity scores with unbounded BM25 scores directly is problematic due to scale differences. RRF combines rankings using a position-based reciprocal formula ($1 / (k + \text{rank})$) that consistently outperforms score normalization.  [How to score results form multiple retrieval methods in RAG](https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a). 

4. **Dense Sentence Embeddings:**
  - *Reference:* Reimers, N., & Gurevych, I. (2019). [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084). *EMNLP 2019*.
  - *Model:* [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
  - *Key Insight:* Captures semantic intent and paraphrasing by projecting text chunks into a 384-dimensional dense metric space where cosine similarity models conceptual equivalence.

5. **Structural Code Parsing via AST:**
  - *Reference:* [Python Standard Library: `ast` — Abstract Syntax Trees](https://docs.python.org/3/library/ast.html).
  - *Key Insight:* Code possesses hierarchical syntax rather than simple sentence boundaries. Extracting top-level function and class definitions via AST prevents arbitrary mid-statement slicing and preserves syntactic integrity.

6. **Information Retrieval Evaluation:**
  - *Reference:* Manning, C. D., Raghavan, P., & Schütze, H. (2008). [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/). *Cambridge University Press*.
  - *Key Insight:* Recall@k measures the fraction of ground-truth citations found within top-k predictions. Applying Intersection over Union (IoU) over character coordinates guarantees that returned spans cover the exact target region.

- [FastAPI Framework](https://fastapi.tiangolo.com/)
- [Pydantic Validation](https://docs.pydantic.dev/)
- [Qwen Model Family](https://huggingface.co/Qwen)
- [The Python Fire](https://google.github.io/python-fire/guide)
- [The pickle — Python object serialization](https://docs.python.org/3.10/library/pickle.html)

---

### AI Usage Disclosure

In compliance with the **42 Curriculum AI Guidelines (Chapter III & VIII)**, AI tools were utilized ethically and systematically:

- **Tasks Supported by AI:**
  - **Architecture Review:** Discussing SOLID boundary separation between chunkers, indexers, retrievers, and generators.
  - **Regular Expression Tuning:** Crafting and refining regex patterns for camelCase and snake_case subword tokenization.
  - **Type Annotations & PEP 257:** Formatting Google-style docstring templates and troubleshooting mypy type constraints.
  - **Debugging Mismatches:** Identifying corpus path prefix discrepancies and tuning chunk overlap parameters to satisfy the $IoU \ge 0.05$ threshold.
- **Verification:** All algorithms, formulas (IoU, BM25, RRF), and data models were independently audited, manually tested, and validated against the official `moulinette` binary and strict linter rules (`flake8`, `mypy --strict`).

All modules and evaluation scripts were tested, debugged, and verified manually.
