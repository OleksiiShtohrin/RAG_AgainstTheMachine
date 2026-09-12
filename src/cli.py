"""Command-Line Interface entrypoint exposed via Python Fire."""

import os
import sys
from typing import Any, Dict, List
from tqdm import tqdm

from src.indexing.indexer import CorpusIndexer
from src.retrieval.lexical_retriever import LexicalRetriever
from src.generation.generator import AnswerGenerator
from src.evaluation.metrics import calculate_recall_at_k
from src.models.question import AnsweredQuestion
from src.models.source import MinimalSource
from src.models.results import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.utils.file_io import read_json_file, save_pydantic_to_json
from src.utils.cache import QueryCache


class CLI:
    """RAG Against the Machine command-line interface."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        output_dir: str = "data/processed",
    ) -> None:
        """Ingest codebase and build index.

        Args:
            max_chunk_size: Maximum characters per chunk (<= 2000).
            raw_dir: Path to raw documents folder.
            output_dir: Destination folder for index artifacts.
        """
        try:
            if max_chunk_size <= 0 or max_chunk_size > 2000:
                print(
                    "Error: max_chunk_size must be between 1 and 2000.",
                    file=sys.stderr,
                )
                return

            if not os.path.exists(raw_dir):
                print(
                    f"Error: Raw directory not found: {raw_dir}",
                    file=sys.stderr,
                )
                return

            indexer = CorpusIndexer(max_chunk_size=max_chunk_size)
            index = indexer.index_directory(
                raw_dir=raw_dir,
                output_dir=output_dir,
            )

            print(
                f"Ingestion complete! "
                f"Indexed {len(index.chunks)} chunks under {output_dir}/"
            )
        except Exception as e:
            print(f"An error occurred during indexing: {e}", file=sys.stderr)

    def search(
        self,
        query: str = "",
        k: int = 5,
        index_dir: str = "data/processed",
    ) -> None:
        """Search the index for top-k source locations for a single query.

        Args:
            query: Question text to search.
            k: Number of top results to return.
            index_dir: Directory where index is saved.
        """
        try:
            if not query.strip():
                print("Error: Search query cannot be empty.", file=sys.stderr)
                return

            if k <= 0:
                print("Warning: k <= 0 requested, 0 results returned.")
                return

            index = CorpusIndexer.load_index(index_dir)
            retriever = LexicalRetriever(index)

            cache = QueryCache()

            cached = cache.get(
                "search",
                query,
                k=k,
                index_dir=index_dir,
            )

            if cached is not None:
                sources = [MinimalSource(**item) for item in cached]
            else:
                sources = retriever.retrieve(query=query, k=k)

                cache.set(
                    "search",
                    query,
                    [source.model_dump() for source in sources],
                    k=k,
                    index_dir=index_dir,
                )

            for src in sources:
                loc = (
                    f"[{src.first_character_index}:{src.last_character_index}]"
                )
                print(f"{src.file_path} {loc}")
        except Exception as e:
            print(f"An error occurred during search: {e}", file=sys.stderr)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results/UnansweredQuestions",
        index_dir: str = "data/processed",
    ) -> None:
        """Batch search over questions from a dataset file.

        Args:
            dataset_path: Path to dataset JSON file.
            k: Number of results per question.
            save_directory: Folder to write StudentSearchResults JSON.
            index_dir: Folder containing BM25 index.
        """
        try:
            raw_data = read_json_file(dataset_path)
            if raw_data is None:
                print(
                    f"Error: Could not read dataset file at {dataset_path}",
                    file=sys.stderr,
                )
                return

            questions_data: List[Dict[str, Any]] = (
                raw_data.get("rag_questions", [])
                if isinstance(raw_data, dict)
                else raw_data
            )

            index = CorpusIndexer.load_index(index_dir)
            retriever = LexicalRetriever(index)

            results: List[MinimalSearchResults] = []
            for item in tqdm(
                questions_data, desc="Searching dataset", unit="q"
            ):
                q_id = str(item.get("question_id", ""))
                q_text = str(item.get("question", ""))
                sources = (
                    retriever.retrieve(query=q_text, k=k) if q_text else []
                )
                results.append(
                    MinimalSearchResults(
                        question_id=q_id,
                        question=q_text,
                        retrieved_sources=sources,
                    )
                )

            student_results = StudentSearchResults(search_results=results, k=k)

            filename = os.path.basename(dataset_path)
            dest_file = os.path.join(save_directory, filename)
            save_pydantic_to_json(student_results, dest_file)
            print(f"Saved student_search_results to {dest_file}")
        except Exception as e:
            print(
                f"An error occurred during dataset search: {e}",
                file=sys.stderr,
            )

    def answer(
        self,
        query: str = "",
        k: int = 5,
        index_dir: str = "data/processed",
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Answer a single query using retrieved context and local LLM.

        Args:
            query: Question text.
            k: Number of source chunks for context.
            index_dir: Folder with index.
            model_name: Model identifier.
        """
        try:
            if not query.strip():
                print("Error: Query cannot be empty.", file=sys.stderr)
                return

            if k <= 0:
                print(
                    "Warning: k <= 0 requested, 0 results returned."
                )
                return

            index = CorpusIndexer.load_index(index_dir)
            retriever = LexicalRetriever(index)
            sources = retriever.retrieve(query=query, k=k)

            generator = AnswerGenerator(model_name=model_name)
            answer_text = generator.generate_answer(query, sources)

            print("\n--- Retrieved Sources ---")
            for src in sources:
                loc = (
                    f"[{src.first_character_index}:{src.last_character_index}]"
                )
                print(f"- {src.file_path} {loc}")
            print("\n--- Answer ---")
            print(answer_text)
        except Exception as e:
            print(
                f"An error occurred during answer generation: {e}",
                file=sys.stderr,
            )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = (
            "data/output/search_results_and_answer/UnansweredQuestions"
        ),
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Generate answers for a batch of search results.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            save_directory: Folder to write StudentSearchResultsAndAnswer JSON.
            model_name: Model identifier.
        """
        try:
            raw_data = read_json_file(student_search_results_path)
            if raw_data is None:
                print(
                    f"Error: Invalid file at {student_search_results_path}",
                    file=sys.stderr,
                )
                return

            search_results_obj = StudentSearchResults.model_validate(raw_data)
            generator = AnswerGenerator(model_name=model_name)

            answered_items: List[MinimalAnswer] = []
            for item in tqdm(
                search_results_obj.search_results,
                desc="Generating answers",
                unit="q",
            ):
                ans = generator.generate_answer(
                    item.question, item.retrieved_sources
                )
                answered_items.append(
                    MinimalAnswer(
                        question_id=item.question_id,
                        question=item.question,
                        retrieved_sources=item.retrieved_sources,
                        answer=ans,
                    )
                )

            output_obj = StudentSearchResultsAndAnswer(
                search_results=answered_items,
                k=search_results_obj.k,
            )

            filename = os.path.basename(student_search_results_path)
            dest_file = os.path.join(save_directory, filename)
            save_pydantic_to_json(output_obj, dest_file)
            print(f"Saved student_search_results_and_answer to {dest_file}")
        except Exception as e:
            print(
                f"An error occurred during dataset answer generation: {e}",
                file=sys.stderr,
            )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Evaluate retrieval performance (Recall@k) against ground-truth.

        Args:
            student_search_results_path: Path to student search results JSON.
            dataset_path: Path to ground-truth AnsweredQuestions JSON.
        """
        try:
            student_raw = read_json_file(student_search_results_path)
            gt_raw = read_json_file(dataset_path)

            if student_raw is None or gt_raw is None:
                print("Error: Could not load files.", file=sys.stderr)
                return

            student_obj = StudentSearchResults.model_validate(student_raw)
            gt_items = (
                gt_raw.get("rag_questions", [])
                if isinstance(gt_raw, dict)
                else gt_raw
            )
            ground_truth_questions: List[AnsweredQuestion] = []
            for item in gt_items:
                try:
                    ground_truth_questions.append(
                        AnsweredQuestion.model_validate(item)
                    )
                except Exception:
                    continue

            k_eval = [1, 3, 5, 10]
            scores = calculate_recall_at_k(
                ground_truth_questions=ground_truth_questions,
                student_results=student_obj.search_results,
                k_values=k_eval,
            )

            print("Evaluation Results")
            print("========================================")
            print(
                f"Recall@1: {scores.get(1, 0.0):.3f}  "
                f"Recall@3: {scores.get(3, 0.0):.3f}  "
                f"Recall@5: {scores.get(5, 0.0):.3f}  "
                f"Recall@10: {scores.get(10, 0.0):.3f}"
            )
        except Exception as e:
            print(f"An error occurred during evaluation: {e}", file=sys.stderr)

    def index_semantic(
        self,
        output_dir: str = "data/processed",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """Build and save semantic vector index (Bonus 1).

        Args:
            output_dir: Folder containing BM25 index and saving vector index.
            model_name: Identifier for sentence embedding model.
        """
        from src.indexing.semantic_index import SemanticIndex

        bm25_index = CorpusIndexer.load_index(output_dir)
        print(f"Generating embeddings for {len(bm25_index.chunks)} chunks...")
        sem_index = SemanticIndex.build(
            bm25_index.chunks, model_name=model_name
        )
        sem_index.save(os.path.join(output_dir, "semantic_index.pkl"))
        print("Semantic vector index successfully saved!")

    def index_incremental(
        self,
        raw_dir: str = "data/raw",
        output_dir: str = "data/processed",
    ) -> None:
        """Run incremental delta indexing for changed files (Bonus 3).

        Args:
            raw_dir: Root path to raw documents folder.
            output_dir: Destination path for index and manifest.
        """
        from src.indexing.incremental_indexer import IncrementalIndexer

        indexer = IncrementalIndexer()
        indexer.update_index(raw_dir=raw_dir, output_dir=output_dir)

    def search_hybrid(
        self,
        query: str,
        k: int = 5,
        output_dir: str = "data/processed",
    ) -> None:
        """Search using Hybrid BM25 + Semantic RRF (Bonus 2).

        Args:
            query: The user search query string.
            k: Number of top fused candidates to return.
            output_dir: Folder containing both BM25 and vector indices.
        """
        from src.indexing.semantic_index import SemanticIndex
        from src.retrieval.hybrid_retriever import HybridRetriever

        bm25_idx = CorpusIndexer.load_index(output_dir)
        sem_idx = SemanticIndex.load(
            os.path.join(output_dir, "semantic_index.pkl")
        )
        retriever = HybridRetriever(bm25_idx, sem_idx)

        cache = QueryCache()

        cached = cache.get(
            "hybrid_search",
            query,
            k=k,
            output_dir=output_dir,
        )

        if cached is not None:
            sources = [MinimalSource(**item) for item in cached]
        else:
            sources = retriever.retrieve(query, k=k)
            cache.set(
                "hybrid_search",
                query,
                [source.model_dump() for source in sources],
                k=k,
                output_dir=output_dir,
            )

        for s in sources:
            loc = f"[{s.first_character_index}:{s.last_character_index}]"
            print(f"{s.file_path} {loc}")

    def search_hybrid_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = (
            "data/output/search_results_hybrid/UnansweredQuestions"
        ),
        output_dir: str = "data/processed",
    ) -> None:
        """Batch hybrid search over a dataset (Bonus 2).

        Args:
            dataset_path: Path to dataset JSON file.
            k: Number of results requested per question.
            save_directory: Directory to save student hybrid search results.
            output_dir: Directory where index files are located.
        """
        try:
            raw_data = read_json_file(dataset_path)
            if raw_data is None:
                print(
                    f"Error: Could not read dataset file at {dataset_path}",
                    file=sys.stderr,
                )
                return

            questions_data: List[Dict[str, Any]] = (
                raw_data.get("rag_questions", [])
                if isinstance(raw_data, dict)
                else raw_data
            )

            from src.indexing.semantic_index import SemanticIndex
            from src.retrieval.hybrid_retriever import HybridRetriever

            bm25_index = CorpusIndexer.load_index(output_dir)
            semantic_index = SemanticIndex.load(
                os.path.join(output_dir, "semantic_index.pkl")
            )
            retriever = HybridRetriever(
                bm25_index,
                semantic_index,
            )

            results: List[MinimalSearchResults] = []

            for item in tqdm(
                questions_data,
                desc="Hybrid search dataset",
                unit="q",
            ):
                q_id = str(item.get("question_id", ""))
                q_text = str(item.get("question", ""))

                sources = (
                    retriever.retrieve(query=q_text, k=k)
                    if q_text
                    else []
                )

                results.append(
                    MinimalSearchResults(
                        question_id=q_id,
                        question=q_text,
                        retrieved_sources=sources,
                    )
                )

            student_results = StudentSearchResults(
                search_results=results,
                k=k,
            )

            filename = os.path.basename(dataset_path)
            dest_file = os.path.join(
                save_directory,
                filename,
            )

            save_pydantic_to_json(
                student_results,
                dest_file,
            )

            print(
                f"Saved hybrid student_search_results to {dest_file}"
            )

        except Exception as e:
            print(
                f"An error occurred during hybrid dataset search: {e}",
                file=sys.stderr,
            )

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Start local HTTP REST API server (Bonus 5).

        Args:
            host: Binding IP host interface.
            port: Binding TCP listening port.
        """
        import uvicorn

        print(f"Starting RAG API at http://{host}:{port}")
        uvicorn.run("src.server:app", host=host, port=port, reload=False)
