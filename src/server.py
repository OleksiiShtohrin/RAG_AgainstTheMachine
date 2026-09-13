"""Local HTTP REST API server exposing RAG functionality."""

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.indexing.indexer import CorpusIndexer
from src.models.source import MinimalSource
from src.retrieval.lexical_retriever import LexicalRetriever
from src.generation.generator import AnswerGenerator

app: Any = FastAPI(
    title="RAG Against the Machine API",
    description="Local HTTP REST API for Codebase RAG",
    version="1.0.0",
)

# Lazy singletons
_retriever: Optional[LexicalRetriever] = None
_generator: Optional[AnswerGenerator] = None


def get_retriever(index_dir: str = "data/processed") -> LexicalRetriever:
    """Get or lazily load lexical retriever singleton with graceful checks.

    Args:
        index_dir: Directory path containing the BM25 index artifact.

    Returns:
        Loaded LexicalRetriever instance.

    Raises:
        HTTPException: If index is not found (503) or cannot be loaded (500).
    """
    global _retriever
    if _retriever is None:
        index_file = os.path.join(index_dir, "bm25_index.pkl")
        if not os.path.exists(index_file):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"BM25 index not found at '{index_file}'. "
                    "Please run indexing first ('make index')."
                ),
            )
        try:
            index = CorpusIndexer.load_index(index_dir)
            _retriever = LexicalRetriever(index)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load index: {exc}",
            ) from exc
    return _retriever


def get_generator() -> AnswerGenerator:
    """Get or lazily load answer generator singleton with error handling.

    Returns:
        Loaded AnswerGenerator instance.

    Raises:
        HTTPException: If model weights cannot be initialized (500).
    """
    global _generator
    if _generator is None:
        try:
            _generator = AnswerGenerator()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize language model: {exc}",
            ) from exc
    return _generator


class SearchRequest(BaseModel):
    """Search request model.

    Attributes:
        query: Search question or prompt string.
        k: Maximum number of source results to return.
    """

    query: str
    k: int = 5


class SearchResponse(BaseModel):
    """Search response model.

    Attributes:
        query: Echoed search query string.
        k: Requested candidate count.
        results: Retrieved top-k MinimalSource locations.
    """

    query: str
    k: int
    results: List[MinimalSource]


class AnswerRequest(BaseModel):
    """Answer request model.

    Attributes:
        query: Search question string.
        k: Number of retrieved source snippets for context.
    """

    query: str
    k: int = 5


class AnswerResponse(BaseModel):
    """Answer response model.

    Attributes:
        query: Echoed question string.
        answer: Generated natural language answer.
        sources: Retrieved source citations used as context.
    """

    query: str
    answer: str
    sources: List[MinimalSource]


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        Service status dictionary indicating API availability.
    """
    return {"status": "ok", "service": "rag-api"}


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest) -> SearchResponse:
    """Retrieve top-k sources for a query with graceful input validation.

    Args:
        req: SearchRequest containing query and k.

    Returns:
        SearchResponse with retrieved source citations.

    Raises:
        HTTPException: If query is empty, k <= 0, or index is missing.
    """
    if not req.query.strip():
        raise HTTPException(
            status_code=400, detail="Query cannot be empty."
        )
    if req.k <= 0:
        raise HTTPException(
            status_code=400,
            detail="Parameter 'k' must be a positive integer.",
        )

    retriever = get_retriever()
    try:
        sources = retriever.retrieve(req.query, k=req.k)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Search retrieval error: {exc}",
        ) from exc

    return SearchResponse(query=req.query, k=req.k, results=sources)


@app.post("/answer", response_model=AnswerResponse)
def answer_endpoint(req: AnswerRequest) -> AnswerResponse:
    """Generate grounded answer with context citations and graceful checks.

    Args:
        req: AnswerRequest containing question and k.

    Returns:
        AnswerResponse with generated text and cited sources.

    Raises:
        HTTPException: If query is empty, k <= 0, or generation fails.
    """
    if not req.query.strip():
        raise HTTPException(
            status_code=400, detail="Query cannot be empty."
        )
    if req.k <= 0:
        raise HTTPException(
            status_code=400,
            detail="Parameter 'k' must be a positive integer.",
        )

    retriever = get_retriever()
    generator = get_generator()

    try:
        sources = retriever.retrieve(req.query, k=req.k)
        answer_text = generator.generate_answer(req.query, sources)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation error: {exc}",
        ) from exc

    return AnswerResponse(
        query=req.query, answer=answer_text, sources=sources
    )
