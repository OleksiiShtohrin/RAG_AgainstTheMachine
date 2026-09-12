"""Local HTTP REST API server exposing RAG functionality."""

from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.indexing.indexer import CorpusIndexer
from src.models.source import MinimalSource
from src.retrieval.lexical_retriever import LexicalRetriever
from src.generation.generator import AnswerGenerator

app = FastAPI(
    title="RAG Against the Machine API",
    description="Local HTTP REST API for Codebase RAG",
    version="1.0.0",
)

# Lazy singletons
_retriever: Optional[LexicalRetriever] = None
_generator: Optional[AnswerGenerator] = None


def get_retriever() -> LexicalRetriever:
    """Get or lazily load lexical retriever singleton.

    Returns:
        Loaded LexicalRetriever instance.
    """
    global _retriever
    if _retriever is None:
        index = CorpusIndexer.load_index("data/processed")
        _retriever = LexicalRetriever(index)
    return _retriever


def get_generator() -> AnswerGenerator:
    """Get or lazily load answer generator singleton.

    Returns:
        Loaded AnswerGenerator instance.
    """
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
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
    """Retrieve top-k sources for a query.

    Args:
        req: SearchRequest containing query and k.

    Returns:
        SearchResponse with retrieved source citations.

    Raises:
        HTTPException: If query string is empty.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    retriever = get_retriever()
    sources = retriever.retrieve(req.query, k=req.k)
    return SearchResponse(query=req.query, k=req.k, results=sources)


@app.post("/answer", response_model=AnswerResponse)
def answer_endpoint(req: AnswerRequest) -> AnswerResponse:
    """Generate grounded answer with context citations.

    Args:
        req: AnswerRequest containing question and k.

    Returns:
        AnswerResponse with generated text and cited sources.

    Raises:
        HTTPException: If query string is empty.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    retriever = get_retriever()
    generator = get_generator()

    sources = retriever.retrieve(req.query, k=req.k)
    answer_text = generator.generate_answer(req.query, sources)

    return AnswerResponse(
        query=req.query, answer=answer_text, sources=sources
    )
