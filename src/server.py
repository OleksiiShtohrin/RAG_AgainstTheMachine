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
    """Get or load retriever."""
    global _retriever
    if _retriever is None:
        index = CorpusIndexer.load_index("data/processed")
        _retriever = LexicalRetriever(index)
    return _retriever


def get_generator() -> AnswerGenerator:
    """Get or load answer generator."""
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator


class SearchRequest(BaseModel):
    """Search request model."""

    query: str
    k: int = 5


class SearchResponse(BaseModel):
    """Search response model."""

    query: str
    k: int
    results: List[MinimalSource]


class AnswerRequest(BaseModel):
    """Answer request model."""

    query: str
    k: int = 5


class AnswerResponse(BaseModel):
    """Answer response model."""

    query: str
    answer: str
    sources: List[MinimalSource]


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "rag-api"}


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest) -> SearchResponse:
    """Retrieve top-k sources for a query."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    retriever = get_retriever()
    sources = retriever.retrieve(req.query, k=req.k)
    return SearchResponse(query=req.query, k=req.k, results=sources)


@app.post("/answer", response_model=AnswerResponse)
def answer_endpoint(req: AnswerRequest) -> AnswerResponse:
    """Generate grounded answer with context citations."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    retriever = get_retriever()
    generator = get_generator()

    sources = retriever.retrieve(req.query, k=req.k)
    answer_text = generator.generate_answer(req.query, sources)

    return AnswerResponse(
        query=req.query, answer=answer_text, sources=sources
    )
