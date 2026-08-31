"""Corpus ingestion module exports."""

from src.ingestion.corpus_reader import CorpusReader
from src.ingestion.document import Document

__all__ = [
    "CorpusReader",
    "Document",
]
