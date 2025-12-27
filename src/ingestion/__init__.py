"""
Document ingestion package.

Contains components for loading, chunking, embedding, and storing documents.
"""

from src.ingestion.document_loader import DocumentLoader, Document, DocumentLoadError

__all__ = [
    "DocumentLoader",
    "Document",
    "DocumentLoadError"
]