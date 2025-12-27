"""
Document Chunker - Split documents into chunks with hierarchy.

Implements hierarchical chunking:
- Parent chunks: Large context (2000 tokens)
- Child chunks: Retrieval units (500 tokens)
"""

from typing import List, Dict, Any, Optional
import hashlib
from dataclasses import dataclass

from src.config import get_settings
from src.utils.logger import setup_logger


@dataclass
class Chunk:
    """
    Document chunk with metadata.
    
    Attributes:
        text: Chunk text content
        doc_id: Source document ID
        chunk_id: Unique chunk identifier
        parent_id: Parent chunk ID (for hierarchical chunking)
        metadata: Additional metadata
        start_char: Start character position in document
        end_char: End character position in document
    
    Example:
        >>> chunk = Chunk(
        ...     text="Chunk content here",
        ...     doc_id="doc123",
        ...     chunk_id="chunk1",
        ...     metadata={"page": 1}
        ... )
    """
    text: str
    doc_id: str
    chunk_id: str
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    start_char: int = 0
    end_char: int = 0
    
    def __post_init__(self):
        """Initialize default metadata."""
        if self.metadata is None:
            self.metadata = {}
    
    def __len__(self) -> int:
        """Get chunk text length."""
        return len(self.text)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Chunk(chunk_id={self.chunk_id}, length={len(self.text)})"


class DocumentChunker:
    """
    Hierarchical document chunker.
    
    Creates parent chunks (large context) and child chunks (retrieval units).
    
    Strategy:
    - Parent chunks: 2000 tokens with 200 overlap
    - Child chunks: 500 tokens with 50 overlap
    - Each child knows its parent for context retrieval
    
    Example:
        >>> from src.ingestion.document_loader import Document
        >>> chunker = DocumentChunker()
        >>> doc = Document(text="Long document text here...", metadata={})
        >>> chunks = chunker.chunk(doc)
        >>> print(f"Created {len(chunks)} chunks")
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        parent_chunk_size: Optional[int] = None
    ):
        """
        Initialize chunker with configuration.
        
        Args:
            chunk_size: Child chunk size in tokens (default from config)
            chunk_overlap: Overlap between child chunks (default from config)
            parent_chunk_size: Parent chunk size in tokens (default from config)
        """
        self.logger = setup_logger("chunker")
        settings = get_settings()
        
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.parent_chunk_size = parent_chunk_size or settings.parent_chunk_size
        
        # Calculate parent overlap (10% of parent size)
        self.parent_overlap = int(self.parent_chunk_size * 0.1)
        
        self.logger.info(
            f"Initialized chunker: "
            f"child={self.chunk_size}±{self.chunk_overlap}, "
            f"parent={self.parent_chunk_size}±{self.parent_overlap}"
        )
    
    def chunk(self, document) -> List[Chunk]:
        """
        Chunk document hierarchically.
        
        Creates parent chunks first, then child chunks within each parent.
        
        Args:
            document: Document object with text and metadata
        
        Returns:
            List of Chunk objects (both parent and child chunks)
        
        Example:
            >>> chunks = chunker.chunk(document)
            >>> parent_chunks = [c for c in chunks if c.parent_id is None]
            >>> child_chunks = [c for c in chunks if c.parent_id is not None]
        """
        self.logger.info(
            f"Chunking document {document.doc_id[:8]}... "
            f"({len(document.text)} chars)"
        )
        
        all_chunks = []
        
        # Step 1: Create parent chunks
        parent_chunks = self._create_parent_chunks(document)
        all_chunks.extend(parent_chunks)
        
        # Step 2: Create child chunks within each parent
        for parent in parent_chunks:
            child_chunks = self._create_child_chunks(parent, document)
            all_chunks.extend(child_chunks)
        
        self.logger.info(
            f"Created {len(parent_chunks)} parent chunks, "
            f"{len(all_chunks) - len(parent_chunks)} child chunks"
        )
        
        return all_chunks
    
    def _create_parent_chunks(self, document) -> List[Chunk]:
        """
        Create parent chunks from document.
        
        Args:
            document: Document object
        
        Returns:
            List of parent Chunk objects
        """
        text = document.text
        
        # Simple character-based chunking (approximation for tokens)
        # In production, use tiktoken for accurate token counting
        char_per_token = 4  # Rough approximation
        parent_chunk_chars = self.parent_chunk_size * char_per_token
        parent_overlap_chars = self.parent_overlap * char_per_token
        
        chunks = []
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = min(start + parent_chunk_chars, len(text))
            
            # Find sentence boundary for natural split
            if end < len(text):
                # Look for sentence ending near the boundary
                sentence_ends = [". ", ".\n", "! ", "?\n", "? "]
                best_end = end
                
                # Search backwards for sentence boundary (up to 100 chars)
                for i in range(end, max(end - 100, start), -1):
                    for sent_end in sentence_ends:
                        if text[i:i+len(sent_end)] == sent_end:
                            best_end = i + len(sent_end)
                            break
                    if best_end != end:
                        break
                
                end = best_end
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_id = self._generate_chunk_id(
                    document.doc_id,
                    f"parent_{chunk_num}"
                )
                
                chunk = Chunk(
                    text=chunk_text,
                    doc_id=document.doc_id,
                    chunk_id=chunk_id,
                    parent_id=None,  # This is a parent chunk
                    metadata={
                        **document.metadata,
                        "chunk_type": "parent",
                        "chunk_index": chunk_num
                    },
                    start_char=start,
                    end_char=end
                )
                
                chunks.append(chunk)
                chunk_num += 1
            
            # Move to next chunk with overlap
            start = end - parent_overlap_chars
        
        return chunks
    
    def _create_child_chunks(self, parent: Chunk, document) -> List[Chunk]:
        """
        Create child chunks within a parent chunk.
        
        Args:
            parent: Parent Chunk object
            document: Original document
        
        Returns:
            List of child Chunk objects
        """
        text = parent.text
        
        # Character-based chunking for child chunks
        char_per_token = 4
        chunk_chars = self.chunk_size * char_per_token
        overlap_chars = self.chunk_overlap * char_per_token
        
        chunks = []
        start = 0
        child_num = 0
        
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            
            # Find sentence boundary
            if end < len(text):
                sentence_ends = [". ", ".\n", "! ", "?\n", "? "]
                best_end = end
                
                for i in range(end, max(end - 50, start), -1):
                    for sent_end in sentence_ends:
                        if text[i:i+len(sent_end)] == sent_end:
                            best_end = i + len(sent_end)
                            break
                    if best_end != end:
                        break
                
                end = best_end
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_id = self._generate_chunk_id(
                    document.doc_id,
                    f"{parent.chunk_id}_child_{child_num}"
                )
                
                chunk = Chunk(
                    text=chunk_text,
                    doc_id=document.doc_id,
                    chunk_id=chunk_id,
                    parent_id=parent.chunk_id,  # Link to parent
                    metadata={
                        **document.metadata,
                        "chunk_type": "child",
                        "chunk_index": child_num,
                        "parent_index": parent.metadata.get("chunk_index")
                    },
                    start_char=parent.start_char + start,
                    end_char=parent.start_char + end
                )
                
                chunks.append(chunk)
                child_num += 1
            
            # Move to next chunk with overlap
            start = end - overlap_chars
        
        return chunks
    
    def _generate_chunk_id(self, doc_id: str, chunk_key: str) -> str:
        """
        Generate unique chunk ID.
        
        Args:
            doc_id: Document ID
            chunk_key: Unique key for this chunk
        
        Returns:
            Chunk ID (MD5 hash)
        """
        combined = f"{doc_id}_{chunk_key}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_child_chunks_only(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Filter to get only child chunks (for retrieval).
        
        Args:
            chunks: List of all chunks
        
        Returns:
            List of child chunks only
        
        Example:
            >>> all_chunks = chunker.chunk(document)
            >>> retrieval_chunks = chunker.get_child_chunks_only(all_chunks)
        """
        return [c for c in chunks if c.parent_id is not None]
    
    def get_parent_chunks_only(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Filter to get only parent chunks.
        
        Args:
            chunks: List of all chunks
        
        Returns:
            List of parent chunks only
        """
        return [c for c in chunks if c.parent_id is None]
    
    def get_chunk_with_parent(
        self,
        chunk_id: str,
        all_chunks: List[Chunk]
    ) -> Dict[str, Chunk]:
        """
        Get chunk and its parent for context.
        
        Args:
            chunk_id: ID of child chunk
            all_chunks: List of all chunks
        
        Returns:
            Dictionary with 'chunk' and 'parent' keys
        
        Example:
            >>> result = chunker.get_chunk_with_parent(chunk_id, chunks)
            >>> print(result['chunk'].text)  # Child text
            >>> print(result['parent'].text)  # Parent context
        """
        chunk = next((c for c in all_chunks if c.chunk_id == chunk_id), None)
        
        if not chunk:
            return {"chunk": None, "parent": None}
        
        parent = None
        if chunk.parent_id:
            parent = next(
                (c for c in all_chunks if c.chunk_id == chunk.parent_id),
                None
            )
        
        return {"chunk": chunk, "parent": parent}