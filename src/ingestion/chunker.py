"""
Document Chunker - Fast hierarchical chunking with LangChain.

Uses LangChain's RecursiveCharacterTextSplitter for efficient splitting.
Much faster than manual character-by-character scanning.
"""

from typing import List, Dict, Any, Optional
import hashlib
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    Fast hierarchical document chunker using LangChain.
    
    Creates parent chunks (large context) and child chunks (retrieval units).
    Uses RecursiveCharacterTextSplitter for efficient, smart splitting.
    
    Strategy:
    - Parent chunks: 2000 chars with 200 overlap
    - Child chunks: 500 chars with 50 overlap
    - Smart splitting at sentence/paragraph boundaries
    
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
        Initialize chunker with LangChain splitters.
        
        Args:
            chunk_size: Child chunk size in characters (default from config)
            chunk_overlap: Overlap between child chunks (default from config)
            parent_chunk_size: Parent chunk size in characters (default from config)
        """
        self.logger = setup_logger("chunker")
        settings = get_settings()
        
        # Convert token sizes to character approximation (4 chars ≈ 1 token)
        self.chunk_size = (chunk_size or settings.chunk_size) * 4
        self.chunk_overlap = (chunk_overlap or settings.chunk_overlap) * 4
        self.parent_chunk_size = (parent_chunk_size or settings.parent_chunk_size) * 4
        self.parent_overlap = int(self.parent_chunk_size * 0.1)
        
        # Initialize LangChain splitters
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Try paragraphs, sentences, words
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.logger.info(
            f"Initialized LangChain chunker: "
            f"child={self.chunk_size}±{self.chunk_overlap}, "
            f"parent={self.parent_chunk_size}±{self.parent_overlap}"
        )
    
    def chunk(self, document) -> List[Chunk]:
        """
        Chunk document hierarchically using LangChain.
        
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
        
        # Step 1: Create parent chunks (fast with LangChain)
        parent_texts = self.parent_splitter.split_text(document.text)
        
        self.logger.debug(f"Created {len(parent_texts)} parent text chunks")
        
        # Create parent chunk objects
        current_pos = 0
        parent_chunks = []
        
        for i, parent_text in enumerate(parent_texts):
            # Find position in original text
            start_pos = document.text.find(parent_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(parent_text)
            
            chunk_id = self._generate_chunk_id(document.doc_id, f"parent_{i}")
            
            chunk = Chunk(
                text=parent_text,
                doc_id=document.doc_id,
                chunk_id=chunk_id,
                parent_id=None,
                metadata={
                    **document.metadata,
                    "chunk_type": "parent",
                    "chunk_index": i
                },
                start_char=start_pos,
                end_char=end_pos
            )
            
            parent_chunks.append(chunk)
            all_chunks.append(chunk)
            current_pos = end_pos
        
        # Step 2: Create child chunks within each parent (fast with LangChain)
        for parent_idx, parent in enumerate(parent_chunks):
            child_texts = self.child_splitter.split_text(parent.text)
            
            current_pos = 0
            for child_idx, child_text in enumerate(child_texts):
                # Find position in parent text
                start_pos = parent.text.find(child_text, current_pos)
                if start_pos == -1:
                    start_pos = current_pos
                end_pos = start_pos + len(child_text)
                
                chunk_id = self._generate_chunk_id(
                    document.doc_id,
                    f"{parent.chunk_id}_child_{child_idx}"
                )
                
                chunk = Chunk(
                    text=child_text,
                    doc_id=document.doc_id,
                    chunk_id=chunk_id,
                    parent_id=parent.chunk_id,
                    metadata={
                        **document.metadata,
                        "chunk_type": "child",
                        "chunk_index": child_idx,
                        "parent_index": parent_idx
                    },
                    start_char=parent.start_char + start_pos,
                    end_char=parent.start_char + end_pos
                )
                
                all_chunks.append(chunk)
                current_pos = end_pos
        
        child_count = len([c for c in all_chunks if c.parent_id is not None])
        
        self.logger.info(
            f"Created {len(parent_chunks)} parent chunks, "
            f"{child_count} child chunks (total: {len(all_chunks)})"
        )
        
        return all_chunks
    
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