"""
Vector Store - Store and retrieve document chunks using ChromaDB.

Manages vector database for semantic search.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings
from src.ingestion.chunker import Chunk
from src.utils.logger import setup_logger
from src.utils.exceptions import AgenticRAGException


class VectorStoreError(AgenticRAGException):
    """Error during vector store operations."""
    pass


class VectorStore:
    """
    ChromaDB vector store for document chunks.
    
    Features:
    - Store chunks with embeddings
    - Semantic search (cosine similarity)
    - Metadata filtering
    - Collection management
    
    Example:
        >>> store = VectorStore()
        >>> store.add_chunks(chunks, embeddings)
        >>> results = store.search(query_embedding, top_k=5)
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of ChromaDB collection (default from config)
            persist_directory: Directory for persistence (default from config)
        """
        self.logger = setup_logger("vector_store")
        settings = get_settings()
        
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity
            )
            
            self.logger.info(
                f"Initialized vector store: {self.collection_name} "
                f"({self.collection.count()} chunks)"
            )
            
        except Exception as e:
            raise VectorStoreError(
                message=f"Failed to initialize vector store: {str(e)}",
                details={"error": str(e)}
            ) from e
    
    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]]
    ) -> None:
        """
        Add chunks with embeddings to vector store.
        
        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors (same order as chunks)
        
        Raises:
            VectorStoreError: If adding chunks fails
        
        Example:
            >>> chunks = [chunk1, chunk2, chunk3]
            >>> embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
            >>> store.add_chunks(chunks, embeddings)
        """
        if not chunks:
            self.logger.warning("No chunks to add")
            return
        
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                message="Chunks and embeddings length mismatch",
                details={
                    "chunks_count": len(chunks),
                    "embeddings_count": len(embeddings)
                }
            )
        
        self.logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        try:
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = [self._prepare_metadata(chunk) for chunk in chunks]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            self.logger.info(
                f"Successfully added {len(chunks)} chunks "
                f"(total: {self.collection.count()})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add chunks: {str(e)}")
            raise VectorStoreError(
                message=f"Failed to add chunks to vector store: {str(e)}",
                details={"chunk_count": len(chunks), "error": str(e)}
            ) from e
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
        
        Returns:
            List of search results with chunks and scores
        
        Example:
            >>> results = store.search(query_embedding, top_k=5)
            >>> for result in results:
            ...     print(result['chunk_id'], result['score'])
        """
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'chunk_id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'score': self._distance_to_score(results['distances'][0][i])
                    })
            
            self.logger.debug(f"Search returned {len(formatted_results)} results")
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise VectorStoreError(
                message=f"Vector search failed: {str(e)}",
                details={"top_k": top_k, "error": str(e)}
            ) from e
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
        
        Returns:
            Chunk data or None if not found
        
        Example:
            >>> chunk = store.get_chunk_by_id("chunk_abc123")
            >>> print(chunk['text'])
        """
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if result and result['ids']:
                return {
                    'chunk_id': result['ids'][0],
                    'text': result['documents'][0],
                    'metadata': result['metadatas'][0],
                    'embedding': result['embeddings'][0] if result['embeddings'] else None
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get chunk {chunk_id}: {str(e)}")
            return None
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Number of chunks deleted
        
        Example:
            >>> deleted_count = store.delete_by_doc_id("doc_123")
            >>> print(f"Deleted {deleted_count} chunks")
        """
        try:
            # Get chunks for this document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["documents"]
            )
            
            if not results or not results['ids']:
                self.logger.info(f"No chunks found for doc_id: {doc_id}")
                return 0
            
            chunk_ids = results['ids']
            
            # Delete chunks
            self.collection.delete(ids=chunk_ids)
            
            self.logger.info(
                f"Deleted {len(chunk_ids)} chunks for doc_id: {doc_id}"
            )
            
            return len(chunk_ids)
            
        except Exception as e:
            self.logger.error(f"Failed to delete chunks for {doc_id}: {str(e)}")
            raise VectorStoreError(
                message=f"Failed to delete chunks: {str(e)}",
                details={"doc_id": doc_id, "error": str(e)}
            ) from e
    
    def count(self) -> int:
        """
        Get total number of chunks in store.
        
        Returns:
            Number of chunks
        
        Example:
            >>> total = store.count()
            >>> print(f"Total chunks: {total}")
        """
        return self.collection.count()
    
    def reset(self) -> None:
        """
        Delete all chunks from store.
        
        Warning: This is irreversible!
        
        Example:
            >>> store.reset()  # Delete everything
        """
        try:
            # Delete collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate empty collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.logger.warning("Vector store reset - all chunks deleted")
            
        except Exception as e:
            raise VectorStoreError(
                message=f"Failed to reset vector store: {str(e)}",
                details={"error": str(e)}
            ) from e
    
    def _prepare_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Prepare metadata for ChromaDB storage.
        
        ChromaDB only accepts certain types in metadata.
        
        Args:
            chunk: Chunk object
        
        Returns:
            Metadata dictionary
        """
        metadata = {
            "doc_id": chunk.doc_id,
            "chunk_type": chunk.metadata.get("chunk_type", "child"),
            "start_char": chunk.start_char,
            "end_char": chunk.end_char
        }
        
        # Add parent_id if exists
        if chunk.parent_id:
            metadata["parent_id"] = chunk.parent_id
        
        # Add optional metadata (only simple types)
        for key in ["filename", "file_type", "source", "chunk_index"]:
            if key in chunk.metadata:
                value = chunk.metadata[key]
                # ChromaDB accepts: str, int, float, bool
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
        
        return metadata
    
    def _distance_to_score(self, distance: float) -> float:
        """
        Convert cosine distance to similarity score.
        
        ChromaDB returns cosine distance (0 = identical, 2 = opposite).
        Convert to score (1.0 = identical, 0.0 = opposite).
        
        Args:
            distance: Cosine distance from ChromaDB
        
        Returns:
            Similarity score (0.0-1.0)
        """
        # Cosine distance to similarity: similarity = 1 - (distance / 2)
        return max(0.0, min(1.0, 1.0 - (distance / 2.0)))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Statistics dictionary
        
        Example:
            >>> stats = store.get_stats()
            >>> print(stats['total_chunks'])
        """
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.count(),
            "persist_directory": self.persist_directory
        }