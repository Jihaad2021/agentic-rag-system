"""
Keyword Search Agent - Operational Level 3 Agent.

Performs keyword-based search using BM25 algorithm.
This is a MOCK implementation for testing the coordinator.

Real implementation will use:
- BM25 algorithm for ranking
- Inverted index for fast lookup
- Elasticsearch or custom implementation
"""

from typing import List

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.utils.exceptions import RetrievalError


class KeywordSearchAgent(BaseAgent):
    """
    Keyword Search Agent - BM25 retrieval (MOCK).
    
    Performs keyword-based search using exact term matching
    and BM25 ranking algorithm.
    
    NOTE: This is a mock implementation that returns dummy chunks.
    Real implementation will use BM25 with inverted index.
    
    Attributes:
        top_k: Number of chunks to retrieve
        mock_mode: If True, returns dummy chunks (default: True)
        
    Example:
        >>> agent = KeywordSearchAgent(top_k=10)
        >>> state = AgentState(query="Python programming")
        >>> result = agent.run(state)
        >>> print(len(result.chunks))  # 10
    """
    
    def __init__(self, top_k: int = 10, mock_mode: bool = True):
        """
        Initialize Keyword Search Agent.
        
        Args:
            top_k: Number of chunks to retrieve
            mock_mode: Use mock data (default: True)
        
        Example:
            >>> agent = KeywordSearchAgent(top_k=5)
        """
        super().__init__(name="keyword_search", version="1.0.0")
        
        self.top_k = top_k
        self.mock_mode = mock_mode
        
        self.log(
            f"Initialized in {'MOCK' if mock_mode else 'REAL'} mode with top_k={top_k}",
            level="debug"
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute keyword search.
        
        Args:
            state: Current agent state with query
        
        Returns:
            Updated state with retrieved chunks
        
        Raises:
            RetrievalError: If search fails
        """
        try:
            query = state.query
            
            self.log(f"Performing keyword search for: {query[:50]}...", level="info")
            
            if self.mock_mode:
                chunks = self._mock_search(query)
            else:
                chunks = self._real_search(query)
            
            self.log(f"Retrieved {len(chunks)} chunks via keyword search", level="info")
            
            # Update state
            state.chunks = chunks
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata["source"] = "keyword"
                chunk.metadata["method"] = "bm25"
            
            return state
            
        except Exception as e:
            self.log(f"Keyword search failed: {str(e)}", level="error")
            raise RetrievalError(
                retrieval_type="keyword",
                message=f"Keyword search failed: {str(e)}",
                details={"query": state.query}
            ) from e
    
    def _mock_search(self, query: str) -> List[Chunk]:
        """
        Mock keyword search returning dummy chunks.
        
        Simulates BM25 by looking for exact keyword matches.
        
        Args:
            query: User query string
        
        Returns:
            List of mock chunks
        """
        chunks = []
        
        # Extract keywords from query
        keywords = query.lower().split()
        
        # Mock chunk templates with keyword emphasis
        query_lower = query.lower()
        
        if "python" in query_lower:
            templates = [
                "Python programming language documentation: Python is versatile and Python is easy to learn.",
                "The official Python website provides Python tutorials and Python resources for developers.",
                "Python software foundation maintains Python core development and Python community standards.",
                "Learn Python basics: Python syntax, Python data types, and Python functions.",
                "Python applications span from Python web frameworks to Python data analysis tools.",
            ]
        elif "machine learning" in query_lower:
            templates = [
                "Machine learning algorithms: machine learning models use machine learning techniques for predictions.",
                "Introduction to machine learning: supervised machine learning and unsupervised machine learning methods.",
                "Machine learning frameworks like TensorFlow enable efficient machine learning development.",
                "Machine learning applications in industry leverage machine learning for automation.",
                "Deep machine learning with neural networks advances machine learning capabilities.",
            ]
        else:
            # Generic keyword-rich templates
            first_keyword = keywords[0] if keywords else "topic"
            templates = [
                f"Comprehensive guide to {first_keyword}: {first_keyword} basics and {first_keyword} applications.",
                f"Understanding {first_keyword}: {first_keyword} fundamentals explained with {first_keyword} examples.",
                f"Advanced {first_keyword} topics: {first_keyword} techniques and {first_keyword} best practices.",
                f"The complete {first_keyword} reference: {first_keyword} documentation and {first_keyword} tutorials.",
                f"Practical {first_keyword} guide: {first_keyword} implementation and {first_keyword} usage patterns.",
            ]
        
        # Create chunks with scores based on keyword frequency
        for i in range(min(self.top_k, len(templates))):
            # Score based on keyword matches (mock)
            keyword_count = sum(kw in templates[i].lower() for kw in keywords)
            score = min(0.85 - (i * 0.06), 0.85)  # Slightly lower than vector
            
            chunk = Chunk(
                text=templates[i],
                doc_id=f"mock_doc_{i // 2 + 10}",  # Different doc IDs from vector
                chunk_id=f"keyword_chunk_{i}",
                score=score,
                metadata={"keyword_matches": keyword_count}
            )
            chunks.append(chunk)
        
        # Generate more if needed
        while len(chunks) < self.top_k:
            i = len(chunks)
            chunk = Chunk(
                text=f"Document mentioning {query} with relevant keywords.",
                doc_id=f"mock_doc_{i // 2 + 10}",
                chunk_id=f"keyword_chunk_{i}",
                score=max(0.4, 0.85 - (i * 0.06)),
                metadata={"keyword_matches": 1}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _real_search(self, query: str) -> List[Chunk]:
        """
        Real keyword search implementation.
        
        TODO: Implement in Week 4
        - Build inverted index
        - Apply BM25 ranking
        - Return top-k by BM25 score
        
        Args:
            query: User query string
        
        Returns:
            List of chunks from BM25 search
        """
        raise NotImplementedError(
            "Real keyword search not implemented yet. "
            "Will be implemented in Week 4."
        )