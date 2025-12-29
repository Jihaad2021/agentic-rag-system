"""
Query Decomposer Agent - Tactical Level 2
Breaks complex queries into manageable sub-questions for multi-hop reasoning.
"""

from typing import List, Dict, Any
from langchain_anthropic import ChatAnthropic

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState
from src.config import get_settings
from src.utils.exceptions import AgenticRAGException


class QueryDecomposerError(AgenticRAGException):
    """Error during query decomposition."""
    pass


class QueryDecomposer(BaseAgent):
    """
    Query Decomposer - Multi-hop reasoning specialist.
    
    Breaks complex queries into sequential sub-questions.
    Each sub-question can be answered independently,
    then combined for final comprehensive answer.
    
    Example:
        Query: "Compare X and Y in terms of A, B, and C"
        Sub-queries:
        1. "What is X's approach to A?"
        2. "What is Y's approach to A?"
        3. "What is X's approach to B?"
        ...
    """
    
    def __init__(self, llm: ChatAnthropic = None):
        super().__init__(name="query_decomposer", version="1.0.0")
        
        settings = get_settings()
        
        if llm is None:
            self.llm = ChatAnthropic(
                model=settings.llm_model,
                temperature=0.0,  # Deterministic
                max_tokens=2000,
                api_key=settings.anthropic_api_key
            )
        else:
            self.llm = llm
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Decompose complex query into sub-questions.
        
        Args:
            state: Current state with query
        
        Returns:
            Updated state with sub_queries list
        """
        query = state.query
        
        # Check if decomposition needed
        if not self._needs_decomposition(query):
            self.log("Query is simple, no decomposition needed", level="info")
            state.sub_queries = [query]  # Keep original
            return state
        
        self.log(f"Decomposing query: {query[:50]}...", level="info")
        
        # Decompose
        sub_queries = self._decompose(query)
        
        state.sub_queries = sub_queries
        state.metadata["decomposition"] = {
            "original_query": query,
            "sub_query_count": len(sub_queries),
            "sub_queries": sub_queries
        }
        
        self.log(f"Generated {len(sub_queries)} sub-queries", level="info")
        
        return state
    
    def _needs_decomposition(self, query: str) -> bool:
        """Check if query is complex enough for decomposition."""
        # Simple heuristics
        indicators = [
            "compare" in query.lower(),
            "contrast" in query.lower(),
            "difference" in query.lower(),
            "and" in query.lower() and len(query.split()) > 10,
            "both" in query.lower(),
            "versus" in query.lower(),
            "vs" in query.lower(),
        ]
        
        return any(indicators)
    
    def _decompose(self, query: str) -> List[str]:
        """Decompose query using LLM."""
        
        prompt = f"""You are a query decomposition expert. Break down complex questions into simpler sub-questions.

Original Query: {query}

Instructions:
1. Identify the main aspects or dimensions of the query
2. Create 3-6 focused sub-questions that:
   - Can be answered independently
   - Together cover the original query
   - Are specific and clear
3. Order them logically

Respond with ONLY a numbered list of sub-questions, nothing else.

Sub-questions:"""
        
        try:
            response = self.llm.invoke(prompt)
            text = response.content
            
            # Parse numbered list
            sub_queries = self._parse_sub_queries(text)
            
            return sub_queries
            
        except Exception as e:
            self.log(f"Decomposition failed: {e}", level="error")
            # Fallback: return original query
            return [query]
    
    def _parse_sub_queries(self, text: str) -> List[str]:
        """Parse LLM response into list of sub-queries."""
        import re
        
        # Find numbered lines
        lines = text.strip().split('\n')
        sub_queries = []
        
        for line in lines:
            # Match patterns like "1. Question" or "1) Question"
            match = re.match(r'^\s*\d+[\.\)]\s*(.+)$', line)
            if match:
                sub_queries.append(match.group(1).strip())
        
        # Fallback: if no numbered items, split by newlines
        if not sub_queries:
            sub_queries = [l.strip() for l in lines if l.strip()]
        
        return sub_queries