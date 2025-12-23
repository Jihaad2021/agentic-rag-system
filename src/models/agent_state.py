"""
Agent State Model - Shared state between all agents.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class Strategy(str, Enum):
    """Query execution strategies"""
    SIMPLE = "simple"
    MULTIHOP = "multihop"
    GRAPH = "graph"


class Chunk(BaseModel):
    """Document chunk with metadata"""
    text: str = Field(..., description="Chunk text content")
    doc_id: str = Field(..., description="Source document ID")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    embedding: Optional[List[float]] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('score')
    def validate_score(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("Score must be between 0.0 and 1.0")
        return v
    
    class Config:
        arbitrary_types_allowed = True


class AgentState(BaseModel):
    """Shared state passed between agents"""
    
    # Input
    query: str
    
    # Planner outputs
    complexity: Optional[float] = Field(None, ge=0.0, le=1.0)
    strategy: Optional[Strategy] = None
    
    # Retrieval outputs
    chunks: List[Chunk] = Field(default_factory=list)
    retrieval_round: int = 0
    
    # Validator outputs
    validation_status: Optional[str] = None
    validation_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Generator outputs
    answer: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging"""
        return {
            "query": self.query,
            "complexity": self.complexity,
            "strategy": self.strategy,
            "num_chunks": len(self.chunks),
            "retrieval_round": self.retrieval_round,
            "validation_status": self.validation_status,
            "has_answer": self.answer is not None,
        }
    
    def add_chunk(self, chunk: Chunk) -> None:
        """Add chunk to state"""
        self.chunks.append(chunk)
    
    def get_top_chunks(self, k: int = 5) -> List[Chunk]:
        """Get top-k chunks by score"""
        sorted_chunks = sorted(
            self.chunks,
            key=lambda c: c.score or 0.0,
            reverse=True
        )
        return sorted_chunks[:k]