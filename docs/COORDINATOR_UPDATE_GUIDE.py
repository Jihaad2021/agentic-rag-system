"""
Update for RetrievalCoordinator - Add SynthesisAgent Integration

This shows how to integrate SynthesisAgent into existing coordinator.

INSTRUCTIONS:
Add this code to your existing RetrievalCoordinator in:
src/agents/retrieval_coordinator.py
"""

# Add to imports at top of file:
from src.agents.synthesis import SynthesisAgent

# Update __init__ method to include synthesis agent:
def __init__(
    self,
    vector_agent=None,
    keyword_agent=None,
    graph_agent=None,
    top_k: int = None,
    parallel: bool = None,
    use_synthesis: bool = True,  # NEW PARAMETER
    use_reranker: bool = False   # NEW PARAMETER
):
    """
    Initialize Retrieval Coordinator.
    
    Args:
        vector_agent: Vector search agent instance
        keyword_agent: Keyword search agent instance
        graph_agent: Graph search agent instance
        top_k: Number of final chunks to return
        parallel: Enable parallel retrieval
        use_synthesis: Use SynthesisAgent for result fusion (default: True)
        use_reranker: Use Cohere reranking (default: False)
    """
    # ... existing init code ...
    
    # Add SynthesisAgent initialization
    if use_synthesis:
        self.synthesis_agent = SynthesisAgent(
            top_k=self.top_k,
            use_reranker=use_reranker
        )
        self.log("SynthesisAgent enabled for result fusion", level="info")
    else:
        self.synthesis_agent = None
        self.log("SynthesisAgent disabled, using simple deduplication", level="info")


# Update execute method to use SynthesisAgent:
def execute(self, state: AgentState) -> AgentState:
    """
    Execute swarm retrieval with synthesis.
    
    Args:
        state: Current agent state with query
    
    Returns:
        Updated state with synthesized chunks
    """
    try:
        query = state.query
        
        self.log(f"Coordinating retrieval for: {query[:50]}...", level="info")
        
        # Step 1: Spawn swarm and retrieve
        all_results = self._spawn_swarm(query)
        
        # Flatten results
        all_chunks = []
        for agent_results in all_results:
            all_chunks.extend(agent_results)
        
        self.log(f"Swarm retrieved {len(all_chunks)} total chunks", level="debug")
        
        # Step 2: Synthesis or simple deduplication
        if self.synthesis_agent:
            # Use SynthesisAgent for intelligent fusion
            temp_state = AgentState(query=query, chunks=all_chunks)
            synthesized_state = self.synthesis_agent.run(temp_state)
            final_chunks = synthesized_state.chunks
            
            # Copy synthesis metadata
            state.metadata["synthesis"] = synthesized_state.metadata.get("synthesis", {})
            
        else:
            # Simple deduplication + top-k (old method)
            deduplicated = self._deduplicate(all_chunks)
            final_chunks = self._select_top_k(deduplicated, self.top_k)
        
        # Step 3: Update state
        state.chunks = final_chunks
        state.retrieval_round += 1
        
        # Add coordinator metadata
        state.metadata["retrieval_coordinator"] = {
            "total_retrieved": len(all_chunks),
            "final_count": len(final_chunks),
            "round": state.retrieval_round,
            "synthesis_used": self.synthesis_agent is not None
        }
        
        self.log(
            f"Coordination complete: {len(final_chunks)} final chunks "
            f"(round {state.retrieval_round})",
            level="info"
        )
        
        return state
        
    except Exception as e:
        self.log(f"Coordination failed: {str(e)}", level="error")
        raise RetrievalError(
            retrieval_type="coordinator",
            message=f"Coordination failed: {str(e)}",
            details={"query": state.query}
        ) from e


# Keep existing methods (_spawn_swarm, _deduplicate, etc.)
# They're still used as fallback when synthesis is disabled