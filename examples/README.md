# Examples

This directory contains example usage scripts for the Agentic RAG system.

## Available Examples

### `test_planner_usage.py`

Demonstrates Planner Agent usage:
- Query complexity analysis
- Strategy selection
- Feature extraction breakdown
- Custom threshold configuration
- Query comparison

**Run:**
```bash
python examples/test_planner_usage.py
```

**Requirements:**
- Valid `ANTHROPIC_API_KEY` in `.env`
- Claude API access

**Output:**
```
Planner Agent Demo
==================================================

Query: What is Python?
  Complexity: 0.245
  Strategy: SIMPLE
  Features:
    - Length: 0.12
    - Questions: 0.30
    - Entities: 0.00
    - Relationships: 0.00
```

## Configuration

Examples use settings from `src/config.py` and `.env` file.

Key settings:
- `ANTHROPIC_API_KEY` - Required for LLM calls
- `PLANNER_COMPLEXITY_THRESHOLD_SIMPLE` - Default: 0.3
- `PLANNER_COMPLEXITY_THRESHOLD_MULTIHOP` - Default: 0.7

## Creating New Examples

1. Import required components:
```python
from src.agents.planner import PlannerAgent
from src.models.agent_state import AgentState
from src.config import get_settings
```

2. Initialize agent:
```python
settings = get_settings()
llm = ChatAnthropic(api_key=settings.anthropic_api_key)
planner = PlannerAgent(llm=llm)
```

3. Use agent:
```python
state = AgentState(query="Your query")
result = planner.run(state)
print(result.complexity, result.strategy)
```