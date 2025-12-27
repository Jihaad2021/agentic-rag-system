# Examples

## Available Examples

### `test_planner_usage.py`
Demonstrates Planner Agent usage.

**Run:**
```bash
python examples/test_planner_usage.py
```

### `test_workflow_usage.py`
Demonstrates complete LangGraph workflow with all agents.

**Features:**
- Complete pipeline: Planner → Coordinator → Validator
- Automatic retry logic
- Execution tracing
- Multiple query types

**Run:**
```bash
python examples/test_workflow_usage.py
```

**Requirements:**
- Valid `ANTHROPIC_API_KEY` in `.env`
- All agents initialized
- LangGraph installed

**Output:**
```
Agentic RAG Workflow Demo
======================================================================

Step 1: Initializing components...
✓ LLM: claude-3-5-sonnet-20241022
✓ Planner Agent initialized
✓ Validator Agent initialized
✓ Retrieval Agents initialized (MOCK mode)

Query 1: What is Python?
Strategy: SIMPLE
Complexity: 0.245
Chunks Retrieved: 10
Validation Status: PROCEED
```

## Demos Included

1. **Main Demo** - Basic workflow execution
2. **Retry Scenario** - Demonstrates retry loop
3. **Strategy Selection** - Shows different complexity handling