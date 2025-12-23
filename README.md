# 🤖 Agentic RAG System

> **Hierarchical Multi-Agent RAG System** with Self-Reflection, GraphRAG, and Adaptive Reasoning

A production-ready Retrieval-Augmented Generation system where AI agents autonomously decide how, when, and whether to retrieve information based on query complexity.

---

## ✅ Week 1 Complete - Traditional RAG Foundation

**Status:** Phase 1 Complete (Day 6/14)  
**Progress:** 43% of Phase 1  
**Accuracy:** ~65% (baseline established)

### 🎉 Achievements

- ✅ Multi-format document support (PDF, DOCX, TXT)
- ✅ Intelligent text chunking (500 tokens, 50 overlap)
- ✅ Voyage AI embeddings (1536 dimensions)
- ✅ Vector similarity search
- ✅ Claude answer generation with citations
- ✅ Streamlit web interface
- ✅ Comprehensive test suite (8 tests, 100% pass rate)
- ✅ Performance benchmarks (<5s query time)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- API Keys: [Anthropic](https://console.anthropic.com/), [Voyage AI](https://dashboard.voyageai.com/)

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/agentic-rag-system.git
cd agentic-rag-system

# Create virtual environment
python3.11 -m venv env_rag2
source env_rag2/bin/activate  # Windows: env_rag2\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys

# Run web interface
streamlit run app.py
```

---

## 📊 Current Features

### Document Processing
- **Multi-format:** PDF, DOCX, TXT
- **Smart chunking:** Hierarchical with overlap
- **Fast processing:** <2s per document

### RAG Pipeline
- **Embeddings:** Voyage AI (1536d vectors)
- **Search:** Cosine similarity
- **Generation:** Claude Haiku
- **Citations:** Automatic source attribution

### Web Interface
- **Upload:** Drag & drop files
- **Chat:** Interactive Q&A
- **Preview:** Document content viewer
- **Export:** Chat history download

---

## 🧪 Testing
```bash
# Run automated tests
python tests/test_rag_system.py

# Run benchmarks
python tests/benchmark.py

# Test coverage: 100% (8/8 tests passing)
```

---

## 📈 Performance Metrics

| Metric | Current | Target (Week 2) |
|--------|---------|-----------------|
| **Accuracy** | 65% | 67% |
| **Query Time** | 4-5s | <8s |
| **Formats** | 3 | 3 |
| **Test Pass Rate** | 100% | 100% |

---

## 🎯 Roadmap

### ✅ Week 1 (Complete)
- Traditional RAG baseline
- Multi-format support
- Web interface
- Testing framework

### 🔜 Week 2 (Next)
- Hierarchical chunking (parent/child)
- PostgreSQL metadata storage
- ChromaDB persistent storage
- Enhanced evaluation

### 📅 Future Phases
- **Phase 2:** Multi-Agent Core
- **Phase 3:** Self-Reflection
- **Phase 4:** Agent Debate
- **Phase 5:** GraphRAG
- **Phase 6:** Learning & Optimization

---

## 📁 Project Structure
```
agentic-rag-system/
├── app.py                  # Streamlit web interface
├── rag_poc.py              # Core RAG components
├── test_api.py             # API connection tests
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── data/
│   ├── uploads/            # User documents
│   └── chroma_db/          # Vector storage
├── tests/
│   ├── test_rag_system.py  # Automated tests
│   ├── test_dataset.json   # Test queries
│   └── benchmark.py        # Performance tests
└── docs/
    └── DAILY_PROGRESS.md   # Development log
```

---

## 🛠️ Tech Stack

- **LLM:** Claude 3 Haiku (Anthropic)
- **Embeddings:** Voyage AI Large-2
- **Framework:** LangChain
- **Vector DB:** In-memory (→ ChromaDB in Week 2)
- **Backend:** Python 3.11
- **Frontend:** Streamlit
- **Testing:** pytest

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 🤝 Contributing

This is a personal learning project. Feedback welcome!

---

**Status:** 🚧 Week 1 Complete - Production Baseline Ready  
**Next:** Week 2 - Enhanced Storage & Evaluation

---

## 🤖 Agent System

### Architecture

The system uses **11 specialized agents** organized in 3 levels:

**Level 1 (Strategic):**
- Planner Agent - Analyzes query complexity and selects strategy

**Level 2 (Tactical):**
- Query Decomposer - Breaks complex queries into sub-questions
- Retrieval Coordinator - Manages parallel retrieval swarm
- Validator - Quality control for retrieved content
- Synthesis - Deduplicates and ranks results
- Writer - Generates final answer
- Critic - Reviews and approves/regenerates

**Level 3 (Operational):**
- Vector Search Agent - Semantic search
- Keyword Search Agent - BM25 exact matching
- Graph Search Agent - Relationship-based retrieval

### Creating Custom Agents
```python
from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="my_agent", version="1.0.0")
    
    def execute(self, state: AgentState) -> AgentState:
        # Your logic here
        state.my_field = "processed"
        return state

# Usage
agent = MyAgent()
result = agent.run(state)
metrics = agent.get_metrics()
```

See `docs/AGENT_ARCHITECTURE.md` for detailed patterns and best practices.

_Building intelligent document Q&A, one week at a time._ 🚀