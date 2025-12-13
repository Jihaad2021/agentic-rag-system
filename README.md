# 🤖 Agentic RAG System

> **Hierarchical Multi-Agent RAG System** with Self-Reflection, GraphRAG, and Adaptive Reasoning

A production-ready Retrieval-Augmented Generation system where AI agents autonomously decide how, when, and whether to retrieve information based on query complexity.

---

## 🎯 Project Overview

**What makes this "Agentic"?**
- ✅ **Autonomous Decision Making** - Planner Agent analyzes query complexity
- ✅ **Self-Reflection** - Validator & Critic Agents ensure quality
- ✅ **Multi-Agent Collaboration** - 11 agents working hierarchically
- ✅ **Adaptive Behavior** - Different strategy per query type
- ✅ **Continuous Learning** - Fine-tuning and memory systems

**vs Traditional RAG:**
```
Traditional: Query → Retrieve (always same) → Generate → Answer
Agentic:     Query → Planner → Adaptive Strategy → Validate → Generate → Critique → Answer
```

---

## 📊 Project Timeline

**Duration:** 12 weeks (6 phases × 2 weeks)

| Phase | Weeks | Focus | Accuracy Target |
|-------|-------|-------|----------------|
| **Phase 1** | 1-2 | Foundation (Traditional RAG) | 67% |
| **Phase 2** | 3-4 | Multi-Agent Core | 80% |
| **Phase 3** | 5-6 | Self-Reflection | 85% |
| **Phase 4** | 7-8 | Agent Debate | 87% |
| **Phase 5** | 9-10 | GraphRAG | 90% |
| **Phase 6** | 11-12 | Learning & Optimization | 92% |

---

## 🏗️ Architecture

### **Agent Hierarchy:**
```
Level 1 (Strategic):  Planner Agent
                          ↓
Level 2 (Tactical):   Query Decomposer → Retrieval Coordinator 
                      → Validator → Synthesis → Writer → Critic
                          ↓
Level 3 (Operational): [Vector, Keyword, Graph] Agents (Swarm)
```

### **Tech Stack:**

- **LLM:** Claude 3.5 Sonnet (Anthropic)
- **Embeddings:** Voyage AI → Fine-tuned (Phase 6)
- **Framework:** LangChain + LangGraph
- **Vector DB:** ChromaDB
- **Graph DB:** NetworkX / Neo4j
- **RDBMS:** PostgreSQL
- **Cache:** Redis
- **UI:** Streamlit
- **API:** FastAPI
- **Monitoring:** LangSmith + Weights & Biases
- **Evaluation:** RAGAS

---

## 🚀 Current Status

**Phase:** 1 - Foundation  
**Week:** 1  
**Day:** 1  
**Progress:** Setting up project structure

### Completed:
- ✅ Project planning & architecture design
- ✅ GitHub repository setup
- ⏳ Development environment setup

---

## 📁 Project Structure
```
agentic-rag-system/
├── data/
│   ├── uploads/          # User uploaded documents
│   └── chroma_db/        # Vector database storage
├── .gitignore
├── README.md
└── LICENSE
```

---

## 🛠️ Setup (Coming Soon)

Setup instructions will be added as development progresses.

**Prerequisites:**
- Python 3.11+
- Git
- API Keys: Anthropic, Voyage AI

---

## 📖 Documentation

- [Project Overview](docs/PROJECT_OVERVIEW.md) - Coming soon
- [Architecture](docs/ARCHITECTURE.md) - Coming soon
- [Weekly Progress](docs/WEEKLY_PROGRESS.md) - Coming soon

---

## 🎯 Goals

### Technical:
- 92% accuracy on diverse queries
- <2s latency for simple queries
- Self-correcting with 80%+ success rate
- Production-ready deployment

### Portfolio:
- Demonstrate senior-level system design
- Showcase multi-agent orchestration
- Implement cutting-edge research (GraphRAG, self-reflection)
- Create comprehensive documentation

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 🤝 Contributing

This is a personal learning project, but feedback and suggestions are welcome!

---

**Status:** 🚧 Under Active Development  
**Started:** December 2024  
**Expected Completion:** March 2025

---

_Building the future of intelligent document Q&A, one agent at a time._ 🚀