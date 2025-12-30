"""
Streamlit Web Interface for Agentic RAG System
Phase 1 Day 10 - ChromaDB Persistent Storage
"""

import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import numpy as np

# Import RAG components
from rag_poc import (
    DocumentLoader,
    Embedder,
    AnswerGenerator
)

# Import hierarchical components ← NEW
from src.ingestion.hierarchical_chunker import HierarchicalChunker
from src.storage.chroma_store import ChromaVectorStore 
from src.evaluation.simple_evaluator import SimpleEvaluator
from src.ingestion.embedder import EmbeddingGenerator
from src.models.agent_state import Chunk

# Keep old components for comparison
from rag_poc import TextChunker, SimpleVectorStore

# Page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .assistant-message {
        background-color: #F5F5F5;
    }
</style>
""", unsafe_allow_html=True)

st.session_state.embedder = EmbeddingGenerator()

def init_session_state():
    """Initialize session state variables."""
    
    # Chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Documents
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    # RAG components - HIERARCHICAL ← UPDATED
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
        st.session_state.vector_store = None
        st.session_state.embedder = None
        st.session_state.generator = None
        st.session_state.parent_chunks = []  # ← NEW
        st.session_state.child_chunks = []   # ← NEW
    
    # Processing status
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Chunking mode selection ← NEW
    if 'chunking_mode' not in st.session_state:
        st.session_state.chunking_mode = 'hierarchical'  # 'flat' or 'hierarchical'
    
    # RAG components - FORCE REINITIALIZE
    if 'embedder' not in st.session_state or not hasattr(st.session_state.embedder, 'generate'):
        from src.ingestion.embedder import EmbeddingGenerator
        st.session_state.embedder = EmbeddingGenerator()
        print("✅ Embedder reinitialized with EmbeddingGenerator")


def display_header():
    """Display app header."""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<p class="main-header">📚 Agentic RAG System</p>', 
                   unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Intelligent Document Q&A with AI Agents</p>', 
                   unsafe_allow_html=True)
    
    with col2:
        st.metric("Documents", len(st.session_state.documents))
        st.metric("Messages", len(st.session_state.messages))


def sidebar():
    """Render sidebar with document upload and management."""
    
    with st.sidebar:
        st.header("📁 Document Management")
        
        # ============================================
        # CHUNKING MODE SELECTOR (NEW)
        # ============================================
        st.subheader("⚙️ Chunking Mode")
        chunking_mode = st.radio(
            "Select chunking strategy",
            options=['hierarchical', 'flat'],
            format_func=lambda x: {
                'hierarchical': '🔺 Hierarchical (Parent-Child)',
                'flat': '📊 Flat (Single Level)'
            }[x],
            help="""
            **Hierarchical**: Better context, higher accuracy (recommended)
            **Flat**: Simpler, faster processing
            """,
            key='chunking_mode_selector'
        )
        
        st.session_state.chunking_mode = chunking_mode
        
        # Show mode info
        if chunking_mode == 'hierarchical':
            st.info("📈 Parents: 2000 tokens | Children: 500 tokens")
        else:
            st.info("📊 Chunks: 500 tokens")
        
        st.divider()
        
        # ============================================
        # FILE UPLOADER (EXISTING - KEPT)
        # ============================================
        st.subheader("Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, Word (DOCX), or Text (TXT) files",
            label_visibility="collapsed"
        )
        
        # Show file info if uploaded
        if uploaded_file is not None:
            file_size = uploaded_file.size / 1024  # KB
            file_type = uploaded_file.type
            
            st.info(f"""
            📄 **{uploaded_file.name}**
            - Type: {file_type}
            - Size: {file_size:.1f} KB
            """)
            
            if st.button("📤 Process Document", type="primary"):
                process_uploaded_file(uploaded_file)
        
        st.divider()
        
        # ============================================
        # DOCUMENT LIST (UPDATED WITH HIERARCHICAL INFO)
        # ============================================
        st.subheader("📄 Uploaded Documents")
        
        if st.session_state.documents:
            for i, doc in enumerate(st.session_state.documents):
                # Get file icon based on type
                icon = {
                    '.PDF': '📕',
                    '.DOCX': '📘', 
                    '.TXT': '📄'
                }.get(doc.get('type', '.PDF'), '📄')
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.text(f"{icon} {doc['name']}")
                    
                    # Show chunking info (UPDATED)
                    mode = doc.get('chunking_mode', 'flat')
                    if mode == 'hierarchical':
                        st.caption(
                            f"🔺 Hierarchical • "
                            f"{doc.get('parents', 0)} parents • "
                            f"{doc['chunks']} children"
                        )
                    else:
                        st.caption(
                            f"📊 Flat • "
                            f"{doc.get('type', 'PDF')} • "
                            f"{doc['pages']} pages • "
                            f"{doc['chunks']} chunks"
                        )
                
                with col2:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete document"):
                        delete_document(i)
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        # ============================================
        # SAMPLE QUESTIONS (EXISTING - KEPT)
        # ============================================
        if st.session_state.documents:
            st.subheader("💡 Sample Questions")
            
            sample_questions = [
                "What is this document about?",
                "Summarize the main points",
                "What are the key findings?",
                "Give me specific details"
            ]
            
            for question in sample_questions:
                if st.button(f"💬 {question}", key=f"sample_{question}"):
                    st.session_state.sample_query = question
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()
        
        st.divider()
        
        # ============================================
        # EXPORT CHAT (EXISTING - KEPT)
        # ============================================
        if st.session_state.messages:
            st.subheader("💾 Export")
            
            if st.button("📥 Download Chat History"):
                export_chat_history()
        
        st.divider()
        
        # ============================================
        # SYSTEM STATUS (EXISTING - KEPT)
        # ============================================
        st.subheader("⚙️ System Status")
        
        if st.session_state.rag_initialized:
            st.success("✅ RAG System Ready")
            
            # Show ChromaDB stats ← NEW
            try:
                stats = st.session_state.vector_store.get_stats()
                st.caption(f"💾 Vectors in DB: {stats['total_vectors']:,}")
            except:
                pass
        else:
            st.warning("⏳ Upload a document to start")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        
        # Clear database button (NEW - CAREFUL!)
        if st.session_state.rag_initialized:
            st.divider()
            
            with st.expander("⚠️ Advanced Options"):
                st.warning("**Danger Zone**")
                
                if st.button("🗑️ Clear Vector Database", type="secondary"):
                    if st.button("⚠️ Confirm Clear All Vectors?"):
                        try:
                            st.session_state.vector_store.clear_all()
                            st.session_state.documents = []
                            st.session_state.parent_chunks = []
                            st.session_state.child_chunks = []
                            st.success("✅ Database cleared")
                            try:
                                st.rerun()
                            except AttributeError:
                                st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

def export_chat_history():
    """Export chat history as text file."""
    
    chat_text = "# Chat History\n\n"
    chat_text += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    chat_text += "=" * 50 + "\n\n"
    
    for message in st.session_state.messages:
        role = "You" if message['role'] == 'user' else "Assistant"
        chat_text += f"{role}:\n{message['content']}\n\n"
        
        if 'citations' in message and message['citations']:
            chat_text += "Sources:\n"
            for citation in message['citations']:
                chat_text += f"- Source {citation['source_number']}: {citation['text_preview']}\n"
            chat_text += "\n"
        
        chat_text += "-" * 50 + "\n\n"
    
    # Download
    st.download_button(
        label="📄 Download as TXT",
        data=chat_text,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def process_uploaded_file(uploaded_file):
    """Process uploaded document with ChromaDB persistent storage."""
    
    try:
        st.session_state.processing = True

        # Create upload directory
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_ext = file_path.suffix.upper()
        
        # ========== FORCE CLEAN REINIT ==========
        # Always reinitialize to ensure clean state
        st.session_state.embedder = EmbeddingGenerator()
        
        from src.storage.chroma_store import ChromaVectorStore
        
        # Clear if exists
        if st.session_state.get('vector_store'):
            st.session_state.vector_store.clear_all()
            print("🗑️ Cleared old ChromaDB data")
        
        # Fresh vector store
        st.session_state.vector_store = ChromaVectorStore(
            persist_directory="data/chroma_db"
        )
        
        st.session_state.generator = AnswerGenerator()
        st.session_state.documents = []
        st.session_state.messages = []
        st.session_state.rag_initialized = True
        print("✅ Reinitialized with clean state")
        # ========== END REINIT ==========
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initialize - USE CHROMADB ← UPDATED
        status_text.text("🔧 Initializing RAG components...")
        progress_bar.progress(10)
        
        if not st.session_state.rag_initialized:
            st.session_state.embedder = EmbeddingGenerator()
            
            # Use ChromaDB instead of in-memory ← CHANGED
            from src.storage.chroma_store import ChromaVectorStore
            st.session_state.vector_store = ChromaVectorStore(
                persist_directory="data/chroma_db"
            )
            
            st.session_state.generator = AnswerGenerator()
            st.session_state.rag_initialized = True
        
        # Step 2: Load document
        status_text.text(f"📄 Loading {file_ext} file...")
        progress_bar.progress(25)
        
        loader = DocumentLoader()
        text = loader.load(str(file_path))
        
        # Step 3: Chunk - HIERARCHICAL OR FLAT
        status_text.text("✂️ Chunking text...")
        progress_bar.progress(40)
        
        if st.session_state.chunking_mode == 'hierarchical':
            # Hierarchical chunking
            chunker = HierarchicalChunker(
                parent_size=2000,
                child_size=500,
                child_overlap=50
            )
            parent_chunks, child_chunks = chunker.chunk_text(text)
            
            status_text.text(f"✂️ Created {len(parent_chunks)} parents, {len(child_chunks)} children...")
            
        else:
            # Flat chunking
            chunker = TextChunker(chunk_size=500, chunk_overlap=50)
            chunks = chunker.chunk_text(text)
            
            # Convert to Chunk objects for compatibility
            from src.ingestion.hierarchical_chunker import Chunk
            parent_chunks = []
            child_chunks = [
                Chunk(
                    chunk_id=c['chunk_id'],
                    text=c['text'],
                    tokens=c.get('tokens', []),
                    token_count=c['token_count'],
                    start_idx=c['start_idx'],
                    end_idx=c['end_idx'],
                    chunk_type='child'
                )
                for c in chunks
            ]
        
        # Step 4: Embeddings
        total_chunks = len(parent_chunks) + len(child_chunks)
        status_text.text(f"🔢 Generating embeddings ({total_chunks} chunks)...")
        progress_bar.progress(60)
        
        # Embed parents
        if parent_chunks:
            for parent in parent_chunks:
                parent.embedding = st.session_state.embedder.generate([parent.text])[0]
        
        # Embed children
        for child in child_chunks:
            child.embedding = st.session_state.embedder.generate([child.text])[0]
    

        # Step 5: Store
        status_text.text("💾 Storing in vector database...")
        progress_bar.progress(85)
        
        if st.session_state.chunking_mode == 'hierarchical':
            st.session_state.vector_store.add_chunks(
                parent_chunks=parent_chunks,
                child_chunks=child_chunks,
                filename=uploaded_file.name  # ← ADD THIS LINE
            )
            # Store for UI display
            st.session_state.parent_chunks.extend(parent_chunks)
            st.session_state.child_chunks.extend(child_chunks)
        else:
            # For flat mode, use old add_chunks method
            for chunk in child_chunks:
                st.session_state.vector_store.chunks.append(chunk)
        
        # Step 6: Complete
        page_count = loader.count_pages(str(file_path))
        
        st.session_state.documents.append({
            'name': uploaded_file.name,
            'path': str(file_path),
            'type': file_ext,
            'pages': page_count,
            'chunks': len(child_chunks),
            'parents': len(parent_chunks) if parent_chunks else 0,
            'chunking_mode': st.session_state.chunking_mode,
            'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        st.success(f"✅ {file_ext}: {uploaded_file.name} | Mode: {st.session_state.chunking_mode.upper()}")
        st.balloons()
        
        st.session_state.processing = False
        
        import time
        time.sleep(2)
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.session_state.processing = False


def delete_document(index):
    """Delete a document from the list."""
    
    try:
        doc = st.session_state.documents[index]
        
        # Delete file if exists
        if os.path.exists(doc['path']):
            os.remove(doc['path'])
        
        # Remove from list
        st.session_state.documents.pop(index)
        
        st.success(f"Deleted: {doc['name']}")
        
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")


def display_chat_interface():
    """Display main chat interface."""
    
    st.header("💬 Chat with Your Documents")

    # Welcome message if no messages yet
    if not st.session_state.messages and st.session_state.documents:
        st.info("""
        👋 **Welcome!** Your documents are ready. Ask me anything!
        
        Try questions like:
        - "What is this document about?"
        - "Summarize the main points"
        - "What are the key findings?"
        """)
    elif not st.session_state.documents:
        st.warning("""
        📁 **No documents uploaded yet.**
        
        Please upload a PDF document using the sidebar to get started.
        """)

    # Check if there's a sample query to process
    if hasattr(st.session_state, 'sample_query') and st.session_state.sample_query:
        query = st.session_state.sample_query
        st.session_state.sample_query = None  # Clear it
        process_user_query(query)
        st.experimental_rerun()

    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display citations if available
                if "citations" in message and message["citations"]:
                    with st.expander("📚 View Sources"):
                        for citation in message["citations"]:
                            st.caption(f"**Source {citation['source_number']}** (Score: {citation['score']:.4f})")
                            st.text(citation['text_preview'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.rag_initialized:
            st.warning("⚠️ Please upload a document first!")
        else:
            process_user_query(prompt)


def process_user_query(query: str):
    """Process user query and generate response with self-reflection."""
    import time
    start_time = time.time()

    from src.models.agent_state import AgentState, Chunk  # ← ADD THIS LINE
    from src.agents.query_decomposer import QueryDecomposer
    from src.orchestration.multihop_handler import MultiHopHandler

    print("\n" + "="*60)
    print(f"🔍 DEBUG: process_user_query called with: {query}")
    print("="*60)
    
    if not st.session_state.rag_initialized:
        print("❌ DEBUG: RAG not initialized!")
        st.error("Please upload a document first!")
        return
    
    print("✅ DEBUG: RAG initialized")
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })
    print(f"✅ DEBUG: User message added. Total messages: {len(st.session_state.messages)}")
    
    # STEP: Planner Analysis
    with st.spinner("🧠 Analyzing query complexity..."):
        from src.agents.planner import PlannerAgent
        from langchain_anthropic import ChatAnthropic
        from src.config import get_settings

        settings = get_settings()
        llm = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key
        )
        planner = PlannerAgent(llm=llm)

        planner_state = AgentState(query=query)
        planner_state = planner.run(planner_state)
        
        complexity = planner_state.complexity
        strategy = planner_state.strategy
        
        print(f"✅ Planner: Complexity={complexity:.2f}, Strategy={strategy}")
        
        # Show user
        st.info(f"🧠 Complexity: {complexity:.2f} | Strategy: {strategy}")

    # Retrieve chunks
    with st.spinner("🔍 Retrieving relevant chunks..."):
        try:
            print("🔍 DEBUG: Generating embedding...")
            query_embedding = st.session_state.embedder.generate_query_embedding(query)
            print(f"✅ DEBUG: Embedding generated. Length: {len(query_embedding)}")
            
            # NEW: Check if decomposition needed
            print("🔍 DEBUG: Checking if decomposition needed...")
            from src.agents.query_decomposer import QueryDecomposer
            from src.orchestration.multihop_handler import MultiHopHandler
            
            decomposer = QueryDecomposer()
            temp_state = AgentState(query=query)
            temp_state = decomposer.run(temp_state)
            
            if temp_state.sub_queries and len(temp_state.sub_queries) > 1:
                # Multi-hop processing
                print(f"🔀 DEBUG: Multi-hop detected. {len(temp_state.sub_queries)} sub-queries")
                
                st.info(f"🔀 Complex query detected. Decomposed into {len(temp_state.sub_queries)} sub-questions")
                
                with st.expander("👁️ View Sub-questions"):
                    for i, sq in enumerate(temp_state.sub_queries, 1):
                        st.write(f"{i}. {sq}")
                
                # Process all sub-queries
                handler = MultiHopHandler()
                chunks = handler.process_sub_queries(
                    temp_state.sub_queries,
                    st.session_state.vector_store,
                    st.session_state.embedder,
                    top_k=5
                )
                
                # Convert already Chunk objects, no need to convert again
                print(f"✅ DEBUG: Multi-hop complete. {len(chunks)} chunks")
                
            else:
                # Simple query - normal retrieval
                print("🔍 DEBUG: Simple query, normal retrieval...")
                
                search_results = st.session_state.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=10,
                    return_parent=True
                )
                print(f"✅ DEBUG: Search complete. Results: {len(search_results)}")
                
                # Convert to Chunk objects
                from src.models.agent_state import Chunk
                
                chunks = []
                for result in search_results:
                    chunk = Chunk(
                        text=result['text'],
                        doc_id='unknown',
                        chunk_id=result['chunk_id'],
                        score=result['score'],
                        metadata={
                            'filename': result.get('metadata', {}).get('filename', 'uploaded_document'),
                            'chunk_type': result.get('chunk_type', 'child'),
                            **result.get('metadata', {})
                        }
                    )
                    chunks.append(chunk)
                
                print(f"✅ DEBUG: Converted to {len(chunks)} Chunk objects")
                
        except Exception as e:
            print(f"❌ DEBUG: Error in retrieval: {e}")
            import traceback
            traceback.print_exc()
            st.error(f"Error retrieving chunks: {e}")
            return
        
    # Generate answer with self-reflection
    with st.spinner("✍️ Generating answer with self-reflection..."):
        try:
            print("🔍 DEBUG: Importing agents...")
            from src.agents.writer import WriterAgent
            from src.agents.critic import CriticAgent
            from src.agents.self_reflection import SelfReflectionLoop
            from src.models.agent_state import AgentState
            
            print("🔍 DEBUG: Initializing agents...")
            writer = WriterAgent()
            critic = CriticAgent(quality_threshold=0.7)
            loop = SelfReflectionLoop(
                writer=writer,
                critic=critic,
                max_iterations=3
            )
            print("✅ DEBUG: Agents initialized")
            
            print("🔍 DEBUG: Creating AgentState...")
            state = AgentState(query=query, chunks=chunks)
            print(f"✅ DEBUG: State created with {len(state.chunks)} chunks")
            
            print("🔍 DEBUG: Running self-reflection loop...")
            result = loop.run(state)
            print("✅ DEBUG: Self-reflection complete")
            
            print(f"📝 DEBUG: Answer length: {len(result.answer)} chars")
            
            # Extract metadata
            reflection_stats = result.metadata.get("self_reflection", {})
            print(f"📊 DEBUG: Reflection stats: {reflection_stats}")
            
        except Exception as e:
            print(f"❌ DEBUG: Error in generation: {e}")
            import traceback
            traceback.print_exc()
            st.error(f"Error generating answer: {e}")
            return
    
    print("🔍 DEBUG: Preparing citations...")
    # Prepare citations with chunk metadata
    citations = []
    for i, chunk in enumerate(chunks[:5], 1):
        citations.append({
            "source_number": i,
            "filename": chunk.metadata.get('filename', 'unknown'),
            "chunk_type": chunk.metadata.get('chunk_type', 'unknown'),
            "text_preview": chunk.text[:200],
            "score": chunk.score or 0.0
        })
    print(f"✅ DEBUG: {len(citations)} citations prepared")
    
    print("🔍 DEBUG: Adding assistant message...")
    
    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "citations": citations,
        "self_reflection": {
            "iterations": reflection_stats.get("iterations", 0),
            "final_score": reflection_stats.get("final_score", 0.0),
            "improved": reflection_stats.get("improved", False),
            "decision": reflection_stats.get("final_decision", "unknown")
        }
    })
    
    print(f"✅ DEBUG: Assistant message added. Total messages: {len(st.session_state.messages)}")
    print("="*60)
    print("✅ DEBUG: process_user_query COMPLETE")
    print("="*60 + "\n")

    # Calculate latency
    latency = time.time() - start_time
    
    # Track performance
    if 'performance_tracker' not in st.session_state:
        from src.monitoring.performance_tracker import PerformanceTracker
        st.session_state.performance_tracker = PerformanceTracker()
    
    is_multihop = temp_state.sub_queries and len(temp_state.sub_queries) > 1
    
    st.session_state.performance_tracker.track_query(
        query=query,
        latency=latency,
        chunks_retrieved=len(chunks),
        strategy="multi_hop" if is_multihop else "simple",
        iterations=reflection_stats.get("iterations", 0),
        cache_hit=False
    )
    
    print(f"⏱️  Query processed in {latency:.2f}s")

def display_footer():
    """Display footer with info."""
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("🚀 Phase 1 - Week 1 - Day 4")
    
    with col2:
        st.caption("💡 Powered by Claude & Voyage AI")
    
    with col3:
        st.caption("📊 Traditional RAG Baseline")

def display_statistics():
    """Display system statistics with hierarchical info."""
    
    if st.session_state.documents and st.session_state.rag_initialized:
        st.subheader("📊 System Statistics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_chunks = sum(doc['chunks'] for doc in st.session_state.documents)
            st.metric("Total Chunks", total_chunks)
        
        with col2:
            total_parents = sum(doc.get('parents', 0) for doc in st.session_state.documents)
            st.metric("Parent Chunks", total_parents)
        
        with col3:
            queries_count = len([m for m in st.session_state.messages if m['role'] == 'user'])
            st.metric("Queries", queries_count)
        
        with col4:
            # Show mode
            mode = st.session_state.chunking_mode
            mode_icon = "🔺" if mode == 'hierarchical' else "📊"
            st.metric("Mode", f"{mode_icon} {mode.title()}")
        
        with col5:
            # Context size
            if st.session_state.chunking_mode == 'hierarchical':
                context = "2000 tok"
            else:
                context = "500 tok"
            st.metric("Context", context)
        
        st.divider()
def display_evaluation_interface():
    """Display evaluation interface with custom evaluator."""
    
    st.subheader("📊 System Evaluation")
    
    if not st.session_state.documents:
        st.info("Upload documents first to run evaluation")
        return
    
    st.markdown("""
    **Custom Evaluation Metrics:**
    - Citation Rate: Answers include proper citations
    - Context Usage: Retrieved chunks are used in answers
    - Answer Quality: Substantial and complete responses
    - Self-Reflection: Improvement through regeneration
    """)
    
    # Load test questions
    import json
    from pathlib import Path
    
    test_file = Path("data/test_questions.json")
    
    if not test_file.exists():
        st.warning("⚠️ Test questions file not found: data/test_questions.json")
        st.info("Create this file with your test questions first.")
        return
    
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    questions = test_data.get('questions', [])
    
    if not questions:
        st.warning("No questions found in test file")
        return
    
    st.info(f"📝 Loaded {len(questions)} test questions")
    
    # Show sample questions
    with st.expander("👁️ View Test Questions"):
        for i, q in enumerate(questions, 1):
            st.write(f"{i}. {q}")
    
    # Run evaluation button
    if st.button("🚀 Run Evaluation", type="primary"):
        from src.evaluation.simple_evaluator import SimpleEvaluator
        from src.agents.writer import WriterAgent
        from src.agents.critic import CriticAgent
        from src.agents.self_reflection import SelfReflectionLoop
        from src.models.agent_state import AgentState
        
        evaluator = SimpleEvaluator()
        
        # Initialize agents
        writer = WriterAgent()
        critic = CriticAgent(quality_threshold=0.7)
        loop = SelfReflectionLoop(writer, critic, max_iterations=3)
        
        # Process each question
        all_answers = []
        all_chunks_list = []
        all_metadata = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, question in enumerate(questions):
            status_text.text(f"Processing question {i+1}/{len(questions)}...")
            
            # Generate embedding
            query_embedding = st.session_state.embedder.generate_query_embedding(question)
            
            # Search
            search_results = st.session_state.vector_store.search(
                query_embedding=query_embedding,
                top_k=10,
                return_parent=True
            )
            
            # Convert to Chunk objects
            from src.models.agent_state import Chunk
            chunks = []
            for result in search_results:
                chunk = Chunk(
                    text=result['text'],
                    doc_id='unknown',
                    chunk_id=result['chunk_id'],
                    score=result['score'],
                    metadata={'filename': 'uploaded_document'}
                )
                chunks.append(chunk)
            
            # Generate answer with self-reflection
            state = AgentState(query=question, chunks=chunks)
            result = loop.run(state)
            
            # Store results
            all_answers.append(result.answer)
            all_chunks_list.append(chunks)
            all_metadata.append(result.metadata.get('self_reflection', {}))
            
            progress_bar.progress((i + 1) / len(questions))
        
        status_text.empty()
        progress_bar.empty()
        
        # Evaluate
        results = evaluator.evaluate_batch(
            questions, all_answers, all_chunks_list, all_metadata
        )
        
        # Display results
        st.success("✅ Evaluation Complete!")
        
        st.markdown("### 📊 Overall Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score = results['avg_overall']
            color = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
            st.metric("Overall Quality", f"{color} {score:.1%}")
        
        with col2:
            st.metric("Citation Rate", f"{results['avg_citation_rate']:.1%}")
        
        with col3:
            st.metric("Context Usage", f"{results['avg_context_usage']:.1%}")
        
        with col4:
            st.metric("Improvement Rate", f"{results['improvement_rate']:.1%}")
        
        # Additional metrics
        col5, col6 = st.columns(2)
        
        with col5:
            st.metric("Avg Quality Score", f"{results['avg_quality_score']:.1%}")
        
        with col6:
            st.metric("Avg Word Count", f"{results['avg_word_count']:.0f}")
        
        # Detailed results table
        st.markdown("### 📋 Detailed Results")
        
        import pandas as pd
        
        df_data = []
        for score in results['detailed_scores']:
            df_data.append({
                'Question': score['query'][:50] + '...',
                'Overall': f"{score['overall']:.1%}",
                'Citations': '✅' if score['has_citations'] else '❌',
                'Words': score['word_count'],
                'Improved': '✅' if score['was_improved'] else '➖',
                'Iterations': score['iterations']
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

def display_document_preview():
    """Show preview of uploaded documents."""
    
    if st.session_state.documents:
        st.subheader("👁️ Document Preview")
        
        with st.expander("📄 View Document Details", expanded=False):
            # Select document to preview
            doc_names = [doc['name'] for doc in st.session_state.documents]
            selected_doc = st.selectbox(
                "Select document to preview",
                doc_names,
                key="doc_preview_selector"
            )
            
            # Find selected document
            doc = next((d for d in st.session_state.documents if d['name'] == selected_doc), None)
            
            if doc:
                # Document metadata
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Type", doc.get('type', 'PDF'))
                
                with col2:
                    st.metric("Pages", doc.get('pages', 'N/A'))
                
                with col3:
                    st.metric("Chunks", doc['chunks'])
                
                with col4:
                    mode = doc.get('chunking_mode', 'flat')
                    st.metric("Mode", mode.title())
                
                # Show chunking info
                if doc.get('chunking_mode') == 'hierarchical':
                    st.info(f"🔺 Hierarchical: {doc.get('parents', 0)} parents, {doc['chunks']} children")
                else:
                    st.info(f"📊 Flat: {doc['chunks']} chunks")
                
                st.caption(f"📅 Uploaded: {doc.get('uploaded_at', 'Unknown')}")
                
                # Load and show preview (optional - can be slow)
                if st.button("📖 Load Content Preview", key=f"preview_{selected_doc}"):
                    try:
                        from rag_poc import DocumentLoader
                        loader = DocumentLoader()
                        text = loader.load(doc['path'])
                        
                        # Show first 1000 characters
                        preview_text = text[:1000]
                        if len(text) > 1000:
                            preview_text += "..."
                        
                        st.text_area(
                            "Content Preview (first 1000 chars)",
                            preview_text,
                            height=200,
                            key=f"preview_text_{selected_doc}"
                        )
                        
                        # Statistics
                        st.caption(f"Total: {len(text):,} chars | {len(text.split()):,} words")
                        
                    except Exception as e:
                        st.error(f"Error loading preview: {e}")
                        
def display_chat_messages():
    """Display chat message history."""
    
    if not st.session_state.messages:
        st.markdown("""
        ### 👋 Welcome to Agentic RAG System!
        
        Upload a document from the sidebar to get started, then ask questions about it.
        
        **Features:**
        - 🔺 Hierarchical chunking for better context
        - 🔄 Self-reflection (Writer-Critic loop)
        - 📊 Quality evaluation metrics
        - 🎯 Intelligent retrieval
        """)
        return
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show self-reflection info (NEW)
            if message["role"] == "assistant" and "self_reflection" in message:
                reflection = message["self_reflection"]
                
                # Create compact info box
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    iterations = reflection.get("iterations", 0)
                    emoji = "🔄" if iterations > 0 else "✅"
                    st.metric("Iterations", f"{emoji} {iterations}")
                
                with col2:
                    score = reflection.get("final_score", 0.0)
                    color = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                    st.metric("Quality Score", f"{color} {score:.2f}")
                
                with col3:
                    improved = reflection.get("improved", False)
                    st.metric("Improved", "✅ Yes" if improved else "➖ No")
                
                with col4:
                    decision = reflection.get("decision", "unknown")
                    st.metric("Status", decision.upper())
            
            # Show citations with chunk type
            if message["role"] == "assistant" and "citations" in message:
                if message["citations"]:
                    with st.expander("📚 View Sources", expanded=False):
                        for citation in message["citations"]:
                            # Extract chunk info
                            chunk_type = citation.get('chunk_type', 'unknown')
                            chunk_emoji = "📄" if chunk_type == "parent" else "📝"
                            
                            st.caption(
                                f"**[{citation['source_number']}] {citation.get('filename', 'unknown')}** "
                                f"{chunk_emoji} ({chunk_type} chunk, Relevance: {citation['score']:.2%})"
                            )
                            st.text(citation['text_preview'])        

def display_chat_input():
    """Display chat input field (must be outside tabs)."""
    
    # Only show if documents uploaded
    if not st.session_state.documents:
        st.info("👆 Upload a document from the sidebar to start chatting")
        return
    
    # Handle sample query (if triggered from sidebar)
    if 'sample_query' in st.session_state and st.session_state.sample_query:
        query = st.session_state.sample_query
        st.session_state.sample_query = None  # Clear it
        process_user_query(query)
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
        return
    
    # Chat input (must be at root level, not in tabs/columns/expander)
    if prompt := st.chat_input("Ask a question about your documents..."):
        process_user_query(prompt)
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

def main():
    """Main application."""
    
    # Initialize session state
    init_session_state()
    
    # Display header
    display_header()
    
    # Sidebar
    sidebar()
   
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Evaluation", "📈 Statistics", "⚡ Performance"])
    
    with tab1:
        # Display chat history (inside tab OK)
        display_chat_messages()
    
    with tab2:
        # Evaluation interface
        display_evaluation_interface()
    
    with tab3:
        # Statistics and preview
        display_statistics()
        display_document_preview()

    with tab4:
        st.subheader("⚡ Performance Metrics")
        
        if 'performance_tracker' in st.session_state:
            stats = st.session_state.performance_tracker.get_stats()
            
            if stats:
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Queries", stats['total_queries'])
                
                with col2:
                    avg_lat = stats['avg_latency_ms'] / 1000
                    color = "🟢" if avg_lat < 3 else "🟡" if avg_lat < 5 else "🔴"
                    st.metric("Avg Latency", f"{color} {avg_lat:.2f}s")
                
                with col3:
                    st.metric("Cache Hit Rate", f"{stats['cache_hit_rate']:.1%}")
                
                with col4:
                    st.metric("Avg Chunks", f"{stats['avg_chunks']:.1f}")
                
                # Latency breakdown
                st.markdown("### ⏱️ Latency Breakdown")
                
                col5, col6 = st.columns(2)
                
                with col5:
                    st.metric("Min Latency", f"{stats['min_latency_ms']/1000:.2f}s")
                
                with col6:
                    st.metric("Max Latency", f"{stats['max_latency_ms']/1000:.2f}s")
                
                # Session info
                st.info(f"📊 Session Duration: {stats['session_duration_min']:.1f} minutes")
                
                # Save metrics button
                if st.button("💾 Save Metrics"):
                    st.session_state.performance_tracker.save_metrics()
                    st.success("✅ Metrics saved to data/metrics.json")
            
            else:
                st.info("No queries processed yet. Ask some questions to see metrics!")
        
        else:
            st.info("Performance tracking will start after your first query.")

    # Chat input MUST be outside tabs
    display_chat_input()
    
    # Footer
    display_footer()


if __name__ == "__main__":
    main()