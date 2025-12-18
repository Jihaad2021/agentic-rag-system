"""
Streamlit Web Interface for Agentic RAG System
Phase 1 Day 10 - ChromaDB Persistent Storage
"""

import streamlit as st
import os
from pathlib import Path
from datetime import datetime

# Import RAG components
from rag_poc import (
    DocumentLoader,
    Embedder,
    AnswerGenerator
)

# Import hierarchical components ← NEW
from src.ingestion.hierarchical_chunker import HierarchicalChunker
from src.storage.chroma_store import ChromaVectorStore 

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
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initialize - USE CHROMADB ← UPDATED
        status_text.text("🔧 Initializing RAG components...")
        progress_bar.progress(10)
        
        if not st.session_state.rag_initialized:
            st.session_state.embedder = Embedder()
            
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
                parent.embedding = st.session_state.embedder.embed_query(parent.text)
        
        # Embed children
        for child in child_chunks:
            child.embedding = st.session_state.embedder.embed_query(child.text)
        
        # Step 5: Store
        status_text.text("💾 Storing in vector database...")
        progress_bar.progress(85)
        
        if st.session_state.chunking_mode == 'hierarchical':
            st.session_state.vector_store.add_chunks(parent_chunks, child_chunks)
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
    """Process user query with hierarchical support."""
    
    try:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
        
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                
                # Check documents
                if not st.session_state.documents:
                    error_msg = "❌ No documents uploaded."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    return
                
                if not st.session_state.rag_initialized:
                    error_msg = "❌ System not initialized."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    return
                
                # Embed query
                query_embedding = st.session_state.embedder.embed_query(query)
                
                # Search - HIERARCHICAL OR FLAT ← NEW LOGIC
                if st.session_state.chunking_mode == 'hierarchical':
                    # Use hierarchical search
                    relevant_chunks = st.session_state.vector_store.search(
                        query_embedding, 
                        top_k=5,
                        return_parent=True
                    )
                else:
                    # Use flat search (old way)
                    relevant_chunks = st.session_state.vector_store.search(
                        query_embedding, 
                        top_k=5
                    )
                
                if not relevant_chunks:
                    warning_msg = "⚠️ No relevant information found."
                    st.warning(warning_msg)
                    st.session_state.messages.append({"role": "assistant", "content": warning_msg})
                    return
                
                # Generate answer
                result = st.session_state.generator.generate(
                    query, 
                    relevant_chunks,
                    max_chunks=5
                )
                
                # Display answer
                st.markdown(result['answer'])
                
                # Display citations
                if result['citations']:
                    with st.expander("📚 View Sources", expanded=False):
                        for citation in result['citations']:
                            st.caption(f"**Source {citation['source_number']}** (Relevance: {citation['score']:.2%})")
                            st.text(citation['text_preview'])
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "citations": result['citations']
                })
                
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})


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

def main():
    """Main application."""
    
    # Initialize session state
    init_session_state()
    
    # Display header
    display_header()
    
    # Sidebar
    sidebar()
   
    # Add statistics
    display_statistics()    
   
    # Main chat interface
    display_chat_interface()
    
    # Footer
    display_footer()


if __name__ == "__main__":
    main()