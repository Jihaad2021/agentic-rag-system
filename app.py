"""
Streamlit Web Interface for Agentic RAG System
Phase 1 Day 4 - Basic UI
"""

import streamlit as st
import os
from pathlib import Path
from datetime import datetime

# Import our RAG components
from rag_poc import (
    PDFLoader,
    TextChunker,
    Embedder,
    SimpleVectorStore,
    AnswerGenerator
)

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
    
    # RAG components (initialized once)
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
        st.session_state.vector_store = None
        st.session_state.embedder = None
        st.session_state.generator = None
    
    # Processing status
    if 'processing' not in st.session_state:
        st.session_state.processing = False


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
        
        # File uploader
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to analyze",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Process Document", type="primary"):
                process_uploaded_file(uploaded_file)
        
        st.divider()
        
        # Document list
        st.subheader("📄 Uploaded Documents")
        
        if st.session_state.documents:
            for i, doc in enumerate(st.session_state.documents):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.text(f"📄 {doc['name']}")
                    st.caption(f"{doc['pages']} pages • {doc['chunks']} chunks")
                
                with col2:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete document"):
                        delete_document(i)
                        st.rerun()
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        # System status
        st.subheader("⚙️ System Status")
        
        if st.session_state.rag_initialized:
            st.success("✅ RAG System Ready")
        else:
            st.warning("⏳ Upload a document to start")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


def process_uploaded_file(uploaded_file):
    """Process uploaded PDF file."""
    
    try:
        st.session_state.processing = True
        
        # Create upload directory if not exists
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Show processing status
        with st.spinner("Processing document..."):
            
            # Initialize RAG components if not done
            if not st.session_state.rag_initialized:
                st.session_state.embedder = Embedder()
                st.session_state.vector_store = SimpleVectorStore()
                st.session_state.generator = AnswerGenerator()
                st.session_state.rag_initialized = True
            
            # Load PDF
            loader = PDFLoader()
            text = loader.load(str(file_path))
            
            # Chunk text
            chunker = TextChunker(chunk_size=500, chunk_overlap=50)
            chunks = chunker.chunk_text(text)
            
            # Generate embeddings
            chunks = st.session_state.embedder.embed_chunks(chunks)
            
            # Store in vector database
            st.session_state.vector_store.add_chunks(chunks)
            
            # Count pages (simple approximation)
            page_count = len(text) // 2000  # Rough estimate
            
            # Add to documents list
            st.session_state.documents.append({
                'name': uploaded_file.name,
                'path': str(file_path),
                'pages': page_count,
                'chunks': len(chunks),
                'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        st.success(f"✅ Successfully processed: {uploaded_file.name}")
        st.session_state.processing = False
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
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
    """Process user query and generate response."""
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Embed query
                query_embedding = st.session_state.embedder.embed_query(query)
                
                # Search for relevant chunks
                relevant_chunks = st.session_state.vector_store.search(
                    query_embedding, 
                    top_k=5
                )
                
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
                    with st.expander("📚 View Sources"):
                        for citation in result['citations']:
                            st.caption(f"**Source {citation['source_number']}** (Score: {citation['score']:.4f})")
                            st.text(citation['text_preview'])
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "citations": result['citations']
                })
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


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


def main():
    """Main application."""
    
    # Initialize session state
    init_session_state()
    
    # Display header
    display_header()
    
    # Sidebar
    sidebar()
    
    # Main chat interface
    display_chat_interface()
    
    # Footer
    display_footer()


if __name__ == "__main__":
    main()