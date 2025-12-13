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

        # Export chat
        if st.session_state.messages:
            st.subheader("💾 Export")
            
            if st.button("📥 Download Chat History"):
                export_chat_history()

        # Sample questions (only show if documents uploaded)
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
                    # Trigger chat with this question
                    st.session_state.sample_query = question
                    st.experimental_rerun()

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
                        st.experimental_rerun()
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
            st.experimental_rerun()

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
    """Process uploaded PDF file with progress tracking."""
    
    try:
        st.session_state.processing = True
        
        # Create upload directory
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initialize
        status_text.text("🔧 Initializing RAG components...")
        progress_bar.progress(10)
        
        if not st.session_state.rag_initialized:
            st.session_state.embedder = Embedder()
            st.session_state.vector_store = SimpleVectorStore()
            st.session_state.generator = AnswerGenerator()
            st.session_state.rag_initialized = True
        
        # Step 2: Load PDF
        status_text.text("📄 Loading PDF...")
        progress_bar.progress(25)
        
        loader = PDFLoader()
        text = loader.load(str(file_path))
        
        # Step 3: Chunk
        status_text.text("✂️ Chunking text...")
        progress_bar.progress(40)
        
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_text(text)
        
        # Step 4: Embeddings
        status_text.text(f"🔢 Generating embeddings ({len(chunks)} chunks)...")
        progress_bar.progress(60)
        
        chunks = st.session_state.embedder.embed_chunks(chunks)
        
        # Step 5: Store
        status_text.text("💾 Storing in vector database...")
        progress_bar.progress(85)
        
        st.session_state.vector_store.add_chunks(chunks)
        
        # Step 6: Complete
        page_count = len(text) // 2000
        
        st.session_state.documents.append({
            'name': uploaded_file.name,
            'path': str(file_path),
            'pages': page_count,
            'chunks': len(chunks),
            'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        st.success(f"✅ Successfully processed: {uploaded_file.name}")
        st.balloons()  # Celebration!
        
        st.session_state.processing = False
        
        # Auto-rerun after 2 seconds
        import time
        time.sleep(2)
        st.experimental_rerun()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
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

def display_statistics():
    """Display system statistics."""
    
    if st.session_state.documents and st.session_state.rag_initialized:
        st.subheader("📊 System Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_chunks = sum(doc['chunks'] for doc in st.session_state.documents)
            st.metric("Total Chunks", total_chunks)
        
        with col2:
            total_pages = sum(doc['pages'] for doc in st.session_state.documents)
            st.metric("Total Pages", total_pages)
        
        with col3:
            st.metric("Queries Asked", len([m for m in st.session_state.messages if m['role'] == 'user']))
        
        with col4:
            if st.session_state.vector_store:
                st.metric("Vectors Stored", len(st.session_state.vector_store.chunks))
        
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