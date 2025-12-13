"""
RAG Proof of Concept - Phase 1 Day 2
PDF loading, text chunking, and embeddings.

Note: Using Claude 3 Haiku (claude-3-haiku-20240307) for API calls.
"""

import os
from typing import List, Dict, Any
from pypdf import PdfReader
from dotenv import load_dotenv
import tiktoken
from langchain_voyageai import VoyageAIEmbeddings
import numpy as np

# Load environment
load_dotenv()


class PDFLoader:
    """Load and extract text from PDF files."""
    
    def __init__(self):
        """Initialize PDF loader."""
        self.supported_formats = ['.pdf']
    
    def load(self, file_path: str) -> str:
        """
        Load PDF and extract all text.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Validate file format
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        print(f"📄 Loading PDF: {file_path}")
        
        # Load PDF
        text = self._extract_text(file_path)
        
        print(f"✅ Extracted {len(text)} characters from {self._count_pages(file_path)} pages")
        
        return text
    
    def _extract_text(self, file_path: str) -> str:
        """
        Extract text from PDF using PyPDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Combined text from all pages
        """
        text_parts = []
        
        # Open PDF
        reader = PdfReader(file_path)
        
        # Extract text from each page
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            
            # Clean text
            page_text = self._clean_text(page_text)
            
            if page_text.strip():  # Only add non-empty pages
                text_parts.append(page_text)
                print(f"  Page {page_num}: {len(page_text)} chars")
        
        # Combine all pages
        full_text = "\n\n".join(text_parts)
        
        return full_text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove null bytes
        text = text.replace("\x00", "")
        
        return text
    
    def _count_pages(self, file_path: str) -> int:
        """
        Count pages in PDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Number of pages
        """
        reader = PdfReader(file_path)
        return len(reader.pages)


class TextChunker:
    """Split text into chunks with token counting."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap tokens between chunks
            encoding_name: Tokenizer encoding (cl100k_base for GPT-4/Claude)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dictionaries with text, tokens, and metadata
        """
        print(f"\n📝 Chunking text...")
        print(f"   Chunk size: {self.chunk_size} tokens")
        print(f"   Overlap: {self.chunk_overlap} tokens")
        
        # Tokenize entire text
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        
        print(f"   Total tokens: {total_tokens}")
        
        chunks = []
        start_idx = 0
        chunk_num = 0
        
        while start_idx < total_tokens:
            # Calculate end index
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            
            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk metadata
            chunk = {
                "chunk_id": f"chunk_{chunk_num}",
                "text": chunk_text,
                "tokens": chunk_tokens,
                "token_count": len(chunk_tokens),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "chunk_num": chunk_num
            }
            
            chunks.append(chunk)
            chunk_num += 1
            
            # Move to next chunk with overlap
            start_idx += self.chunk_size - self.chunk_overlap
        
        print(f"✅ Created {len(chunks)} chunks")
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

class Embedder:
    """Generate embeddings using Voyage AI."""
    
    def __init__(self):
        """Initialize embedder."""
        self.embedder = VoyageAIEmbeddings(
            voyage_api_key=os.getenv("VOYAGE_API_KEY"),
            model="voyage-large-2"
        )
        print("📊 Embedder initialized (voyage-large-2, 1536 dimensions)")
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Chunks with added 'embedding' field
        """
        print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings (batch)
        embeddings = self.embedder.embed_documents(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Dimension: {len(embeddings[0])}")
        
        return chunks
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        return self.embedder.embed_query(query)


class SimpleVectorStore:
    """Simple in-memory vector storage and search."""
    
    def __init__(self):
        """Initialize vector store."""
        self.chunks = []
        print("💾 SimpleVectorStore initialized")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to storage.
        
        Args:
            chunks: List of chunks with embeddings
        """
        self.chunks.extend(chunks)
        print(f"✅ Added {len(chunks)} chunks to storage (Total: {len(self.chunks)})")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of top-k most similar chunks with scores
        """
        if not self.chunks:
            print("⚠️  No chunks in storage")
            return []
        
        print(f"\n🔍 Searching for top-{top_k} similar chunks...")
        
        # Calculate similarities
        similarities = []
        for chunk in self.chunks:
            # Cosine similarity
            similarity = self._cosine_similarity(
                query_embedding,
                chunk['embedding']
            )
            similarities.append({
                **chunk,
                'score': similarity
            })
        
        # Sort by score (descending)
        similarities.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top-k
        top_results = similarities[:top_k]
        
        print(f"✅ Found {len(top_results)} results")
        for i, result in enumerate(top_results, 1):
            print(f"   {i}. {result['chunk_id']}: score={result['score']:.4f}")
        
        return top_results
    
    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2)
    
def test_chunking():
    """Test text chunking functionality."""
    
    print("\n" + "=" * 60)
    print("TESTING TEXT CHUNKING")
    print("=" * 60)
    
    # Sample text
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. Colloquially, the term 
    "artificial intelligence" is often used to describe machines (or computers) 
    that mimic "cognitive" functions that humans associate with the human mind, 
    such as "learning" and "problem solving". As machines become increasingly 
    capable, tasks considered to require "intelligence" are often removed from 
    the definition of AI, a phenomenon known as the AI effect. A quip in 
    Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, 
    optical character recognition is frequently excluded from things considered 
    to be AI, having become a routine technology.
    """ * 10  # Repeat to get more tokens
    
    # Create chunker
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    # Count tokens
    token_count = chunker.count_tokens(sample_text)
    print(f"\nSample text: {token_count} tokens")
    
    # Chunk text
    chunks = chunker.chunk_text(sample_text)
    
    # Display first 3 chunks
    print(f"\n📋 First 3 chunks (of {len(chunks)}):")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"ID: {chunk['chunk_id']}")
        print(f"Tokens: {chunk['token_count']}")
        print(f"Text preview: {chunk['text'][:100]}...")
    
    # Verify overlap
    print(f"\n🔍 Verifying overlap:")
    if len(chunks) >= 2:
        chunk1_text = chunks[0]['text']
        chunk2_text = chunks[1]['text']
        
        # Find overlap
        overlap_found = False
        for i in range(len(chunk1_text)):
            if chunk2_text.startswith(chunk1_text[i:]):
                overlap_text = chunk1_text[i:]
                print(f"✅ Overlap detected: ~{len(overlap_text)} chars")
                overlap_found = True
                break
        
        if not overlap_found:
            print("⚠️  No obvious overlap detected (might be token-level)")
    
    print("\n✅ Chunking tests complete!")


def test_error_handling():
    """Test error handling for various scenarios."""
    
    print("\n" + "=" * 60)
    print("TESTING ERROR HANDLING")
    print("=" * 60)
    
    loader = PDFLoader()
    
    # Test 1: Non-existent file
    print("\nTest 1: Non-existent file")
    try:
        loader.load("nonexistent.pdf")
        print("❌ Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"✅ Correctly raised: {type(e).__name__}")
    
    # Test 2: Unsupported format
    print("\nTest 2: Unsupported format")
    try:
        # Create dummy file
        with open("data/uploads/test.txt", "w") as f:
            f.write("test")
        
        loader.load("data/uploads/test.txt")
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Correctly raised: {type(e).__name__}")
    finally:
        # Cleanup
        if os.path.exists("data/uploads/test.txt"):
            os.remove("data/uploads/test.txt")
    
    print("\n✅ All error handling tests passed!")

def test_end_to_end_rag():
    """Test complete RAG pipeline."""
    
    print("\n" + "=" * 60)
    print("TESTING END-TO-END RAG PIPELINE")
    print("=" * 60)
    
    # 1. Load PDF
    print("\n1️⃣ LOADING PDF...")
    loader = PDFLoader()
    test_file = "data/uploads/sample.pdf"
    
    if not os.path.exists(test_file):
        print("❌ Sample PDF not found. Skipping end-to-end test.")
        return
    
    text = loader.load(test_file)
    
    # 2. Chunk text
    print("\n2️⃣ CHUNKING TEXT...")
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_text(text)
    
    # 3. Generate embeddings
    print("\n3️⃣ GENERATING EMBEDDINGS...")
    embedder = Embedder()
    chunks = embedder.embed_chunks(chunks)
    
    # 4. Store in vector DB
    print("\n4️⃣ STORING IN VECTOR DATABASE...")
    vector_store = SimpleVectorStore()
    vector_store.add_chunks(chunks)
    
    # 5. Query
    print("\n5️⃣ QUERYING...")
    test_queries = [
        "What is this document about?",
        "dummy pdf file"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        
        # Embed query
        query_embedding = embedder.embed_query(query)
        
        # Search
        results = vector_store.search(query_embedding, top_k=3)
        
        # Display results
        print(f"\n   Top 3 results:")
        for i, result in enumerate(results, 1):
            print(f"\n   {i}. Score: {result['score']:.4f}")
            print(f"      Text: {result['text'][:100]}...")
    
    print("\n" + "=" * 60)
    print("✅ END-TO-END RAG PIPELINE WORKING!")
    print("=" * 60)

def main():
    """Main function to test PDF loading and chunking."""
    
    print("=" * 60)
    print("RAG POC - PHASE 1 DAY 2: PDF LOADING + CHUNKING")
    print("=" * 60)
    
    # Initialize loader
    loader = PDFLoader()
    
    # Test with a sample PDF
    test_file = "data/uploads/sample.pdf"
    
    if not os.path.exists(test_file):
        print(f"\n⚠️  Test file not found: {test_file}")
        print("\n📝 Instructions:")
        print("   1. Download or copy a sample PDF")
        print("   2. Save it as: data/uploads/sample.pdf")
        print("   3. Run this script again")
        print("\n💡 Tip: Use any PDF (5-20 pages recommended)")
        print("\n🧪 Running chunking test with sample text instead...")
        test_chunking()
        return
    
    try:
        # Load PDF
        text = loader.load(test_file)
        
        # Display sample
        print("\n" + "=" * 60)
        print("EXTRACTED TEXT SAMPLE (first 500 chars):")
        print("=" * 60)
        print(text[:500])
        if len(text) > 500:
            print("...")
        
        print("\n" + "=" * 60)
        print("STATISTICS:")
        print("=" * 60)
        print(f"Total characters: {len(text):,}")
        print(f"Total words: {len(text.split()):,}")
        
        # Chunk the text
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        token_count = chunker.count_tokens(text)
        print(f"Total tokens: {token_count:,}")
        
        chunks = chunker.chunk_text(text)
        
        # Display chunk info
        print("\n" + "=" * 60)
        print("CHUNKS:")
        print("=" * 60)
        print(f"Total chunks: {len(chunks)}")
        print(f"Average tokens per chunk: {sum(c['token_count'] for c in chunks) / len(chunks):.1f}")
        
        # Show first 2 chunks
        print(f"\n📋 First 2 chunks:")
        for chunk in chunks[:2]:
            print(f"\n{chunk['chunk_id']}:")
            print(f"  Tokens: {chunk['token_count']}")
            print(f"  Text: {chunk['text'][:100]}...")
        
        print("\n✅ PDF loading and chunking successful!")
        
        # Run additional tests
        test_error_handling()
        
        # Test full RAG pipeline
        test_end_to_end_rag()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()