"""
Semantic Chunker Module - AI-based semantic text splitting
Uses sentence-transformers for embedding-based similarity chunking
Isolated module to prevent regression in existing code
"""
import os
import numpy as np
from typing import List, Dict, Any, Optional
# Import shared embedding model from vector_rag to avoid loading duplicate models
try:
    from vector_rag import LocalEmbeddings
    USE_SHARED_MODEL = True
except ImportError:
    from sentence_transformers import SentenceTransformer
    USE_SHARED_MODEL = False

# Disable tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class SemanticChunker:
    """
    Semantic chunking using sentence embeddings.
    Splits text based on semantic similarity between sentences.
    Uses the same model as VectorRAG to save memory.
    """
    
    _model = None
    _model_name = "intfloat/multilingual-e5-small"  # Same as VectorRAG embedding model
    
    def __init__(self, similarity_threshold: float = 0.5, min_chunk_size: int = 100, max_chunk_size: int = 1000):
        """
        Initialize SemanticChunker.
        
        Args:
            similarity_threshold: Threshold for grouping sentences (0.0-1.0, higher = more similar required)
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
        """
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model (singleton pattern, shared with VectorRAG)"""
        if SemanticChunker._model is None:
            if USE_SHARED_MODEL:
                # Reuse the model already loaded by VectorRAG
                print(f"🧠 Reusing VectorRAG embedding model for semantic chunking")
                embeddings = LocalEmbeddings()  # This uses singleton pattern
                SemanticChunker._model = LocalEmbeddings._model
                print("✅ Semantic chunking model loaded (shared with VectorRAG)")
            else:
                print(f"🧠 Loading semantic chunking model: {SemanticChunker._model_name}")
                SemanticChunker._model = SentenceTransformer(SemanticChunker._model_name)
                print("✅ Semantic chunking model loaded")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using multiple delimiters"""
        import re
        
        # Korean and English sentence delimiters
        # Split on period, question mark, exclamation mark followed by space or newline
        sentences = re.split(r'(?<=[.!?。？！])\s+|\n\n+', text)
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If no sentences found, split by newlines
        if len(sentences) <= 1 and '\n' in text:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
        
        return sentences
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks that are too small"""
        if not chunks:
            return chunks
        
        merged = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) < self.min_chunk_size:
                current_chunk = (current_chunk + " " + chunk).strip() if current_chunk else chunk
            else:
                if current_chunk:
                    merged.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            merged.append(current_chunk)
        
        return merged
    
    def _split_large_chunks(self, chunks: List[str]) -> List[str]:
        """Split chunks that are too large"""
        result = []
        
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                result.append(chunk)
            else:
                # Split by sentences within the chunk
                sentences = self._split_into_sentences(chunk)
                current = ""
                
                for sentence in sentences:
                    if len(current) + len(sentence) <= self.max_chunk_size:
                        current = (current + " " + sentence).strip() if current else sentence
                    else:
                        if current:
                            result.append(current)
                        current = sentence
                
                if current:
                    result.append(current)
        
        return result
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into semantically coherent chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks grouped by semantic similarity
        """
        if not text or not text.strip():
            return []
        
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []
        
        # Step 2: Generate embeddings for all sentences
        embeddings = SemanticChunker._model.encode(sentences, normalize_embeddings=True)
        
        # Step 3: Group sentences by semantic similarity
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            similarity = self._compute_similarity(current_embedding, embeddings[i])
            
            # Check if current chunk would exceed max size
            potential_chunk = " ".join(current_chunk_sentences + [sentences[i]])
            
            if similarity >= self.similarity_threshold and len(potential_chunk) <= self.max_chunk_size:
                # Add to current chunk
                current_chunk_sentences.append(sentences[i])
                # Update embedding as average of chunk sentences
                chunk_indices = list(range(i - len(current_chunk_sentences) + 1, i + 1))
                current_embedding = np.mean(embeddings[chunk_indices], axis=0)
            else:
                # Start new chunk
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i]]
                current_embedding = embeddings[i]
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
        
        # Step 4: Post-process chunks
        chunks = self._merge_small_chunks(chunks)
        chunks = self._split_large_chunks(chunks)
        
        return chunks
    
    def chunk_text_with_details(self, text: str) -> Dict[str, Any]:
        """
        Split text into chunks and return detailed information.
        
        Args:
            text: Input text to chunk
            
        Returns:
            Dictionary with chunks and metadata
        """
        chunks = self.chunk_text(text)
        
        chunk_details = []
        for i, chunk in enumerate(chunks):
            chunk_details.append({
                "chunk_index": i,
                "content": chunk,
                "char_count": len(chunk),
                "word_count": len(chunk.split()),
                "sentence_count": len(self._split_into_sentences(chunk))
            })
        
        return {
            "chunks": chunks,
            "chunk_count": len(chunks),
            "chunk_details": chunk_details,
            "config": {
                "strategy": "semantic",
                "model": self._model_name,
                "similarity_threshold": self.similarity_threshold,
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size
            }
        }


# Singleton instance for reuse
_semantic_chunker_instance: Optional[SemanticChunker] = None


def get_semantic_chunker(
    similarity_threshold: float = 0.5,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1000
) -> SemanticChunker:
    """Get or create a SemanticChunker instance"""
    global _semantic_chunker_instance
    
    if _semantic_chunker_instance is None:
        _semantic_chunker_instance = SemanticChunker(
            similarity_threshold=similarity_threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size
        )
    
    return _semantic_chunker_instance


def semantic_chunk(text: str, **kwargs) -> List[str]:
    """
    Convenience function for semantic chunking.
    
    Args:
        text: Input text to chunk
        **kwargs: Arguments passed to SemanticChunker
        
    Returns:
        List of text chunks
    """
    chunker = get_semantic_chunker(**kwargs)
    return chunker.chunk_text(text)
