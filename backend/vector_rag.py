"""
Standard RAG Module - Vector DB based retrieval using ChromaDB
Uses local embedding model (no API key required for embedding)
"""
import os
import hashlib
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import get_settings


RAG_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question based on the provided context.
If the context doesn't contain relevant information, say so and provide what you know.
Always cite which parts of the context you used.
Answer in Korean.

## Context
{context}

## Instructions
1. Use only the information from the context above to answer.
2. If the context doesn't have enough information, acknowledge it.
3. Be concise and accurate.
4. Mention which sources you referenced."""


class LocalEmbeddings:
    """Local embedding model using sentence-transformers (no API key required)"""
    
    _model = None
    _model_name = "intfloat/multilingual-e5-small"  # 다국어 지원, 가볍고 성능 우수
    
    def __init__(self):
        if LocalEmbeddings._model is None:
            print(f"📥 Loading local embedding model: {LocalEmbeddings._model_name}")
            LocalEmbeddings._model = SentenceTransformer(LocalEmbeddings._model_name)
            print("✅ Local embedding model loaded")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        # E5 모델은 "query: " 또는 "passage: " 접두사 권장
        prefixed_texts = [f"passage: {text}" for text in texts]
        embeddings = LocalEmbeddings._model.encode(prefixed_texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        prefixed_text = f"query: {text}"
        embedding = LocalEmbeddings._model.encode([prefixed_text], normalize_embeddings=True)
        return embedding[0].tolist()
    
    @property
    def model_name(self) -> str:
        return LocalEmbeddings._model_name


class VectorRAG:
    """Standard RAG using ChromaDB for vector storage and retrieval"""
    
    _chroma_client = None
    _collection = None
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        
        # Base URL priority: user-provided > settings > OpenAI default
        if base_url and base_url.strip():
            # User explicitly provided a custom base URL (e.g., Together AI, Groq)
            self.base_url = base_url
        elif api_key:
            # User provided API key but no base URL -> use OpenAI official API
            self.base_url = "https://api.openai.com/v1"
        else:
            # Use settings (sandbox proxy)
            self.base_url = settings.openai_base_url
        
        llm_kwargs = {"model": self.model, "temperature": 0, "api_key": self.api_key}
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # 로컬 임베딩 모델 사용 (API Key 불필요)
        self.embeddings = LocalEmbeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT), ("human", "{question}")
        ])
        
        self._init_chroma()
    
    def _init_chroma(self):
        if VectorRAG._chroma_client is None:
            persist_dir = "/home/ubuntu/graph-rag-builder/data/chroma"
            os.makedirs(persist_dir, exist_ok=True)
            VectorRAG._chroma_client = chromadb.PersistentClient(
                path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
            )
            VectorRAG._collection = VectorRAG._chroma_client.get_or_create_collection(
                name="documents", metadata={"hnsw:space": "cosine"}
            )
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            return {
                "success": True, 
                "chunk_count": VectorRAG._collection.count(),
                "embedding_model": self.embeddings.model_name
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def add_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Add documents to vector store (no API key required - uses local embedding)"""
        all_chunks, all_metadatas, all_ids = [], [], []
        chunk_details = []
        chunking_config = {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "separators": ["\\n\\n", "\\n", ". ", " ", ""],
            "embedding_model": self.embeddings.model_name
        }
        
        try:
            # 1. 청킹 수행
            for doc_idx, doc in enumerate(documents):
                chunks = self.text_splitter.split_text(doc)
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = hashlib.md5(f"{doc_idx}_{chunk_idx}_{chunk[:50]}".encode()).hexdigest()
                    all_chunks.append(chunk)
                    all_metadatas.append({"doc_index": doc_idx, "chunk_index": chunk_idx})
                    all_ids.append(chunk_id)
                    
                    chunk_details.append({
                        "chunk_id": chunk_id,
                        "doc_index": doc_idx,
                        "chunk_index": chunk_idx,
                        "content": chunk,
                        "char_count": len(chunk),
                        "word_count": len(chunk.split())
                    })
            
            if not all_chunks:
                return {
                    "success": False, 
                    "chunks_added": 0,
                    "total_chunks": 0,
                    "error": "No chunks created",
                    "chunk_details": [],
                    "chunking_config": chunking_config
                }
            
            # 2. 로컬 임베딩 생성 (API Key 불필요)
            embeddings = self.embeddings.embed_documents(all_chunks)
            
            # 3. ChromaDB에 저장
            VectorRAG._collection.add(
                ids=all_ids, 
                embeddings=embeddings, 
                documents=all_chunks, 
                metadatas=all_metadatas
            )
            
            return {
                "success": True, 
                "chunks_added": len(all_chunks), 
                "total_chunks": VectorRAG._collection.count(),
                "chunk_details": chunk_details,
                "chunking_config": chunking_config
            }
            
        except Exception as e:
            return {
                "success": False, 
                "chunks_added": 0,
                "total_chunks": VectorRAG._collection.count() if VectorRAG._collection else 0,
                "error": str(e),
                "chunk_details": chunk_details,
                "chunking_config": chunking_config
            }
    
    def add_documents_semantic(self, documents: List[str]) -> Dict[str, Any]:
        """
        Add documents using semantic chunking (AI-based).
        This is a NEW method - does not modify existing add_documents.
        """
        from semantic_chunker import get_semantic_chunker
        
        all_chunks, all_metadatas, all_ids = [], [], []
        chunk_details = []
        
        # Semantic chunking config
        chunking_config = {
            "chunk_size": 0,  # Not applicable for semantic
            "chunk_overlap": 0,  # Not applicable for semantic
            "separators": [],
            "embedding_model": self.embeddings.model_name,
            "strategy": "semantic",
            "similarity_threshold": 0.5
        }
        
        try:
            # Get semantic chunker instance
            chunker = get_semantic_chunker(
                similarity_threshold=0.5,
                min_chunk_size=100,
                max_chunk_size=1000
            )
            
            # 1. Semantic chunking
            for doc_idx, doc in enumerate(documents):
                result = chunker.chunk_text_with_details(doc)
                
                for chunk_idx, detail in enumerate(result['chunk_details']):
                    chunk = detail['content']
                    chunk_id = hashlib.md5(f"semantic_{doc_idx}_{chunk_idx}_{chunk[:50]}".encode()).hexdigest()
                    
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "doc_index": doc_idx, 
                        "chunk_index": chunk_idx,
                        "strategy": "semantic"
                    })
                    all_ids.append(chunk_id)
                    
                    chunk_details.append({
                        "chunk_id": chunk_id,
                        "doc_index": doc_idx,
                        "chunk_index": chunk_idx,
                        "content": chunk,
                        "char_count": detail['char_count'],
                        "word_count": detail['word_count']
                    })
            
            if not all_chunks:
                return {
                    "success": False,
                    "chunks_added": 0,
                    "total_chunks": 0,
                    "error": "No chunks created from semantic analysis",
                    "chunk_details": [],
                    "chunking_config": chunking_config
                }
            
            # 2. Generate embeddings (using local model)
            embeddings = self.embeddings.embed_documents(all_chunks)
            
            # 3. Store in ChromaDB
            VectorRAG._collection.add(
                ids=all_ids,
                embeddings=embeddings,
                documents=all_chunks,
                metadatas=all_metadatas
            )
            
            return {
                "success": True,
                "chunks_added": len(all_chunks),
                "total_chunks": VectorRAG._collection.count(),
                "chunk_details": chunk_details,
                "chunking_config": chunking_config
            }
            
        except Exception as e:
            return {
                "success": False,
                "chunks_added": 0,
                "total_chunks": VectorRAG._collection.count() if VectorRAG._collection else 0,
                "error": str(e),
                "chunk_details": chunk_details,
                "chunking_config": chunking_config
            }
    
    def clear(self) -> Dict[str, Any]:
        try:
            VectorRAG._chroma_client.delete_collection("documents")
            VectorRAG._collection = VectorRAG._chroma_client.create_collection(
                name="documents", metadata={"hnsw:space": "cosine"}
            )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def retrieve(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve relevant chunks (no API key required - uses local embedding)"""
        try:
            if VectorRAG._collection.count() == 0:
                return {"success": False, "error": "Vector store is empty"}
            
            # 로컬 임베딩으로 쿼리 벡터 생성
            query_embedding = self.embeddings.embed_query(question)
            
            results = VectorRAG._collection.query(
                query_embeddings=[query_embedding], 
                n_results=min(top_k, VectorRAG._collection.count())
            )
            
            chunks = [
                {
                    "content": doc, 
                    "metadata": meta, 
                    "similarity": 1 - dist,  # cosine distance to similarity
                    "rank": i + 1
                }
                for i, (doc, meta, dist) in enumerate(
                    zip(results['documents'][0], results['metadatas'][0], results['distances'][0])
                )
            ]
            
            return {"success": True, "chunks": chunks}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Generate answer using retrieved context (requires API key for LLM)"""
        try:
            # 1. 검색 (로컬 임베딩 - API Key 불필요)
            retrieval = self.retrieve(question, top_k)
            if not retrieval["success"]:
                return {
                    "success": False, 
                    "answer": "", 
                    "error": retrieval.get("error"), 
                    "method": "vector_rag"
                }
            
            chunks = retrieval["chunks"]
            context = "\n\n".join([
                f"[Source {c['rank']}] (Similarity: {c['similarity']:.2f})\n{c['content']}" 
                for c in chunks
            ])
            sources = [
                {"rank": c["rank"], "content": c["content"], "similarity": c["similarity"]} 
                for c in chunks
            ]
            
            # 2. 답변 생성 (OpenAI LLM - API Key 필요)
            response = (self.prompt | self.llm).invoke({"context": context, "question": question})
            
            return {
                "success": True, 
                "answer": response.content, 
                "model": self.model, 
                "method": "vector_rag", 
                "sources": sources,
                "embedding_model": self.embeddings.model_name
            }
        except Exception as e:
            return {
                "success": False, 
                "answer": "", 
                "error": str(e), 
                "method": "vector_rag"
            }
