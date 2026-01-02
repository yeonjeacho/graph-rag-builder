"""
LLM & RAG Comparative Analysis System - FastAPI Backend
Python/LangChain/LangGraph implementation with Baseline, Vector RAG, and Graph RAG
"""
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from config import get_settings
from neo4j_service import Neo4jService
from graph_extractor import extract_knowledge_graph, create_extractor_with_config
from graph_retriever import retrieve_from_graph, create_retriever_with_config
from file_parser import parse_file, get_supported_extensions
from baseline_llm import BaselineLLM
from vector_rag import VectorRAG


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    print("🚀 Starting LLM & RAG Comparative Analysis System")
    settings = get_settings()
    print(f"📊 Neo4j URI: {settings.neo4j_uri}")
    
    # Verify Neo4j connection
    status = Neo4jService.verify_connection()
    if status["connected"]:
        print("✅ Neo4j connection verified")
    else:
        print(f"⚠️ Neo4j connection failed: {status.get('error')}")
    
    # Initialize Vector RAG
    try:
        VectorRAG()
        print("✅ ChromaDB initialized")
    except Exception as e:
        print(f"⚠️ ChromaDB initialization failed: {e}")
    
    yield
    
    print("👋 Shutting down...")
    Neo4jService.close()


# ============== App ==============

app = FastAPI(
    title="LLM & RAG Comparative Analysis System",
    description="Compare Baseline LLM, Vector RAG, and Graph RAG approaches",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Request/Response Models ==============

class StatusResponse(BaseModel):
    connected: bool
    error: Optional[str] = None


class StatsResponse(BaseModel):
    nodeCount: int
    relationshipCount: int
    error: Optional[str] = None


class VectorStatsResponse(BaseModel):
    success: bool
    chunk_count: int = 0
    error: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    params: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    success: bool
    records: List[Dict[str, Any]] = []
    nodes: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    summary: Dict[str, int] = {}
    error: Optional[str] = None


class LLMConfig(BaseModel):
    api_key: str = Field(..., min_length=1)
    model: str = Field(default="gpt-4o")
    base_url: Optional[str] = None


class ExtractRequest(BaseModel):
    document: str = Field(..., min_length=10)
    llm_config: Optional[LLMConfig] = None


class Entity(BaseModel):
    name: str
    type: str
    properties: Optional[Dict[str, Any]] = None


class Relationship(BaseModel):
    source: str
    target: str
    type: str
    properties: Optional[Dict[str, Any]] = None


class ExtractResponse(BaseModel):
    success: bool
    entities: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    error: Optional[str] = None


class BuildRequest(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]


class BuildResponse(BaseModel):
    success: bool
    nodesCreated: int = 0
    relationshipsCreated: int = 0
    error: Optional[str] = None


class ClearResponse(BaseModel):
    success: bool
    error: Optional[str] = None


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=2)
    llm_config: Optional[LLMConfig] = None


class ExtractedEntity(BaseModel):
    name: str
    type: str
    confidence: float
    matchedNodeId: Optional[int] = None
    matchedNodeName: Optional[str] = None


class RetrievalStep(BaseModel):
    step: str
    description: str
    count: Optional[int] = None


class RetrieveResponse(BaseModel):
    success: bool
    query: str = ""
    extractedEntities: List[ExtractedEntity] = []
    matchedNodes: List[Dict[str, Any]] = []
    nodes: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    context: str = ""
    answer: str = ""
    retrievalSteps: List[RetrievalStep] = []
    error: Optional[str] = None


class FileUploadResponse(BaseModel):
    success: bool
    text: str = ""
    filename: str = ""
    file_size: int = 0
    error: Optional[str] = None


class ModelsResponse(BaseModel):
    models: List[Dict[str, str]]


# ============== New Models for Comparative Analysis ==============

class BaselineRequest(BaseModel):
    question: str = Field(..., min_length=2)
    llm_config: Optional[LLMConfig] = None


class BaselineResponse(BaseModel):
    success: bool
    answer: str = ""
    model: str = ""
    method: str = "baseline"
    error: Optional[str] = None


class VectorBuildRequest(BaseModel):
    documents: List[str] = Field(..., min_length=1)
    llm_config: Optional[LLMConfig] = None
    chunking_strategy: str = Field(default="recursive", description="Chunking strategy: 'recursive' (default) or 'semantic'")


class ChunkDetail(BaseModel):
    chunk_id: str
    doc_index: int
    chunk_index: int
    content: str
    char_count: int
    word_count: int


class ChunkingConfig(BaseModel):
    chunk_size: int = 0
    chunk_overlap: int = 0
    separators: List[str] = []
    embedding_model: Optional[str] = None
    strategy: str = "recursive"
    similarity_threshold: Optional[float] = None


class VectorBuildResponse(BaseModel):
    success: bool
    chunks_added: int = 0
    total_chunks: int = 0
    error: Optional[str] = None
    chunk_details: List[ChunkDetail] = []
    chunking_config: Optional[ChunkingConfig] = None


class VectorRetrieveRequest(BaseModel):
    question: str = Field(..., min_length=2)
    llm_config: Optional[LLMConfig] = None
    top_k: int = Field(default=5, ge=1, le=20)


class VectorRetrieveResponse(BaseModel):
    success: bool
    answer: str = ""
    model: str = ""
    method: str = "vector_rag"
    sources: List[Dict[str, Any]] = []
    error: Optional[str] = None


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=2)
    llm_config: Optional[LLMConfig] = None


class CompareResult(BaseModel):
    method: str
    success: bool
    answer: str = ""
    model: str = ""
    sources: List[Dict[str, Any]] = []
    nodes: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    error: Optional[str] = None
    latency_ms: float = 0


class CompareResponse(BaseModel):
    success: bool
    question: str
    baseline: CompareResult
    vector_rag: CompareResult
    graph_rag: CompareResult
    error: Optional[str] = None


# ============== API Endpoints ==============

@app.get("/")
async def root():
    return {
        "name": "LLM & RAG Comparative Analysis System",
        "version": "3.0.0",
        "framework": "LangChain/LangGraph",
        "status": "running"
    }


@app.get("/api/neo4j/status", response_model=StatusResponse)
async def get_status():
    return Neo4jService.verify_connection()


@app.get("/api/neo4j/stats", response_model=StatsResponse)
async def get_stats():
    try:
        return Neo4jService.get_stats()
    except Exception as e:
        return StatsResponse(nodeCount=0, relationshipCount=0, error=str(e))


@app.get("/api/vector/stats", response_model=VectorStatsResponse)
async def get_vector_stats():
    """Get Vector DB statistics"""
    try:
        rag = VectorRAG()
        result = rag.get_stats()
        return VectorStatsResponse(**result)
    except Exception as e:
        return VectorStatsResponse(success=False, error=str(e))


@app.get("/api/models", response_model=ModelsResponse)
async def get_available_models():
    models = [
        {"id": "gpt-4o", "name": "GPT-4o", "description": "OpenAI GPT-4o (최신 멀티모달)"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "description": "GPT-4o 경량 버전"},
        {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "description": "GPT-4 Turbo (128K 컨텍스트)"},
        {"id": "gpt-4", "name": "GPT-4", "description": "OpenAI GPT-4"},
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "빠르고 경제적인 모델"},
        {"id": "gpt-oss-20b", "name": "GPT-OSS 20B", "description": "GPT-OSS 20B 오픈소스 모델"},
    ]
    return ModelsResponse(models=models)


# ============== Neo4j / Graph RAG Endpoints ==============

@app.post("/api/neo4j/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    try:
        result = Neo4jService.run_query(request.query, request.params or {})
        return QueryResponse(success=True, **result)
    except Exception as e:
        return QueryResponse(success=False, error=str(e))


@app.post("/api/neo4j/extract", response_model=ExtractResponse)
async def extract_graph(request: ExtractRequest):
    try:
        if request.llm_config:
            result = extract_knowledge_graph(
                request.document,
                api_key=request.llm_config.api_key,
                model=request.llm_config.model,
                base_url=request.llm_config.base_url
            )
        else:
            result = extract_knowledge_graph(request.document)
        return ExtractResponse(success=True, **result)
    except Exception as e:
        return ExtractResponse(success=False, error=str(e))


@app.post("/api/neo4j/build", response_model=BuildResponse)
async def build_graph(request: BuildRequest):
    try:
        entities = [e.model_dump() for e in request.entities]
        relationships = [r.model_dump() for r in request.relationships]
        result = Neo4jService.create_knowledge_graph(entities, relationships)
        return BuildResponse(success=True, **result)
    except Exception as e:
        return BuildResponse(success=False, error=str(e))


@app.post("/api/neo4j/clear", response_model=ClearResponse)
async def clear_graph():
    try:
        Neo4jService.clear_graph()
        return ClearResponse(success=True)
    except Exception as e:
        return ClearResponse(success=False, error=str(e))


@app.post("/api/neo4j/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    try:
        if request.llm_config:
            result = retrieve_from_graph(
                request.question,
                api_key=request.llm_config.api_key,
                model=request.llm_config.model,
                base_url=request.llm_config.base_url
            )
        else:
            result = retrieve_from_graph(request.question)
        return RetrieveResponse(success=True, **result)
    except Exception as e:
        return RetrieveResponse(success=False, query=request.question, error=str(e))


# ============== Baseline LLM Endpoints ==============

@app.post("/api/baseline/chat", response_model=BaselineResponse)
async def baseline_chat(request: BaselineRequest):
    """Chat with LLM directly without any retrieval (Baseline)"""
    try:
        if request.llm_config:
            llm = BaselineLLM(
                api_key=request.llm_config.api_key,
                model=request.llm_config.model,
                base_url=request.llm_config.base_url
            )
        else:
            llm = BaselineLLM()
        
        result = llm.answer(request.question)
        return BaselineResponse(**result)
    except Exception as e:
        return BaselineResponse(success=False, error=str(e))


# ============== Vector RAG Endpoints ==============

@app.post("/api/vector/build", response_model=VectorBuildResponse)
async def vector_build(request: VectorBuildRequest):
    """Add documents to Vector DB with optional semantic chunking"""
    try:
        if request.llm_config:
            rag = VectorRAG(
                api_key=request.llm_config.api_key,
                model=request.llm_config.model,
                base_url=request.llm_config.base_url
            )
        else:
            rag = VectorRAG()
        
        # Branch based on chunking strategy
        if request.chunking_strategy == "semantic":
            result = rag.add_documents_semantic(request.documents)
        else:
            # Default: recursive chunking (existing behavior)
            result = rag.add_documents(request.documents)
        
        return VectorBuildResponse(**result)
    except Exception as e:
        return VectorBuildResponse(success=False, error=str(e))


@app.post("/api/vector/clear", response_model=ClearResponse)
async def vector_clear():
    """Clear Vector DB"""
    try:
        rag = VectorRAG()
        result = rag.clear()
        return ClearResponse(**result)
    except Exception as e:
        return ClearResponse(success=False, error=str(e))


@app.post("/api/vector/retrieve", response_model=VectorRetrieveResponse)
async def vector_retrieve(request: VectorRetrieveRequest):
    """Retrieve answer using Vector RAG"""
    try:
        if request.llm_config:
            rag = VectorRAG(
                api_key=request.llm_config.api_key,
                model=request.llm_config.model,
                base_url=request.llm_config.base_url
            )
        else:
            rag = VectorRAG()
        
        result = rag.answer(request.question, request.top_k)
        return VectorRetrieveResponse(**result)
    except Exception as e:
        return VectorRetrieveResponse(success=False, error=str(e))


# ============== Comparison Endpoint ==============

@app.post("/api/compare", response_model=CompareResponse)
async def compare_all(request: CompareRequest):
    """Compare all three methods: Baseline, Vector RAG, Graph RAG"""
    import time
    
    question = request.question
    llm_config = request.llm_config
    
    results = {
        "baseline": CompareResult(method="baseline", success=False),
        "vector_rag": CompareResult(method="vector_rag", success=False),
        "graph_rag": CompareResult(method="graph_rag", success=False)
    }
    
    # 1. Baseline LLM
    try:
        start = time.time()
        if llm_config:
            llm = BaselineLLM(api_key=llm_config.api_key, model=llm_config.model, base_url=llm_config.base_url)
        else:
            llm = BaselineLLM()
        baseline_result = llm.answer(question)
        latency = (time.time() - start) * 1000
        
        results["baseline"] = CompareResult(
            method="baseline",
            success=baseline_result.get("success", False),
            answer=baseline_result.get("answer", ""),
            model=baseline_result.get("model", ""),
            error=baseline_result.get("error"),
            latency_ms=latency
        )
    except Exception as e:
        results["baseline"] = CompareResult(method="baseline", success=False, error=str(e))
    
    # 2. Vector RAG
    try:
        start = time.time()
        if llm_config:
            rag = VectorRAG(api_key=llm_config.api_key, model=llm_config.model, base_url=llm_config.base_url)
        else:
            rag = VectorRAG()
        vector_result = rag.answer(question)
        latency = (time.time() - start) * 1000
        
        results["vector_rag"] = CompareResult(
            method="vector_rag",
            success=vector_result.get("success", False),
            answer=vector_result.get("answer", ""),
            model=vector_result.get("model", ""),
            sources=vector_result.get("sources", []),
            error=vector_result.get("error"),
            latency_ms=latency
        )
    except Exception as e:
        results["vector_rag"] = CompareResult(method="vector_rag", success=False, error=str(e))
    
    # 3. Graph RAG
    try:
        start = time.time()
        if llm_config:
            graph_result = retrieve_from_graph(
                question,
                api_key=llm_config.api_key,
                model=llm_config.model,
                base_url=llm_config.base_url
            )
        else:
            graph_result = retrieve_from_graph(question)
        latency = (time.time() - start) * 1000
        
        results["graph_rag"] = CompareResult(
            method="graph_rag",
            success=True,
            answer=graph_result.get("answer", ""),
            model=llm_config.model if llm_config else "default",
            nodes=graph_result.get("nodes", []),
            relationships=graph_result.get("relationships", []),
            latency_ms=latency
        )
    except Exception as e:
        results["graph_rag"] = CompareResult(method="graph_rag", success=False, error=str(e))
    
    return CompareResponse(
        success=True,
        question=question,
        baseline=results["baseline"],
        vector_rag=results["vector_rag"],
        graph_rag=results["graph_rag"]
    )


# ============== File Upload ==============

@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename or "unknown"
        text = parse_file(filename, content)
        
        if not text.strip():
            return FileUploadResponse(success=False, error="파일에서 텍스트를 추출할 수 없습니다")
        
        return FileUploadResponse(success=True, text=text, filename=filename, file_size=len(content))
    except Exception as e:
        return FileUploadResponse(success=False, error=str(e))


@app.get("/api/supported-formats")
async def get_supported_formats():
    return {"formats": get_supported_extensions(), "description": "지원되는 파일 형식: PDF, TXT, MD, DOCX"}


# ============== Health Check ==============

@app.get("/health")
async def health_check():
    neo4j_status = Neo4jService.verify_connection()
    vector_stats = VectorRAG().get_stats() if VectorRAG._chroma_client else {"success": False}
    
    return {
        "status": "healthy" if neo4j_status["connected"] else "degraded",
        "neo4j": neo4j_status,
        "vector_db": vector_stats
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
