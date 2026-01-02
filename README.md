# Graph RAG Builder v1.9.8

LLM & RAG 비교 분석 시스템 - Knowledge Graph 기반 RAG vs Vector RAG vs Baseline LLM 비교

## 🚀 배포 구조

```
Frontend (Vercel) ──────► Backend (Railway) ──────► Neo4j Aura
                                   │
                                   └──────────────► Together AI
```

## 📁 프로젝트 구조

```
graph-rag-builder/
├── backend/           # FastAPI 백엔드 (Railway 배포)
│   ├── main.py
│   ├── graph_extractor.py
│   ├── graph_retriever.py
│   ├── neo4j_service.py
│   ├── vector_rag.py
│   ├── baseline_llm.py
│   ├── config.py
│   ├── requirements.txt
│   └── Procfile
├── frontend/          # React 프론트엔드 (Vercel 배포)
│   ├── src/
│   ├── package.json
│   ├── vite.config.ts
│   └── vercel.json
└── README.md
```

## 🔧 환경 변수

### Backend (Railway)
```
NEO4J_URI=bolt+s://xxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=xxx
TOGETHER_API_KEY=xxx
TOGETHER_BASE_URL=https://api.together.xyz/v1
TOGETHER_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
```

### Frontend (Vercel)
```
VITE_API_URL=https://your-backend.railway.app
```

## 📦 로컬 개발

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 5176
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## ✨ 주요 기능

1. **Graph RAG**: Neo4j 기반 Knowledge Graph 검색
2. **Vector RAG**: ChromaDB 기반 벡터 검색
3. **Baseline LLM**: 순수 LLM 응답
4. **비교 분석**: 세 가지 방식 동시 비교

## 📝 버전 이력

### v1.9.8 (2026-01-02)
- 병렬 처리 구현 (3개 청크 동시 처리)
- 처리 시간 3배 단축
- chunk_size 최적화 (600자)
