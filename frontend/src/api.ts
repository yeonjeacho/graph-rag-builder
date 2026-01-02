/**
 * API Client for LLM & RAG Comparative Analysis System
 */

const API_BASE = import.meta.env.VITE_API_BASE || '/api/neo4j';
const API_ROOT = import.meta.env.VITE_API_ROOT || '/api';

// ============== Types ==============

export interface StatusResponse {
  connected: boolean;
  error?: string;
}

export interface StatsResponse {
  nodeCount: number;
  relationshipCount: number;
  error?: string;
}

export interface VectorStatsResponse {
  success: boolean;
  chunk_count: number;
  error?: string;
}

export interface Entity {
  name: string;
  type: string;
  properties?: Record<string, unknown>;
}

export interface Relationship {
  source: string;
  target: string;
  type: string;
  relation?: string;
  properties?: Record<string, unknown>;
}

export interface LLMConfig {
  api_key: string;
  model: string;
  base_url?: string;
}

export interface ExtractResponse {
  success: boolean;
  entities: Entity[];
  relationships: Relationship[];
  error?: string;
}

export interface BuildResponse {
  success: boolean;
  nodesCreated: number;
  relationshipsCreated: number;
  error?: string;
}

export interface QueryRecord {
  [key: string]: unknown;
}

export interface QueryResponse {
  success: boolean;
  records: QueryRecord[];
  nodes?: GraphNode[];
  relationships?: GraphRelationship[];
  summary: {
    nodesCreated: number;
    relationshipsCreated: number;
  };
  error?: string;
}

export interface ExtractedEntity {
  name: string;
  type: string;
  confidence: number;
  matchedNodeId?: number;
  matchedNodeName?: string;
}

export interface RetrievalStep {
  step: string;
  description: string;
  result?: string;
  count?: number;
}

export interface GraphNode {
  id: number;
  labels: string[];
  properties: Record<string, unknown>;
}

export interface GraphRelationship {
  id: number;
  type: string;
  startNodeId: number;
  endNodeId: number;
  properties: Record<string, unknown>;
}

export interface RetrieveResponse {
  success: boolean;
  query: string;
  extractedEntities: ExtractedEntity[];
  matchedNodes: GraphNode[];
  nodes: GraphNode[];
  relationships: GraphRelationship[];
  context: string;
  answer: string;
  retrievalSteps: RetrievalStep[];
  error?: string;
}

export interface FileUploadResponse {
  success: boolean;
  text: string;
  content?: string;
  filename: string;
  file_size: number;
  error?: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

// ============== New Types for Comparative Analysis ==============

export interface BaselineResponse {
  success: boolean;
  answer: string;
  model: string;
  method: string;
  error?: string;
}

export interface ChunkDetail {
  chunk_id: string;
  doc_index: number;
  chunk_index: number;
  content: string;
  preview?: string;
  char_count: number;
  word_count: number;
}

export interface ChunkingConfig {
  chunk_size: number;
  chunk_overlap: number;
  separators?: string[];
  strategy?: string;
  similarity_threshold?: number;
}

export type ChunkingStrategy = 'recursive' | 'semantic';

export interface VectorBuildResponse {
  success: boolean;
  chunks_added: number;
  chunks?: ChunkDetail[];
  total_chunks: number;
  chunk_details?: ChunkDetail[];
  chunking_config?: ChunkingConfig;
  error?: string;
}

export interface VectorSource {
  rank: number;
  content: string;
  similarity: number;
  metadata?: Record<string, unknown>;
}

export interface VectorRetrieveResponse {
  success: boolean;
  answer: string;
  model: string;
  method: string;
  sources: VectorSource[];
  error?: string;
}

export interface CompareResult {
  method: string;
  success: boolean;
  answer: string;
  model: string;
  sources: VectorSource[];
  nodes: GraphNode[];
  relationships: GraphRelationship[];
  steps?: RetrievalStep[];
  error?: string;
  latency_ms: number;
}

export interface CompareResponse {
  success: boolean;
  question: string;
  baseline: CompareResult;
  vector_rag: CompareResult;
  graph_rag: CompareResult;
  error?: string;
}

// ============== API Functions ==============

async function fetchApi<T>(endpoint: string, options?: RequestInit, timeoutMs: number = 300000): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
      signal: controller.signal,
    });
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
  } finally {
    clearTimeout(timeoutId);
  }
}

async function fetchRootApi<T>(endpoint: string, options?: RequestInit, timeoutMs: number = 300000): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  
  const headers: Record<string, string> = {};
  if (options?.body && typeof options.body === 'string') {
    headers['Content-Type'] = 'application/json';
  }
  
  try {
    const response = await fetch(`${API_ROOT}${endpoint}`, {
      headers,
      ...options,
      signal: controller.signal,
    });
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
  } finally {
    clearTimeout(timeoutId);
  }
}

export const api = {
  // ============== Neo4j / Graph RAG ==============
  
  getStatus: () => fetchApi<StatusResponse>('/status'),
  
  getStats: () => fetchApi<StatsResponse>('/stats'),
  
  getModels: () => fetchRootApi<ModelsResponse>('/models'),
  
  extract: (document: string, llmConfig?: LLMConfig) => 
    fetchApi<ExtractResponse>('/extract', {
      method: 'POST',
      body: JSON.stringify({ document, llm_config: llmConfig }),
    }, 900000),  // 15분 타임아웃 (긴 문서 처리용)
  
  build: (entities: Entity[], relationships: Relationship[]) =>
    fetchApi<BuildResponse>('/build', {
      method: 'POST',
      body: JSON.stringify({ entities, relationships }),
    }),
  
  query: (query: string, params?: Record<string, unknown>) =>
    fetchApi<QueryResponse>('/query', {
      method: 'POST',
      body: JSON.stringify({ query, params }),
    }),
  
  retrieve: (question: string, llmConfig?: LLMConfig) =>
    fetchApi<RetrieveResponse>('/retrieve', {
      method: 'POST',
      body: JSON.stringify({ question, llm_config: llmConfig }),
    }),
  
  clear: () =>
    fetchApi<{ success: boolean; error?: string }>('/clear', {
      method: 'POST',
    }),
  
  uploadFile: async (file: File): Promise<FileUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_ROOT}/upload`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
  },
  
  // ============== Baseline LLM ==============
  
  baselineChat: (question: string, llmConfig?: LLMConfig) =>
    fetchRootApi<BaselineResponse>('/baseline/chat', {
      method: 'POST',
      body: JSON.stringify({ question, llm_config: llmConfig }),
    }),
  
  // ============== Vector RAG ==============
  
  getVectorStats: () => fetchRootApi<VectorStatsResponse>('/vector/stats'),
  
  vectorBuild: (documents: string[], llmConfig?: LLMConfig, chunkingStrategy: ChunkingStrategy = 'recursive') =>
    fetchRootApi<VectorBuildResponse>('/vector/build', {
      method: 'POST',
      body: JSON.stringify({ documents, llm_config: llmConfig, chunking_strategy: chunkingStrategy }),
    }),
  
  vectorClear: () =>
    fetchRootApi<{ success: boolean; error?: string }>('/vector/clear', {
      method: 'POST',
    }),
  
  vectorRetrieve: (question: string, llmConfig?: LLMConfig, topK: number = 5) =>
    fetchRootApi<VectorRetrieveResponse>('/vector/retrieve', {
      method: 'POST',
      body: JSON.stringify({ question, llm_config: llmConfig, top_k: topK }),
    }),
  
  // ============== Additional Vector RAG ==============
  
  indexDocument: (document: string, config?: { chunk_size?: number; chunk_overlap?: number }) =>
    fetchRootApi<VectorBuildResponse>('/vector/build', {
      method: 'POST',
      body: JSON.stringify({ documents: [document], chunk_size: config?.chunk_size, chunk_overlap: config?.chunk_overlap }),
    }),
  
  vectorSearch: (question: string, llmConfig?: LLMConfig) =>
    fetchRootApi<VectorRetrieveResponse>('/vector/retrieve', {
      method: 'POST',
      body: JSON.stringify({ question, llm_config: llmConfig }),
    }),
  
  // ============== Additional Graph RAG ==============
  
  extractAndStore: async (document: string, llmConfig?: LLMConfig) => {
    // First extract entities and relationships (15분 타임아웃)
    const extractResult = await fetchApi<ExtractResponse>('/extract', {
      method: 'POST',
      body: JSON.stringify({ document, llm_config: llmConfig }),
    }, 900000);  // 15분 타임아웃 (긴 문서 처리용)
    
    if (!extractResult.success || !extractResult.entities || !extractResult.relationships) {
      return extractResult;
    }
    
    // Then build the graph
    await fetchApi<BuildResponse>('/build', {
      method: 'POST',
      body: JSON.stringify({ 
        entities: extractResult.entities, 
        relationships: extractResult.relationships 
      }),
    });
    
    return extractResult;
  },
  
  graphSearch: (question: string, llmConfig?: LLMConfig) =>
    fetchApi<RetrieveResponse>('/retrieve', {
      method: 'POST',
      body: JSON.stringify({ question, llm_config: llmConfig }),
    }),
  
  clearGraph: () =>
    fetchApi<{ success: boolean; error?: string }>('/clear', {
      method: 'POST',
    }),
  
  // ============== Comparison ==============
  
  compare: (question: string, llmConfig?: LLMConfig) =>
    fetchRootApi<CompareResponse>('/compare', {
      method: 'POST',
      body: JSON.stringify({ question, llm_config: llmConfig }),
    }),
};
