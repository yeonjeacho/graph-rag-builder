import { useState, useEffect, useRef } from 'react';
import { api } from './api';
import type { Entity, Relationship, LLMConfig, ModelInfo, RetrievalStep, GraphNode, GraphRelationship, VectorSource, CompareResponse, ChunkDetail, ChunkingConfig } from './api';
import GraphVisualization from './components/GraphVisualization';

interface LLMSettings {
  apiKey: string;
  model: string;
  isConfigured: boolean;
  provider: 'openai' | 'together' | 'groq' | 'custom';
  baseUrl: string;
}

// ===== State interfaces for persistence =====
interface BaselineState {
  question: string;
  answer: string;
}

interface VectorRAGState {
  document: string;
  question: string;
  answer: string;
  sources: VectorSource[];
  chunks: ChunkDetail[];
  showChunks: boolean;
  chunkConfig: ChunkingConfig;
  chunkingStrategy: 'recursive' | 'semantic';
}

interface GraphRAGState {
  document: string;
  question: string;
  answer: string;
  entities: Entity[];
  relationships: Relationship[];
  steps: RetrievalStep[];
  refNodes: GraphNode[];
  refRelationships: GraphRelationship[];
}

interface CompareState {
  question: string;
  results: CompareResponse | null;
}

interface ExplorerState {
  query: string;
  results: { records: Record<string, unknown>[]; nodes: GraphNode[]; relationships: GraphRelationship[] } | null;
  selectedPreset: string;
}

// ===== Collapsible Section Component =====
function CollapsibleSection({ 
  title, 
  count, 
  children, 
  defaultOpen = false,
  colorClass = 'text-blue-600'
}: { 
  title: string; 
  count?: number; 
  children: React.ReactNode; 
  defaultOpen?: boolean;
  colorClass?: string;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  
  return (
    <div className="accordion">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="accordion-header"
      >
        <span className={colorClass}>
          {isOpen ? '▼' : '▶'} {title} {count !== undefined && `(${count}개)`}
        </span>
      </button>
      {isOpen && (
        <div className="p-4 max-h-64 overflow-auto" style={{ backgroundColor: 'var(--color-bg-secondary)' }}>
          {children}
        </div>
      )}
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState<'baseline' | 'vector' | 'graph' | 'compare' | 'explorer'>('baseline');
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('theme');
    return (saved === 'dark' || saved === 'light') ? saved : 'dark';
  });
  const [llmSettings, setLlmSettings] = useState<LLMSettings>({ apiKey: '', model: 'gpt-4o', isConfigured: false, provider: 'openai', baseUrl: '' });
  const [showLLMModal, setShowLLMModal] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [neo4jConnected, setNeo4jConnected] = useState(false);
  const [nodeCount, setNodeCount] = useState(0);
  const [relationshipCount, setRelationshipCount] = useState(0);
  const [vectorChunkCount, setVectorChunkCount] = useState(0);

  // ===== Lifted State for Tab Persistence =====
  const [baselineState, setBaselineState] = useState<BaselineState>({ question: '', answer: '' });
  const [vectorState, setVectorState] = useState<VectorRAGState>({
    document: '', question: '', answer: '', sources: [], chunks: [], showChunks: false,
    chunkConfig: { chunk_size: 500, chunk_overlap: 50 },
    chunkingStrategy: 'recursive'
  });
  const [graphState, setGraphState] = useState<GraphRAGState>({
    document: '', question: '', answer: '', entities: [], relationships: [], steps: [], refNodes: [], refRelationships: []
  });
  const [compareState, setCompareState] = useState<CompareState>({ question: '', results: null });
  const [explorerState, setExplorerState] = useState<ExplorerState>({ query: '', results: null, selectedPreset: '' });

  useEffect(() => {
    api.getModels().then(res => setModels(res.models)).catch(console.error);
    refreshStats();
  }, []);

  // Theme effect
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const refreshStats = async () => {
    try {
      const [status, stats, vectorStats] = await Promise.all([
        api.getStatus(),
        api.getStats(),
        api.getVectorStats()
      ]);
      setNeo4jConnected(status.connected);
      setNodeCount(stats.nodeCount);
      setRelationshipCount(stats.relationshipCount);
      setVectorChunkCount(vectorStats.chunk_count || 0);
    } catch (e) {
      console.error('Failed to refresh stats:', e);
    }
  };

  const getLLMConfig = (): LLMConfig | undefined => {
    if (llmSettings.isConfigured && llmSettings.apiKey) {
      return { 
        api_key: llmSettings.apiKey, 
        model: llmSettings.model,
        base_url: llmSettings.baseUrl || undefined
      };
    }
    return undefined;
  };

  const tabs = [
    { id: 'baseline', label: 'Baseline', icon: '💬' },
    { id: 'vector', label: 'Vector RAG', icon: '📊' },
    { id: 'graph', label: 'Graph RAG', icon: '🔗' },
    { id: 'compare', label: 'Compare', icon: '⚖️' },
    { id: 'explorer', label: 'Explorer', icon: '🔍' },
  ];

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--color-bg-primary)' }}>
      {/* Header */}
      <header className="app-header sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="app-title">RAG Comparison Studio</h1>
            <p className="app-subtitle">Baseline · Vector RAG · Graph RAG</p>
          </div>
          <div className="flex items-center gap-4">
            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="theme-toggle"
              title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            >
              {theme === 'dark' ? '☀️' : '🌙'}
            </button>
            <button
              onClick={() => setShowLLMModal(true)}
              className={`btn ${llmSettings.isConfigured ? 'btn-secondary' : 'btn-primary'}`}
              style={llmSettings.isConfigured ? { backgroundColor: theme === 'dark' ? '#14532d' : '#ecfdf5', color: theme === 'dark' ? '#6ee7b7' : '#059669', borderColor: theme === 'dark' ? '#166534' : '#a7f3d0' } : {}}
            >
              <span>{llmSettings.isConfigured ? '✓' : '🔑'}</span>
              {llmSettings.isConfigured ? `${llmSettings.model}` : 'API Key 설정'}
            </button>
            <div className="flex items-center gap-3">
              {/* Neo4j Status Badge */}
              <div className="status-badge" style={{ backgroundColor: neo4jConnected ? 'var(--color-success-bg)' : 'var(--color-danger-bg)' }}>
                <div className={`status-dot ${neo4jConnected ? 'online' : 'offline'}`}></div>
                <span style={{ color: neo4jConnected ? 'var(--color-success)' : 'var(--color-danger)' }}>Neo4j</span>
              </div>
              {/* Stats Badges */}
              <div className="status-badge" style={{ backgroundColor: 'var(--color-primary-bg)' }}>
                <span style={{ color: 'var(--color-primary)', fontWeight: 600 }}>{nodeCount}</span>
                <span style={{ color: 'var(--color-text-secondary)' }}>nodes</span>
              </div>
              <div className="status-badge" style={{ backgroundColor: 'var(--color-primary-bg)' }}>
                <span style={{ color: 'var(--color-primary)', fontWeight: 600 }}>{relationshipCount}</span>
                <span style={{ color: 'var(--color-text-secondary)' }}>rels</span>
              </div>
              <div className="status-badge" style={{ backgroundColor: 'var(--color-accent-bg)' }}>
                <span style={{ color: 'var(--color-accent)', fontWeight: 600 }}>{vectorChunkCount}</span>
                <span style={{ color: 'var(--color-text-secondary)' }}>chunks</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Navigation - Segmented Control Style */}
      <div className="border-b" style={{ backgroundColor: 'var(--color-bg-secondary)', borderColor: 'var(--color-border)' }}>
        <div className="max-w-7xl mx-auto px-6 py-3">
          <div className="tab-container">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as typeof activeTab)}
                className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              >
                <span className="mr-1.5">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'baseline' && (
          <BaselineTab 
            getLLMConfig={getLLMConfig} 
            state={baselineState} 
            setState={setBaselineState} 
          />
        )}
        {activeTab === 'vector' && (
          <VectorRAGTab 
            getLLMConfig={getLLMConfig} 
            state={vectorState} 
            setState={setVectorState}
            onRefresh={refreshStats}
          />
        )}
        {activeTab === 'graph' && (
          <GraphRAGTab 
            getLLMConfig={getLLMConfig} 
            state={graphState} 
            setState={setGraphState}
            onRefresh={refreshStats} 
          />
        )}
        {activeTab === 'compare' && (
          <CompareTab 
            getLLMConfig={getLLMConfig} 
            state={compareState} 
            setState={setCompareState} 
          />
        )}
        {activeTab === 'explorer' && (
          <GraphExplorerTab 
            state={explorerState} 
            setState={setExplorerState} 
          />
        )}
      </main>

      {/* LLM Settings Modal */}
      {showLLMModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="p-6">
              <h3 className="text-lg font-semibold mb-1" style={{ color: 'var(--color-text-primary)' }}>API Configuration</h3>
              <p className="text-sm mb-6" style={{ color: 'var(--color-text-tertiary)' }}>LLM API 연결 설정</p>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-secondary)' }}>Provider</label>
                  <select
                    value={llmSettings.provider}
                    onChange={(e) => {
                      const provider = e.target.value as 'openai' | 'together' | 'groq' | 'custom';
                      const baseUrlMap: Record<string, string> = {
                        'openai': '',
                        'together': 'https://api.together.xyz/v1',
                        'groq': 'https://api.groq.com/openai/v1',
                        'custom': llmSettings.baseUrl
                      };
                      const defaultModelMap: Record<string, string> = {
                        'openai': 'gpt-4o',
                        'together': 'openai/gpt-oss-20b',
                        'groq': 'llama-3.3-70b-versatile',
                        'custom': llmSettings.model
                      };
                      setLlmSettings(prev => ({ ...prev, provider, baseUrl: baseUrlMap[provider], model: defaultModelMap[provider] }));
                    }}
                    className="input"
                  >
                    <option value="openai">OpenAI</option>
                    <option value="together">Together AI (GPT-OSS)</option>
                    <option value="groq">Groq</option>
                    <option value="custom">Custom</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-secondary)' }}>API Key</label>
                  <input
                    type="password"
                    value={llmSettings.apiKey}
                    onChange={(e) => setLlmSettings(prev => ({ ...prev, apiKey: e.target.value }))}
                    placeholder={llmSettings.provider === 'together' ? 'Together AI API Key' : llmSettings.provider === 'groq' ? 'Groq API Key' : 'sk-...'}
                    className="input"
                  />
                </div>
                {llmSettings.provider !== 'openai' && (
                  <div>
                    <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-secondary)' }}>Base URL</label>
                    <input
                      type="text"
                      value={llmSettings.baseUrl}
                      onChange={(e) => setLlmSettings(prev => ({ ...prev, baseUrl: e.target.value }))}
                      placeholder="https://api.example.com/v1"
                      className="input"
                      disabled={llmSettings.provider !== 'custom'}
                      style={llmSettings.provider !== 'custom' ? { opacity: 0.7 } : {}}
                    />
                    {llmSettings.provider !== 'custom' && (
                      <p className="text-xs mt-1" style={{ color: 'var(--color-text-tertiary)' }}>Provider 선택 시 자동 설정됨</p>
                    )}
                  </div>
                )}
                <div>
                  <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-secondary)' }}>Model</label>
                  {llmSettings.provider === 'openai' ? (
                    <select
                      value={llmSettings.model}
                      onChange={(e) => setLlmSettings(prev => ({ ...prev, model: e.target.value }))}
                      className="input"
                    >
                      {models.map(m => (
                        <option key={m.id} value={m.id}>{m.name}</option>
                      ))}
                    </select>
                  ) : llmSettings.provider === 'together' ? (
                    <select
                      value={llmSettings.model}
                      onChange={(e) => setLlmSettings(prev => ({ ...prev, model: e.target.value }))}
                      className="input"
                    >
                      <option value="openai/gpt-oss-20b">🌟 GPT-OSS 20B (OpenAI)</option>
                      <option value="meta-llama/Llama-3.3-70B-Instruct-Turbo">Llama 3.3 70B Turbo</option>
                      <option value="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo">Llama 3.1 405B Turbo</option>
                      <option value="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo">Llama 3.1 70B Turbo</option>
                      <option value="mistralai/Mixtral-8x22B-Instruct-v0.1">Mixtral 8x22B</option>
                      <option value="Qwen/Qwen2.5-72B-Instruct-Turbo">Qwen 2.5 72B Turbo</option>
                      <option value="deepseek-ai/DeepSeek-V3">DeepSeek V3</option>
                    </select>
                  ) : llmSettings.provider === 'groq' ? (
                    <select
                      value={llmSettings.model}
                      onChange={(e) => setLlmSettings(prev => ({ ...prev, model: e.target.value }))}
                      className="input"
                    >
                      <option value="llama-3.3-70b-versatile">Llama 3.3 70B</option>
                      <option value="llama-3.1-70b-versatile">Llama 3.1 70B</option>
                      <option value="mixtral-8x7b-32768">Mixtral 8x7B</option>
                      <option value="gemma2-9b-it">Gemma 2 9B</option>
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={llmSettings.model}
                      onChange={(e) => setLlmSettings(prev => ({ ...prev, model: e.target.value }))}
                      placeholder="model-name"
                      className="input"
                    />
                  )}
                </div>
              </div>
              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => setShowLLMModal(false)}
                  className="btn btn-secondary flex-1"
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    setLlmSettings(prev => ({ ...prev, isConfigured: !!prev.apiKey }));
                    setShowLLMModal(false);
                  }}
                  className="btn btn-primary flex-1"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="app-footer">
        Powered by LangChain · LangGraph · Neo4j · ChromaDB
      </footer>
    </div>
  );
}

// ===== BaselineTab =====
function BaselineTab({ 
  getLLMConfig, 
  state, 
  setState 
}: { 
  getLLMConfig: () => LLMConfig | undefined;
  state: BaselineState;
  setState: React.Dispatch<React.SetStateAction<BaselineState>>;
}) {
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    const config = getLLMConfig();
    if (!config) {
      alert('먼저 API Key를 설정해주세요.');
      return;
    }
    setLoading(true);
    try {
      const res = await api.baselineChat(state.question, config);
      setState(prev => ({ ...prev, answer: res.answer }));
    } catch (e) {
      setState(prev => ({ ...prev, answer: '오류가 발생했습니다: ' + (e instanceof Error ? e.message : '알 수 없는 오류') }));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>Baseline Chat</h2>
          <p className="text-sm mt-1" style={{ color: 'var(--color-text-tertiary)' }}>RAG 없이 LLM의 기본 지식만으로 답변</p>
        </div>
        <div className="card-body">
          <textarea
            value={state.question}
            onChange={(e) => setState(prev => ({ ...prev, question: e.target.value }))}
            placeholder="질문을 입력하세요..."
            className="textarea h-32 mb-4"
          />
          <button
            onClick={handleAsk}
            disabled={loading || !state.question.trim()}
            className="btn btn-primary w-full"
          >
            {loading ? (
              <><span className="spinner"></span> 생성 중...</>
            ) : (
              '💬 질문하기'
            )}
          </button>
        </div>
      </div>
      <div className="card min-w-0">
        <div className="card-header">
          <h2 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>Response</h2>
        </div>
        <div className="card-body min-w-0">
          {state.answer ? (
            <div className="result-box whitespace-pre-wrap break-words overflow-auto max-h-96" style={{ color: 'var(--color-text-secondary)' }}>
              {state.answer}
            </div>
          ) : (
            <div className="result-box flex items-center justify-center" style={{ color: 'var(--color-text-muted)' }}>
              질문을 입력하면 답변이 여기에 표시됩니다
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ===== VectorRAGTab =====
function VectorRAGTab({ 
  getLLMConfig, 
  state, 
  setState,
  onRefresh
}: { 
  getLLMConfig: () => LLMConfig | undefined;
  state: VectorRAGState;
  setState: React.Dispatch<React.SetStateAction<VectorRAGState>>;
  onRefresh: () => void;
}) {
  const [loading, setLoading] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [building, setBuilding] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const res = await api.uploadFile(file);
      if (res.success && (res.text || res.content)) {
        setState(prev => ({ ...prev, document: res.text || res.content || '' }));
      }
    } catch (err) {
      alert('파일 업로드 실패: ' + (err instanceof Error ? err.message : '알 수 없는 오류'));
    }
  };

  const handleIndex = async () => {
    const config = getLLMConfig();
    if (!config) {
      alert('먼저 API Key를 설정해주세요.');
      return;
    }
    setIndexing(true);
    try {
      const res = await api.indexDocument(state.document, state.chunkConfig);
      setState(prev => ({ ...prev, chunks: res.chunks || res.chunk_details || [], showChunks: true }));
      onRefresh();
    } catch (e) {
      alert('인덱싱 실패: ' + (e instanceof Error ? e.message : '알 수 없는 오류'));
    } finally {
      setIndexing(false);
    }
  };

  const handleSearch = async () => {
    const config = getLLMConfig();
    if (!config) {
      alert('먼저 API Key를 설정해주세요.');
      return;
    }
    setLoading(true);
    try {
      const res = await api.vectorSearch(state.question, config);
      setState(prev => ({ ...prev, answer: res.answer, sources: res.sources || [] }));
    } catch (e) {
      setState(prev => ({ ...prev, answer: '오류가 발생했습니다: ' + (e instanceof Error ? e.message : '알 수 없는 오류') }));
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (!confirm('Vector DB의 모든 데이터를 삭제하시겠습니까?')) return;
    setClearing(true);
    try {
      await api.vectorClear();
      setState(prev => ({ ...prev, document: '', chunks: [], answer: '', sources: [] }));
      // Reset file input to allow re-uploading the same file
      if (fileInputRef.current) fileInputRef.current.value = '';
      onRefresh();
    } catch (e) {
      alert('초기화 실패: ' + (e instanceof Error ? e.message : '알 수 없는 오류'));
    } finally {
      setClearing(false);
    }
  };

  const handleBuildVectorDB = async () => {
    const config = getLLMConfig();
    if (!config) {
      alert('먼저 API Key를 설정해주세요.');
      return;
    }
    if (!state.document.trim()) {
      alert('먼저 문서를 입력해주세요.');
      return;
    }
    setBuilding(true);
    try {
      const res = await api.vectorBuild([state.document], config, state.chunkingStrategy);
      if (res.success) {
        setState(prev => ({ 
          ...prev, 
          chunks: res.chunk_details || [], 
          showChunks: true,
          chunkConfig: { 
            ...prev.chunkConfig, 
            strategy: res.chunking_config?.strategy || state.chunkingStrategy 
          }
        }));
        onRefresh();
        const strategyLabel = state.chunkingStrategy === 'semantic' ? '🧠 정밀 분석' : '⚡ 일반 분할';
        alert(`✅ Vector DB 구축 완료! (${strategyLabel})\n${res.chunks_added || 0}개의 청크가 저장되었습니다.`);
      } else {
        alert('Vector DB 구축 실패: ' + (res.error || '알 수 없는 오류'));
      }
    } catch (e) {
      alert('Vector DB 구축 실패: ' + (e instanceof Error ? e.message : '알 수 없는 오류'));
    } finally {
      setBuilding(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="space-y-6">
        {/* Document Input */}
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>Document Input</h2>
              <p className="text-sm mt-1" style={{ color: 'var(--color-text-tertiary)' }}>문서를 입력하거나 파일을 업로드하세요</p>
            </div>
            <button onClick={handleClear} disabled={clearing} className="btn btn-danger text-xs">
              {clearing ? '...' : '🗑️ 초기화'}
            </button>
          </div>
          <div className="card-body space-y-4">
            <label className="file-upload-area block cursor-pointer">
              <input ref={fileInputRef} type="file" accept=".pdf,.txt,.doc,.docx" onChange={handleFileUpload} className="hidden" />
              <div className="text-center">
                <span className="text-2xl">📁</span>
                <p className="text-sm mt-2" style={{ color: 'var(--color-text-secondary)' }}>파일 첨부 (PDF, TXT, DOC)</p>
              </div>
            </label>
            <textarea
              value={state.document}
              onChange={(e) => setState(prev => ({ ...prev, document: e.target.value }))}
              placeholder="또는 직접 텍스트를 입력하세요..."
              className="textarea h-32"
            />
            {/* Chunking Strategy Selector */}
            <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--color-bg-tertiary)', border: '1px solid var(--color-border)' }}>
              <label className="text-xs font-medium block mb-2" style={{ color: 'var(--color-text-tertiary)' }}>
                📦 청킹 방식 선택
              </label>
              <select
                value={state.chunkingStrategy}
                onChange={(e) => setState(prev => ({ ...prev, chunkingStrategy: e.target.value as 'recursive' | 'semantic' }))}
                className="input w-full"
                style={{ cursor: 'pointer' }}
              >
                <option value="recursive">⚡ 일반 분할 (Recursive)</option>
                <option value="semantic">🧠 정밀 분석 (Semantic AI)</option>
              </select>
              {state.chunkingStrategy === 'semantic' && (
                <p className="text-xs mt-2 flex items-center gap-1" style={{ color: 'var(--color-text-warning, #f59e0b)' }}>
                  ⚠️ AI가 문맥을 분석하여 의미 단위로 분할합니다. 시간이 조금 더 소요됩니다.
                </p>
              )}
            </div>

            {/* Chunk Size / Overlap (only for recursive) */}
            {state.chunkingStrategy === 'recursive' && (
              <div className="flex gap-4 items-center">
                <div className="flex-1">
                  <label className="text-xs font-medium" style={{ color: 'var(--color-text-tertiary)' }}>Chunk Size</label>
                  <input
                    type="number"
                    value={state.chunkConfig.chunk_size}
                    onChange={(e) => setState(prev => ({ ...prev, chunkConfig: { ...prev.chunkConfig, chunk_size: parseInt(e.target.value) || 500 } }))}
                    className="input mt-1"
                  />
                </div>
                <div className="flex-1">
                  <label className="text-xs font-medium" style={{ color: 'var(--color-text-tertiary)' }}>Overlap</label>
                  <input
                    type="number"
                    value={state.chunkConfig.chunk_overlap}
                    onChange={(e) => setState(prev => ({ ...prev, chunkConfig: { ...prev.chunkConfig, chunk_overlap: parseInt(e.target.value) || 50 } }))}
                    className="input mt-1"
                  />
                </div>
              </div>
            )}
            <button
              onClick={handleIndex}
              disabled={indexing || !state.document.trim()}
              className="btn btn-primary w-full"
            >
              {indexing ? <><span className="spinner"></span> 인덱싱 중...</> : '📊 문서 인덱싱'}
            </button>
            <button
              onClick={handleBuildVectorDB}
              disabled={building || !state.document.trim()}
              className="btn btn-secondary w-full mt-2"
              style={{ backgroundColor: '#10B981', color: 'white' }}
            >
              {building ? (
                <>
                  <span className="spinner"></span>
                  {state.chunkingStrategy === 'semantic' ? ' AI가 문맥을 분석 중입니다...' : ' 구축 중...'}
                </>
              ) : (
                '🗄️ Vector DB 구축 (Build Vector DB)'
              )}
            </button>
          </div>
        </div>

        {/* Chunks */}
        {state.chunks.length > 0 && (
          <div className="card">
            <div className="card-body">
              <CollapsibleSection title="생성된 청크" count={state.chunks.length} colorClass="text-blue-600">
                <div className="space-y-2">
                  {state.chunks.map((chunk, i) => (
                    <div key={i} className="p-3 rounded-lg text-sm break-words" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                      <div className="flex justify-between items-center mb-1">
                        <span className="badge badge-info">Chunk {i + 1}</span>
                        <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>{chunk.content?.length || 0} chars</span>
                      </div>
                      <p style={{ color: 'var(--color-text-secondary)' }}>{chunk.content?.substring(0, 150)}...</p>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            </div>
          </div>
        )}

        {/* Search */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>Vector Search</h2>
          </div>
          <div className="card-body">
            <textarea
              value={state.question}
              onChange={(e) => setState(prev => ({ ...prev, question: e.target.value }))}
              placeholder="질문을 입력하세요..."
              className="textarea h-20 mb-4"
            />
            <button
              onClick={handleSearch}
              disabled={loading || !state.question.trim()}
              className="btn btn-primary w-full"
            >
              {loading ? <><span className="spinner"></span> 검색 중...</> : '🔍 벡터 검색'}
            </button>
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="card min-w-0">
        <div className="card-header">
          <h2 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>Search Results</h2>
        </div>
        <div className="card-body space-y-4 min-w-0">
          {state.answer ? (
            <>
              <div className="result-box whitespace-pre-wrap break-words overflow-auto max-h-64" style={{ color: 'var(--color-text-secondary)' }}>
                {state.answer}
              </div>
              {state.sources.length > 0 && (
                <CollapsibleSection title="참조 문서" count={state.sources.length} colorClass="text-blue-600">
                  <div className="space-y-2">
                    {state.sources.map((source, i) => (
                      <div key={i} className="p-3 rounded-lg text-sm break-words" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                        <div className="flex justify-between items-center mb-1">
                          <span className="badge badge-info">Source {i + 1}</span>
                          <span className="badge badge-neutral">{(source.similarity * 100).toFixed(1)}%</span>
                        </div>
                        <p style={{ color: 'var(--color-text-secondary)' }}>{source.content}</p>
                      </div>
                    ))}
                  </div>
                </CollapsibleSection>
              )}
            </>
          ) : (
            <div className="result-box flex items-center justify-center h-32" style={{ color: 'var(--color-text-muted)' }}>
              검색 결과가 여기에 표시됩니다
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ===== GraphRAGTab =====
function GraphRAGTab({ 
  getLLMConfig, 
  state, 
  setState,
  onRefresh
}: { 
  getLLMConfig: () => LLMConfig | undefined;
  state: GraphRAGState;
  setState: React.Dispatch<React.SetStateAction<GraphRAGState>>;
  onRefresh: () => void;
}) {
  const [loading, setLoading] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [buildingGraph, setBuildingGraph] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const res = await api.uploadFile(file);
      if (res.success && (res.text || res.content)) {
        setState(prev => ({ ...prev, document: res.text || res.content || '' }));
      }
    } catch (err) {
      alert('파일 업로드 실패: ' + (err instanceof Error ? err.message : '알 수 없는 오류'));
    }
  };

  const handleClear = async () => {
    if (!confirm('Graph DB의 모든 데이터를 삭제하시겠습니까?')) return;
    setClearing(true);
    try {
      await api.clearGraph();
      setState(prev => ({ ...prev, document: '', entities: [], relationships: [], answer: '', steps: [], refNodes: [], refRelationships: [] }));
      // Reset file input to allow re-uploading the same file
      if (fileInputRef.current) fileInputRef.current.value = '';
      onRefresh();
    } catch (e) {
      alert('초기화 실패: ' + (e instanceof Error ? e.message : '알 수 없는 오류'));
    } finally {
      setClearing(false);
    }
  };

  const handleExtract = async () => {
    const config = getLLMConfig();
    if (!config) {
      alert('먼저 API Key를 설정해주세요.');
      return;
    }
    setExtracting(true);
    try {
      const res = await api.extractAndStore(state.document, config);
      // 부분 결과가 있으면 항상 표시 (success 여부와 무관하게)
      const entities = res.entities || [];
      const relationships = res.relationships || [];
      
      if (entities.length > 0 || relationships.length > 0) {
        setState(prev => ({ ...prev, entities, relationships }));
        onRefresh();
      }
      
      // 에러가 있으면 알림 (부분 결과와 함께)
      if (!res.success && res.error) {
        if (entities.length > 0) {
          alert(`⚠️ 부분 추출 완료: ${entities.length}개 엔티티, ${relationships.length}개 관계\n\n오류: ${res.error}`);
        } else {
          alert('추출 실패: ' + res.error);
        }
        return;
      }
      
      // 성공 시 알림
      if (entities.length === 0 && relationships.length === 0) {
        alert('추출된 엔티티/관계가 없습니다. 다른 문서를 시도해보세요.');
      }
    } catch (e) {
      // 타임아웃 에러 특별 처리
      const errorMsg = e instanceof Error ? e.message : '알 수 없는 오류';
      if (errorMsg.includes('aborted') || errorMsg.includes('abort')) {
        alert('⏱️ 요청 시간 초과 (15분)\n\n긴 문서는 처리 시간이 오래 걸릴 수 있습니다.\n더 짧은 문서로 시도하거나, 백엔드 로그를 확인해주세요.');
      } else {
        alert('추출 실패: ' + errorMsg);
      }
    } finally {
      setExtracting(false);
    }
  };

  const handleBuildGraphDB = async () => {
    const config = getLLMConfig();
    if (!config) {
      alert('먼저 API Key를 설정해주세요.');
      return;
    }
    if (state.entities.length === 0 && state.relationships.length === 0) {
      alert('먼저 엔티티/관계를 추출해주세요.');
      return;
    }
    setBuildingGraph(true);
    try {
      const res = await api.build(state.entities, state.relationships);
      if (res.success) {
        onRefresh();
        alert('✅ Graph DB 구축 완료! ' + (res.nodesCreated || 0) + '개 노드, ' + (res.relationshipsCreated || 0) + '개 관계가 저장되었습니다.');
      } else {
        alert('Graph DB 구축 실패: ' + (res.error || '알 수 없는 오류'));
      }
    } catch (e) {
      alert('Graph DB 구축 실패: ' + (e instanceof Error ? e.message : '알 수 없는 오류'));
    } finally {
      setBuildingGraph(false);
    }
  };

  const handleSearch = async () => {
    const config = getLLMConfig();
    if (!config) {
      alert('먼저 API Key를 설정해주세요.');
      return;
    }
    setLoading(true);
    try {
      const res = await api.graphSearch(state.question, config);
      setState(prev => ({ 
        ...prev, 
        answer: res.answer, 
        steps: res.retrievalSteps || [], 
        refNodes: res.nodes || [],
        refRelationships: res.relationships || []
      }));
    } catch (e) {
      setState(prev => ({ ...prev, answer: '오류가 발생했습니다: ' + (e instanceof Error ? e.message : '알 수 없는 오류') }));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="space-y-6">
        {/* Document Input */}
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>Document Input</h2>
              <p className="text-sm mt-1" style={{ color: 'var(--color-text-tertiary)' }}>지식 그래프로 변환할 문서</p>
            </div>
            <button onClick={handleClear} disabled={clearing} className="btn btn-danger text-xs">
              {clearing ? '...' : '🗑️ 초기화'}
            </button>
          </div>
          <div className="card-body space-y-4">
            <label className="file-upload-area block cursor-pointer">
              <input ref={fileInputRef} type="file" accept=".pdf,.txt,.doc,.docx" onChange={handleFileUpload} className="hidden" />
              <div className="text-center">
                <span className="text-2xl">📁</span>
                <p className="text-sm mt-2" style={{ color: 'var(--color-text-secondary)' }}>파일 첨부 (PDF, TXT, DOC)</p>
              </div>
            </label>
            <textarea
              value={state.document}
              onChange={(e) => setState(prev => ({ ...prev, document: e.target.value }))}
              placeholder="또는 직접 텍스트를 입력하세요..."
              className="textarea h-32"
            />
            <button
              onClick={handleExtract}
              disabled={extracting || !state.document.trim()}
              className="btn btn-primary w-full"
            >
              {extracting ? <><span className="spinner"></span> 추출 중...</> : '🔗 엔티티/관계 추출'}
            </button>
            <button
              onClick={handleBuildGraphDB}
              disabled={buildingGraph || (state.entities.length === 0 && state.relationships.length === 0)}
              className="btn btn-secondary w-full mt-2"
              style={{ backgroundColor: '#8B5CF6', color: 'white' }}
            >
              {buildingGraph ? <><span className="spinner"></span> 구축 중...</> : '🌐 Graph DB 구축 (Build Graph DB)'}
            </button>
          </div>
        </div>

        {/* Extracted Entities & Relations */}
        {(state.entities.length > 0 || state.relationships.length > 0) && (
          <div className="card">
            <div className="card-body space-y-4">
              {state.entities.length > 0 && (
                <CollapsibleSection title="추출된 엔티티" count={state.entities.length} colorClass="text-purple-600">
                  <div className="flex flex-wrap gap-2">
                    {state.entities.map((entity, i) => (
                      <span key={i} className="badge badge-info">{entity.name} ({entity.type})</span>
                    ))}
                  </div>
                </CollapsibleSection>
              )}
              {state.relationships.length > 0 && (
                <CollapsibleSection title="추출된 관계" count={state.relationships.length} colorClass="text-purple-600">
                  <div className="space-y-1">
                    {state.relationships.map((rel, i) => (
                      <div key={i} className="text-sm p-2 rounded" style={{ backgroundColor: 'var(--color-bg-tertiary)', color: 'var(--color-text-secondary)' }}>
                        <span className="text-blue-600">{rel.source}</span>
                        <span className="text-purple-600 font-medium"> → {rel.relation || rel.type} → </span>
                        <span className="text-blue-600">{rel.target}</span>
                      </div>
                    ))}
                  </div>
                </CollapsibleSection>
              )}
            </div>
          </div>
        )}

        {/* Search */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>Graph Search</h2>
          </div>
          <div className="card-body">
            <textarea
              value={state.question}
              onChange={(e) => setState(prev => ({ ...prev, question: e.target.value }))}
              placeholder="질문을 입력하세요..."
              className="textarea h-20 mb-4"
            />
            <button
              onClick={handleSearch}
              disabled={loading || !state.question.trim()}
              className="btn btn-primary w-full"
            >
              {loading ? <><span className="spinner"></span> 검색 중...</> : '🔗 그래프 검색'}
            </button>
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="card min-w-0">
        <div className="card-header">
          <h2 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>Search Results</h2>
        </div>
        <div className="card-body space-y-4 min-w-0">
          {state.answer ? (
            <>
              <div className="result-box whitespace-pre-wrap break-words overflow-auto max-h-64" style={{ color: 'var(--color-text-secondary)' }}>
                {state.answer}
              </div>
              
              {/* Graph Visualization */}
              {(state.refNodes.length > 0 || state.refRelationships.length > 0) && (
                <CollapsibleSection 
                  title="참조된 그래프" 
                  count={state.refNodes.length} 
                  colorClass="text-purple-600"
                  defaultOpen={true}
                >
                  <div className="h-48 border rounded-lg overflow-hidden" style={{ borderColor: 'var(--color-border)' }}>
                    <GraphVisualization 
                      nodes={state.refNodes} 
                      relationships={state.refRelationships} 
                    />
                  </div>
                  <div className="mt-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                    노드: {state.refNodes.length}개, 관계: {state.refRelationships.length}개
                  </div>
                </CollapsibleSection>
              )}

              {/* Referenced Nodes */}
              {state.refNodes.length > 0 && (
                <CollapsibleSection title="참조된 노드" count={state.refNodes.length} colorClass="text-blue-600">
                  <div className="space-y-2">
                    {state.refNodes.map((node, i) => (
                      <div key={i} className="p-2 rounded text-sm break-words" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-medium text-blue-600">{String(node.properties?.name || `Node ${node.id}`)}</span>
                          <span className="badge badge-neutral">{node.labels.join(', ')}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CollapsibleSection>
              )}

              {/* Referenced Relationships */}
              {state.refRelationships.length > 0 && (
                <CollapsibleSection title="참조된 관계" count={state.refRelationships.length} colorClass="text-purple-600">
                  <div className="space-y-1">
                    {state.refRelationships.map((rel, i) => {
                      const sourceNode = state.refNodes.find(n => n.id === rel.startNodeId);
                      const targetNode = state.refNodes.find(n => n.id === rel.endNodeId);
                      return (
                        <div key={i} className="text-sm p-2 rounded break-words" style={{ backgroundColor: 'var(--color-bg-tertiary)', color: 'var(--color-text-secondary)' }}>
                          <span className="text-blue-600">{String(sourceNode?.properties?.name || `Node ${rel.startNodeId}`)}</span>
                          <span className="text-purple-600 font-medium"> → {rel.type} → </span>
                          <span className="text-blue-600">{String(targetNode?.properties?.name || `Node ${rel.endNodeId}`)}</span>
                        </div>
                      );
                    })}
                  </div>
                </CollapsibleSection>
              )}

              {/* Search Steps */}
              {state.steps.length > 0 && (
                <CollapsibleSection title="검색 과정" count={state.steps.length} colorClass="text-gray-600">
                  <div className="space-y-2">
                    {state.steps.map((step, i) => (
                      <div key={i} className="p-3 rounded text-sm break-words" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                        <div className="font-medium text-purple-600 mb-1">{step.step}</div>
                        <p style={{ color: 'var(--color-text-tertiary)' }}>{step.result}</p>
                      </div>
                    ))}
                  </div>
                </CollapsibleSection>
              )}
            </>
          ) : (
            <div className="result-box flex items-center justify-center h-32" style={{ color: 'var(--color-text-muted)' }}>
              검색 결과가 여기에 표시됩니다
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ===== CompareTab =====
function CompareTab({ 
  getLLMConfig, 
  state, 
  setState 
}: { 
  getLLMConfig: () => LLMConfig | undefined;
  state: CompareState;
  setState: React.Dispatch<React.SetStateAction<CompareState>>;
}) {
  const [loading, setLoading] = useState(false);

  const handleCompare = async () => {
    const config = getLLMConfig();
    if (!config) {
      alert('먼저 API Key를 설정해주세요.');
      return;
    }
    setLoading(true);
    try {
      const res = await api.compare(state.question, config);
      setState(prev => ({ ...prev, results: res }));
    } catch (e) {
      alert('비교 실패: ' + (e instanceof Error ? e.message : '알 수 없는 오류'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Input */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>Comparison Test</h2>
          <p className="text-sm mt-1" style={{ color: 'var(--color-text-tertiary)' }}>Baseline, Vector RAG, Graph RAG 세 가지 방식을 동시에 비교</p>
        </div>
        <div className="card-body">
          <div className="flex gap-4">
            <textarea
              value={state.question}
              onChange={(e) => setState(prev => ({ ...prev, question: e.target.value }))}
              placeholder="비교할 질문을 입력하세요..."
              className="textarea flex-1 h-16"
            />
            <button
              onClick={handleCompare}
              disabled={loading || !state.question.trim()}
              className="btn btn-primary whitespace-nowrap px-8"
            >
              {loading ? <><span className="spinner"></span> 비교 중...</> : '⚖️ 비교 실행'}
            </button>
          </div>
        </div>
      </div>

      {/* Results */}
      {state.results && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Baseline */}
          <div className="card min-w-0">
            <div className="card-header flex items-center justify-between">
              <h3 className="font-semibold" style={{ color: 'var(--color-text-primary)' }}>💬 Baseline</h3>
              <span className="badge badge-neutral">{state.results.baseline.latency_ms}ms</span>
            </div>
            <div className="card-body min-w-0">
              <div className="result-box whitespace-pre-wrap break-words overflow-auto max-h-64 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                {state.results.baseline.answer}
              </div>
            </div>
          </div>

          {/* Vector RAG */}
          <div className="card min-w-0">
            <div className="card-header flex items-center justify-between">
              <h3 className="font-semibold" style={{ color: 'var(--color-text-primary)' }}>📊 Vector RAG</h3>
              <span className="badge badge-neutral">{state.results.vector_rag.latency_ms}ms</span>
            </div>
            <div className="card-body space-y-4 min-w-0">
              <div className="result-box whitespace-pre-wrap break-words overflow-auto max-h-64 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                {state.results.vector_rag.answer}
              </div>
              {state.results.vector_rag.sources && state.results.vector_rag.sources.length > 0 && (
                <CollapsibleSection title="참조 문서" count={state.results.vector_rag.sources.length} colorClass="text-blue-600">
                  <div className="space-y-2">
                    {state.results.vector_rag.sources.map((source, i) => (
                      <div key={i} className="p-2 rounded text-xs break-words" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                        <div className="flex justify-between mb-1">
                          <span className="font-medium text-blue-600">Source {i + 1}</span>
                          <span style={{ color: 'var(--color-text-muted)' }}>{(source.similarity * 100).toFixed(1)}%</span>
                        </div>
                        <p style={{ color: 'var(--color-text-secondary)' }}>{source.content}</p>
                      </div>
                    ))}
                  </div>
                </CollapsibleSection>
              )}
            </div>
          </div>

          {/* Graph RAG */}
          <div className="card min-w-0">
            <div className="card-header flex items-center justify-between">
              <h3 className="font-semibold" style={{ color: 'var(--color-text-primary)' }}>🔗 Graph RAG</h3>
              <span className="badge badge-neutral">{state.results.graph_rag.latency_ms}ms</span>
            </div>
            <div className="card-body space-y-4 min-w-0">
              <div className="result-box whitespace-pre-wrap break-words overflow-auto max-h-64 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                {state.results.graph_rag.answer}
              </div>
              
              {/* Graph Visualization */}
              {state.results.graph_rag.nodes && state.results.graph_rag.nodes.length > 0 && (
                <CollapsibleSection 
                  title="참조된 그래프" 
                  count={state.results.graph_rag.nodes.length} 
                  colorClass="text-purple-600"
                  defaultOpen={true}
                >
                  <div className="h-40 border rounded-lg overflow-hidden mb-2" style={{ borderColor: 'var(--color-border)' }}>
                    <GraphVisualization 
                      nodes={state.results.graph_rag.nodes} 
                      relationships={state.results.graph_rag.relationships || []} 
                    />
                  </div>
                  <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                    노드: {state.results.graph_rag.nodes.length}개, 관계: {state.results.graph_rag.relationships?.length || 0}개
                  </div>
                </CollapsibleSection>
              )}

              {/* Referenced Nodes */}
              {state.results.graph_rag.nodes && state.results.graph_rag.nodes.length > 0 && (
                <CollapsibleSection title="참조된 노드" count={state.results.graph_rag.nodes.length} colorClass="text-blue-600">
                  <div className="space-y-2">
                    {state.results.graph_rag.nodes.map((node, i) => (
                      <div key={i} className="p-2 rounded text-xs break-words" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-blue-600">{String(node.properties?.name || `Node ${node.id}`)}</span>
                          <span className="badge badge-neutral text-xs">{node.labels.join(', ')}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CollapsibleSection>
              )}

              {/* Referenced Relationships */}
              {state.results.graph_rag.relationships && state.results.graph_rag.relationships.length > 0 && (
                <CollapsibleSection title="참조된 관계" count={state.results.graph_rag.relationships.length} colorClass="text-purple-600">
                  <div className="space-y-1">
                    {state.results.graph_rag.relationships.map((rel, i) => {
                      const sourceNode = state.results?.graph_rag.nodes?.find(n => n.id === rel.startNodeId);
                      const targetNode = state.results?.graph_rag.nodes?.find(n => n.id === rel.endNodeId);
                      return (
                        <div key={i} className="text-xs p-2 rounded break-words" style={{ backgroundColor: 'var(--color-bg-tertiary)', color: 'var(--color-text-secondary)' }}>
                          <span className="text-blue-600">{String(sourceNode?.properties?.name || `Node ${rel.startNodeId}`)}</span>
                          <span className="text-purple-600 font-medium"> → {rel.type} → </span>
                          <span className="text-blue-600">{String(targetNode?.properties?.name || `Node ${rel.endNodeId}`)}</span>
                        </div>
                      );
                    })}
                  </div>
                </CollapsibleSection>
              )}

              {/* Search Steps */}
              {state.results.graph_rag.steps && state.results.graph_rag.steps.length > 0 && (
                <CollapsibleSection title="검색 과정" count={state.results.graph_rag.steps.length} colorClass="text-gray-600">
                  <div className="space-y-2">
                    {state.results.graph_rag.steps.map((step, i) => (
                      <div key={i} className="p-2 rounded text-xs break-words" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                        <span className="font-medium text-purple-600">{step.step}</span>
                        <p className="mt-1" style={{ color: 'var(--color-text-tertiary)' }}>{step.result}</p>
                      </div>
                    ))}
                  </div>
                </CollapsibleSection>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ===== GraphExplorerTab =====
function GraphExplorerTab({ 
  state, 
  setState 
}: { 
  state: ExplorerState;
  setState: React.Dispatch<React.SetStateAction<ExplorerState>>;
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const presetQueries = [
    { id: 'all_nodes', name: '모든 노드', query: 'MATCH (n) RETURN n LIMIT 50', description: '그래프의 모든 노드를 최대 50개까지 조회' },
    { id: 'all_relationships', name: '모든 관계', query: 'MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50', description: '모든 노드 간의 관계를 조회' },
    { id: 'node_types', name: '노드 타입 통계', query: 'MATCH (n) RETURN labels(n) AS type, count(*) AS count ORDER BY count DESC', description: '노드 타입별 개수 집계' },
    { id: 'relationship_types', name: '관계 타입 통계', query: 'MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC', description: '관계 타입별 개수 집계' },
    { id: 'most_connected', name: '연결 많은 노드', query: 'MATCH (n)-[r]-() RETURN n.name AS name, labels(n) AS type, count(r) AS connections ORDER BY connections DESC LIMIT 10', description: '연결이 가장 많은 상위 10개 노드' },
    { id: 'isolated_nodes', name: '고립 노드', query: 'MATCH (n) WHERE NOT (n)--() RETURN n LIMIT 20', description: '연결되지 않은 고립 노드' },
    { id: 'paths', name: '경로 탐색', query: 'MATCH path = (n)-[*1..2]-(m) WHERE n <> m RETURN path LIMIT 25', description: '2단계 이내의 경로 탐색' },
    { id: 'search_name', name: '이름 검색', query: "MATCH (n) WHERE toLower(n.name) CONTAINS 'ai' RETURN n LIMIT 20", description: '이름에 키워드가 포함된 노드 검색' },
  ];

  const handlePresetSelect = (presetId: string) => {
    const preset = presetQueries.find(p => p.id === presetId);
    if (preset) {
      setState(prev => ({ ...prev, query: preset.query, selectedPreset: presetId }));
    }
  };

  const handleExecute = async () => {
    if (!state.query.trim()) return;
    setLoading(true);
    setError('');
    setState(prev => ({ ...prev, results: null }));
    try {
      const response = await api.query(state.query);
      if (response.success) {
        const nodes: GraphNode[] = response.nodes || [];
        const relationships: GraphRelationship[] = response.relationships || [];
        setState(prev => ({ ...prev, results: { records: response.records, nodes, relationships } }));
      } else {
        setError(response.error || '쿼리 실행 실패');
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : '알 수 없는 오류');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-body">
          <h2 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>Graph Explorer</h2>
          <p className="text-sm mt-1" style={{ color: 'var(--color-text-tertiary)' }}>Neo4j 그래프 데이터베이스를 직접 쿼리하고 시각화</p>
        </div>
      </div>

      {/* Top Row: 3 Boxes */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Preset Queries */}
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium" style={{ color: 'var(--color-text-primary)' }}>📋 Preset Queries</h3>
          </div>
          <div className="card-body">
            <div className="space-y-2 max-h-48 overflow-auto">
              {presetQueries.map(preset => (
                <button
                  key={preset.id}
                  onClick={() => handlePresetSelect(preset.id)}
                  className={`w-full px-3 py-2 text-xs rounded-lg border transition-all text-left ${
                    state.selectedPreset === preset.id
                      ? 'border-blue-400 bg-blue-50 text-blue-700'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  {preset.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Query Input */}
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium" style={{ color: 'var(--color-text-primary)' }}>✏️ Cypher Query</h3>
          </div>
          <div className="card-body">
            <textarea
              value={state.query}
              onChange={(e) => setState(prev => ({ ...prev, query: e.target.value, selectedPreset: '' }))}
              placeholder="MATCH (n) RETURN n LIMIT 10"
              className="textarea h-24 font-mono text-sm mb-3"
            />
            <button
              onClick={handleExecute}
              disabled={loading || !state.query.trim()}
              className="btn btn-primary w-full"
            >
              {loading ? <><span className="spinner"></span> 실행 중...</> : '▶ 실행'}
            </button>
          </div>
        </div>

        {/* Results Summary */}
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium" style={{ color: 'var(--color-text-primary)' }}>📊 Results</h3>
          </div>
          <div className="card-body">
            {error && (
              <div className="p-3 rounded-lg text-sm" style={{ backgroundColor: '#fef2f2', color: '#dc2626' }}>
                {error}
              </div>
            )}
            {state.results && (
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span style={{ color: 'var(--color-text-tertiary)' }}>Records</span>
                  <span className="font-medium" style={{ color: 'var(--color-text-primary)' }}>{state.results.records.length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span style={{ color: 'var(--color-text-tertiary)' }}>Nodes</span>
                  <span className="font-medium" style={{ color: 'var(--color-text-primary)' }}>{state.results.nodes.length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span style={{ color: 'var(--color-text-tertiary)' }}>Relationships</span>
                  <span className="font-medium" style={{ color: 'var(--color-text-primary)' }}>{state.results.relationships.length}</span>
                </div>
              </div>
            )}
            {!state.results && !error && (
              <div className="text-sm text-center py-4" style={{ color: 'var(--color-text-muted)' }}>
                쿼리를 실행하면 결과가 표시됩니다
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Graph Visualization */}
      {state.results && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium" style={{ color: 'var(--color-text-primary)' }}>🌐 Graph Visualization</h3>
          </div>
          <div className="card-body">
            <div className="h-[500px] border rounded-lg overflow-hidden" style={{ borderColor: 'var(--color-border)' }}>
              <GraphVisualization nodes={state.results.nodes} relationships={state.results.relationships} />
            </div>
            <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
              <CollapsibleSection title="노드 목록" count={state.results.nodes.length} colorClass="text-blue-600">
                <div className="space-y-1">
                  {state.results.nodes.map((node, i) => (
                    <div key={i} className="text-xs p-2 rounded break-words" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                      <span className="font-medium text-blue-600">{String(node.properties?.name || `Node ${node.id}`)}</span>
                      <span className="ml-2 badge badge-neutral">{node.labels.join(', ')}</span>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
              <CollapsibleSection title="관계 목록" count={state.results.relationships.length} colorClass="text-purple-600">
                <div className="space-y-1">
                  {state.results.relationships.map((rel, i) => {
                    const source = state.results?.nodes.find(n => n.id === rel.startNodeId);
                    const target = state.results?.nodes.find(n => n.id === rel.endNodeId);
                    return (
                      <div key={i} className="text-xs p-2 rounded break-words" style={{ backgroundColor: 'var(--color-bg-tertiary)', color: 'var(--color-text-secondary)' }}>
                        <span className="text-blue-600">{String(source?.properties?.name || rel.startNodeId)}</span>
                        <span className="text-purple-600 font-medium"> → {rel.type} → </span>
                        <span className="text-blue-600">{String(target?.properties?.name || rel.endNodeId)}</span>
                      </div>
                    );
                  })}
                </div>
              </CollapsibleSection>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
