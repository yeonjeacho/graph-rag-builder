"""
Graph Retriever Module - LangGraph-based Graph RAG Pipeline
Implements a multi-step retrieval pipeline using LangGraph
"""
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from config import get_settings
from neo4j_service import Neo4jService


# ============== State Definition ==============

class ExtractedEntity(TypedDict):
    """Entity extracted from question"""
    name: str
    type: str
    confidence: float
    matchedNodeId: Optional[int]
    matchedNodeName: Optional[str]


class RetrievalStep(TypedDict):
    """Step information for debugging"""
    step: str
    description: str
    count: Optional[int]


class GraphRAGState(TypedDict):
    """State for the Graph RAG pipeline"""
    question: str
    extracted_entities: List[ExtractedEntity]
    matched_nodes: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    context: str
    answer: str
    retrieval_steps: List[RetrievalStep]
    error: Optional[str]


# ============== Configuration ==============

RETRIEVAL_CONFIG = {
    "max_entities": 3,
    "max_matched_nodes": 5,
    "max_neighbor_nodes": 10,
    "max_relationships": 15,
    "hops": 1,
    "min_confidence": 0.6
}


# ============== Prompts ==============

ENTITY_EXTRACTION_PROMPT = """You are an entity extraction expert for knowledge graph retrieval. 
Extract key entities from the question that could be found in a technical knowledge graph.

Question: "{question}"

RULES:
1. Extract 1-3 entities that are most likely to be node names in a knowledge graph
2. Extract proper nouns, technical terms, acronyms, and specific concepts
3. If a term has both acronym and full name, extract BOTH
4. DO NOT extract generic words like "information", "concept", "related", "all", etc.
5. Each entity must have a confidence score (0.0-1.0)

Return a JSON array with "name", "type", and "confidence" fields.
Example: [{{"name": "PDSCH", "type": "TechnicalTerm", "confidence": 0.9}}]

Return ONLY the JSON array, no explanations. If no specific entities found, return empty array []."""


ANSWER_GENERATION_PROMPT = """당신은 지식 그래프 기반 RAG 시스템의 답변 생성 전문가입니다.
아래 지식 그래프에서 검색된 정보를 바탕으로 사용자의 질문에 답변해 주세요.

## 사용자 질문
{question}

## 검색된 노드 정보
{node_details}

## 검색된 관계 정보
{relationship_details}

## 답변 규칙
1. 검색된 정보만을 바탕으로 답변하세요. 추측이나 외부 지식을 사용하지 마세요.
2. 노드 간의 관계를 활용하여 맥락을 설명하세요.
3. 기술 용어의 약어와 전체 이름이 있다면 함께 언급하세요.
4. 답변은 한국어로 작성하세요.
5. 간결하고 명확하게 답변하세요.

## 답변"""


# ============== LangGraph Nodes ==============

class GraphRAGPipeline:
    """LangGraph-based Graph RAG Pipeline"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize GraphRAGPipeline with optional custom LLM configuration
        
        Args:
            api_key: OpenAI API key (uses env if not provided)
            model: Model name (uses env if not provided)
            base_url: Custom API base URL (uses env if not provided)
        """
        settings = get_settings()
        
        # Use provided values or fall back to settings
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
        
        # Initialize LLM
        llm_kwargs = {
            "model": self.model,
            "temperature": 0,
            "api_key": self.api_key,
        }
        
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create workflow
        workflow = StateGraph(GraphRAGState)
        
        # Add nodes
        workflow.add_node("extract_entities", self._extract_entities)
        workflow.add_node("find_matching_nodes", self._find_matching_nodes)
        workflow.add_node("traverse_neighbors", self._traverse_neighbors)
        workflow.add_node("generate_context", self._generate_context)
        workflow.add_node("generate_answer", self._generate_answer)
        
        # Define edges
        workflow.set_entry_point("extract_entities")
        workflow.add_edge("extract_entities", "find_matching_nodes")
        workflow.add_edge("find_matching_nodes", "traverse_neighbors")
        workflow.add_edge("traverse_neighbors", "generate_context")
        workflow.add_edge("generate_context", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def _extract_entities(self, state: GraphRAGState) -> GraphRAGState:
        """Step 1: Extract entities from the question using LLM"""
        question = state["question"]
        retrieval_steps = state.get("retrieval_steps", [])
        
        try:
            prompt = ChatPromptTemplate.from_template(ENTITY_EXTRACTION_PROMPT)
            chain = prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({"question": question})
            
            # Parse JSON response
            import json
            json_str = response.strip()
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            entities = json.loads(json_str)
            
            # Filter and format entities
            extracted_entities = []
            for e in entities:
                confidence = e.get("confidence", 0.8)
                if confidence >= RETRIEVAL_CONFIG["min_confidence"]:
                    extracted_entities.append({
                        "name": e.get("name", ""),
                        "type": e.get("type", "Unknown"),
                        "confidence": confidence,
                        "matchedNodeId": None,
                        "matchedNodeName": None
                    })
            
            extracted_entities = extracted_entities[:RETRIEVAL_CONFIG["max_entities"]]
            
            retrieval_steps.append({
                "step": "엔티티 추출",
                "description": f"질문에서 {len(extracted_entities)}개의 핵심 엔티티를 추출했습니다: {', '.join(e['name'] for e in extracted_entities)}" if extracted_entities else "질문에서 구체적인 엔티티를 찾을 수 없습니다",
                "count": len(extracted_entities)
            })
            
            return {
                **state,
                "extracted_entities": extracted_entities,
                "retrieval_steps": retrieval_steps
            }
            
        except Exception as e:
            retrieval_steps.append({
                "step": "엔티티 추출",
                "description": f"엔티티 추출 실패: {str(e)}",
                "count": 0
            })
            return {
                **state,
                "extracted_entities": [],
                "retrieval_steps": retrieval_steps,
                "error": str(e)
            }
    
    def _find_matching_nodes(self, state: GraphRAGState) -> GraphRAGState:
        """Step 2: Find matching nodes in Neo4j"""
        extracted_entities = state.get("extracted_entities", [])
        retrieval_steps = state.get("retrieval_steps", [])
        
        if not extracted_entities:
            retrieval_steps.append({
                "step": "노드 매칭",
                "description": "추출된 엔티티가 없어 노드 매칭을 건너뜁니다",
                "count": 0
            })
            return {
                **state,
                "matched_nodes": [],
                "retrieval_steps": retrieval_steps
            }
        
        matched_nodes = []
        updated_entities = []
        matched_node_ids = set()
        
        for entity in extracted_entities:
            nodes = Neo4jService.find_nodes_by_name(
                entity["name"], 
                limit=RETRIEVAL_CONFIG["max_matched_nodes"]
            )
            
            matched = False
            for node in nodes:
                if node["id"] not in matched_node_ids:
                    matched_node_ids.add(node["id"])
                    matched_nodes.append(node)
                    
                    if not matched:
                        entity["matchedNodeId"] = node["id"]
                        entity["matchedNodeName"] = node["properties"].get("name", f"Node {node['id']}")
                        matched = True
            
            updated_entities.append(entity)
        
        matched_nodes = matched_nodes[:RETRIEVAL_CONFIG["max_matched_nodes"]]
        
        retrieval_steps.append({
            "step": "노드 매칭",
            "description": f"그래프에서 {len(matched_nodes)}개의 매칭되는 노드를 찾았습니다: {', '.join(n['properties'].get('name', '') for n in matched_nodes)}" if matched_nodes else "그래프에서 매칭되는 노드를 찾을 수 없습니다",
            "count": len(matched_nodes)
        })
        
        return {
            **state,
            "extracted_entities": updated_entities,
            "matched_nodes": matched_nodes,
            "retrieval_steps": retrieval_steps
        }
    
    def _traverse_neighbors(self, state: GraphRAGState) -> GraphRAGState:
        """Step 3: Traverse graph to find neighbors"""
        matched_nodes = state.get("matched_nodes", [])
        retrieval_steps = state.get("retrieval_steps", [])
        
        if not matched_nodes:
            retrieval_steps.append({
                "step": "이웃 탐색",
                "description": "매칭된 노드가 없어 이웃 탐색을 건너뜁니다",
                "count": 0
            })
            return {
                **state,
                "nodes": [],
                "relationships": [],
                "retrieval_steps": retrieval_steps
            }
        
        all_nodes = {n["id"]: n for n in matched_nodes}
        all_relationships = {}
        
        for node in matched_nodes[:RETRIEVAL_CONFIG["max_matched_nodes"]]:
            node_name = node["properties"].get("name")
            if not node_name:
                continue
            
            nodes, rels = Neo4jService.get_neighbors(
                node_name, 
                hops=RETRIEVAL_CONFIG["hops"],
                limit=RETRIEVAL_CONFIG["max_neighbor_nodes"]
            )
            
            for n in nodes:
                if n["id"] not in all_nodes:
                    all_nodes[n["id"]] = n
            
            for r in rels:
                rel_key = f"{r['startNodeId']}-{r['type']}-{r['endNodeId']}"
                if rel_key not in all_relationships:
                    all_relationships[rel_key] = r
        
        final_nodes = list(all_nodes.values())[:RETRIEVAL_CONFIG["max_neighbor_nodes"]]
        final_rels = list(all_relationships.values())[:RETRIEVAL_CONFIG["max_relationships"]]
        
        retrieval_steps.append({
            "step": "이웃 탐색",
            "description": f"{RETRIEVAL_CONFIG['hops']}-hop 탐색으로 {len(final_nodes)}개 노드와 {len(final_rels)}개 관계를 검색했습니다",
            "count": len(final_nodes) + len(final_rels)
        })
        
        return {
            **state,
            "nodes": final_nodes,
            "relationships": final_rels,
            "retrieval_steps": retrieval_steps
        }
    
    def _generate_context(self, state: GraphRAGState) -> GraphRAGState:
        """Step 4: Generate context from retrieved subgraph"""
        nodes = state.get("nodes", [])
        relationships = state.get("relationships", [])
        retrieval_steps = state.get("retrieval_steps", [])
        
        if not nodes:
            retrieval_steps.append({
                "step": "컨텍스트 생성",
                "description": "검색된 노드가 없어 컨텍스트를 생성할 수 없습니다",
                "count": 0
            })
            return {
                **state,
                "context": "",
                "retrieval_steps": retrieval_steps
            }
        
        # Build context string
        context_parts = []
        
        # Add node information
        context_parts.append("=== 노드 정보 ===")
        for node in nodes:
            name = node["properties"].get("name", f"Node {node['id']}")
            labels = ", ".join(node.get("labels", []))
            context_parts.append(f"- {name} ({labels})")
        
        # Add relationship information
        if relationships:
            context_parts.append("\n=== 관계 정보 ===")
            node_id_to_name = {n["id"]: n["properties"].get("name", f"Node {n['id']}") for n in nodes}
            
            for rel in relationships:
                start_name = node_id_to_name.get(rel["startNodeId"], f"Node {rel['startNodeId']}")
                end_name = node_id_to_name.get(rel["endNodeId"], f"Node {rel['endNodeId']}")
                context_parts.append(f"- {start_name} --[{rel['type']}]--> {end_name}")
        
        context = "\n".join(context_parts)
        
        retrieval_steps.append({
            "step": "컨텍스트 생성",
            "description": f"서브그래프로부터 RAG 컨텍스트를 생성했습니다 ({len(context)} 문자)",
            "count": len(context)
        })
        
        return {
            **state,
            "context": context,
            "retrieval_steps": retrieval_steps
        }
    
    def _generate_answer(self, state: GraphRAGState) -> GraphRAGState:
        """Step 5: Generate answer using LLM"""
        question = state["question"]
        nodes = state.get("nodes", [])
        relationships = state.get("relationships", [])
        retrieval_steps = state.get("retrieval_steps", [])
        
        if not nodes:
            retrieval_steps.append({
                "step": "답변 생성",
                "description": "검색된 정보가 없어 답변을 생성할 수 없습니다",
                "count": 0
            })
            return {
                **state,
                "answer": "죄송합니다. 질문과 관련된 정보를 지식 그래프에서 찾을 수 없습니다.",
                "retrieval_steps": retrieval_steps
            }
        
        try:
            # Prepare node details
            node_details = []
            for node in nodes:
                name = node["properties"].get("name", f"Node {node['id']}")
                labels = ", ".join(node.get("labels", []))
                node_details.append(f"- {name} (타입: {labels})")
            
            # Prepare relationship details
            node_id_to_name = {n["id"]: n["properties"].get("name", f"Node {n['id']}") for n in nodes}
            rel_details = []
            for rel in relationships:
                start_name = node_id_to_name.get(rel["startNodeId"], f"Node {rel['startNodeId']}")
                end_name = node_id_to_name.get(rel["endNodeId"], f"Node {rel['endNodeId']}")
                rel_details.append(f"- {start_name} --[{rel['type']}]--> {end_name}")
            
            # Generate answer
            prompt = ChatPromptTemplate.from_template(ANSWER_GENERATION_PROMPT)
            chain = prompt | self.llm | StrOutputParser()
            
            answer = chain.invoke({
                "question": question,
                "node_details": "\n".join(node_details) if node_details else "검색된 노드 없음",
                "relationship_details": "\n".join(rel_details) if rel_details else "검색된 관계 없음"
            })
            
            retrieval_steps.append({
                "step": "답변 생성",
                "description": "LLM이 검색된 컨텍스트를 바탕으로 답변을 생성했습니다",
                "count": len(answer)
            })
            
            return {
                **state,
                "answer": answer.strip(),
                "retrieval_steps": retrieval_steps
            }
            
        except Exception as e:
            retrieval_steps.append({
                "step": "답변 생성",
                "description": f"답변 생성 실패: {str(e)}",
                "count": 0
            })
            return {
                **state,
                "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                "retrieval_steps": retrieval_steps,
                "error": str(e)
            }
    
    def retrieve(self, question: str) -> Dict[str, Any]:
        """
        Run the full retrieval pipeline
        
        Args:
            question: The natural language question
            
        Returns:
            Dictionary with retrieval results
        """
        # Initialize state
        initial_state: GraphRAGState = {
            "question": question,
            "extracted_entities": [],
            "matched_nodes": [],
            "nodes": [],
            "relationships": [],
            "context": "",
            "answer": "",
            "retrieval_steps": [],
            "error": None
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "query": question,
            "extractedEntities": final_state["extracted_entities"],
            "matchedNodes": final_state["matched_nodes"],
            "nodes": final_state["nodes"],
            "relationships": final_state["relationships"],
            "context": final_state["context"],
            "answer": final_state["answer"],
            "retrievalSteps": final_state["retrieval_steps"]
        }


# Singleton instance (for default config)
_pipeline: Optional[GraphRAGPipeline] = None


def get_pipeline() -> GraphRAGPipeline:
    """Get or create GraphRAGPipeline instance with default config"""
    global _pipeline
    if _pipeline is None:
        _pipeline = GraphRAGPipeline()
    return _pipeline


def create_retriever_with_config(api_key: str, model: str, base_url: Optional[str] = None) -> GraphRAGPipeline:
    """Create a new GraphRAGPipeline with custom configuration"""
    return GraphRAGPipeline(api_key=api_key, model=model, base_url=base_url)


def retrieve_from_graph(
    question: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve from graph using RAG pipeline (convenience function)
    
    Args:
        question: The natural language question
        api_key: Optional custom API key
        model: Optional custom model name
        base_url: Optional custom API base URL
        
    Returns:
        Dictionary with retrieval results
    """
    if api_key and model:
        # Use custom configuration
        pipeline = create_retriever_with_config(api_key, model, base_url)
    else:
        # Use default configuration
        pipeline = get_pipeline()
    
    return pipeline.retrieve(question)
