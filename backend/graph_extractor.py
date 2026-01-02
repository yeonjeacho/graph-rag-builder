"""
Graph Extractor Module - LangChain-based Knowledge Graph Extraction
Extracts entities and relationships from documents using LLM
"""
from typing import List, Dict, Any, Optional
import re
import time
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field
from config import get_settings


def invoke_together_ai_with_retry(chain, input_data, max_retries=5, base_delay=2):
    """Together AI API 전용 재시도 로직 - 504/timeout 에러 시 exponential backoff으로 재시도"""
    for attempt in range(max_retries):
        try:
            response = chain.invoke(input_data)
            return response
        except Exception as e:
            error_msg = str(e).lower()
            is_retriable = any(keyword in error_msg for keyword in ['504', 'timeout', 'timed out', 'gateway', 'rate limit', '429', 'too many requests'])
            if is_retriable and attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                wait_time = min(wait_time, 30)
                print(f"⚠️ Together AI error (attempt {attempt + 1}/{max_retries})")
                print(f"   Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            elif is_retriable:
                print(f"❌ Together AI timeout after {max_retries} attempts")
                raise Exception("Together API timeout - max retries exceeded")
            else:
                print(f"❌ API error: {str(e)}")
                raise
    raise Exception("Unexpected: max retries exceeded")


class Entity(BaseModel):
    """Schema for extracted entity"""
    name: str = Field(description="The name of the entity (acronym or full name)")
    type: str = Field(description="The type/label of the entity (e.g., Acronym, TechnicalConcept, Channel, Protocol)")


class Relationship(BaseModel):
    """Schema for extracted relationship"""
    source: str = Field(description="The source entity name")
    target: str = Field(description="The target entity name")
    type: str = Field(description="The relationship type (e.g., DEFINES, RELATED_TO, PART_OF)")


class ExtractionResult(BaseModel):
    """Schema for extraction result"""
    entities: List[Entity] = Field(default_factory=list, description="List of extracted entities")
    relationships: List[Relationship] = Field(default_factory=list, description="List of extracted relationships")


class GraphExtractor:
    """
    LangChain-based Knowledge Graph Extractor
    Uses LLM to extract entities and relationships from documents
    """
    
    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        """
        Initialize the GraphExtractor
        
        Args:
            api_key: API key for the LLM service
            model: Model name to use
            base_url: Base URL for the API
        """
        settings = get_settings()
        
        self.api_key = api_key or settings.together_api_key
        self.model = model or settings.together_model
        self.base_url = base_url or settings.together_base_url
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            temperature=0.0,
            max_tokens=2000
        )
        
        # Create extraction prompt - 더 엄격한 JSON 출력 프롬프트
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledge graph extraction expert. Extract entities and relationships from the given document.

CRITICAL RULES:
1. Output MUST be valid JSON only - no explanations, no markdown, no extra text
2. Use this exact format:
{{"entities": [{{"name": "...", "type": "..."}}], "relationships": [{{"source": "...", "target": "...", "type": "..."}}]}}
3. Entity types: Acronym, TechnicalConcept, Protocol, Channel, Process, Component, Standard, Organization
4. Relationship types: DEFINES, RELATED_TO, PART_OF, USES, CONTAINS, DEPENDS_ON
5. If no entities found, return: {{"entities": [], "relationships": []}}
6. NEVER include markdown code blocks or explanations"""),
            ("human", "Extract entities and relationships from this document:\n\n{document}")
        ])
        
        # JSON parser
        self.parser = JsonOutputParser(pydantic_object=ExtractionResult)
    
    def _process_single_chunk(self, chunk_data: tuple) -> Dict[str, Any]:
        """단일 청크 처리 (병렬 실행용)"""
        i, chunk, prompt_chain = chunk_data
        
        try:
            response = invoke_together_ai_with_retry(prompt_chain, {"document": chunk})
            
            # Parse response
            raw_content = response.content if hasattr(response, 'content') else str(response)
            
            # Regex Cleaning
            content = raw_content.strip()
            content = re.sub(r"```json|```", "", content).strip()
            
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                content = match.group(0)
                try:
                    chunk_result = self.parser.parse(content)
                    entities_count = len(chunk_result.get('entities', []))
                    rels_count = len(chunk_result.get('relationships', []))
                    print(f"✅ Chunk {i+1}: {entities_count} entities, {rels_count} relationships")
                    return {"index": i, "result": chunk_result, "success": True}
                except Exception:
                    print(f"⚠️ Chunk {i+1}: Parse failed, empty result")
                    return {"index": i, "result": {"entities": [], "relationships": []}, "success": True}
            else:
                print(f"⚠️ Chunk {i+1}: No JSON found, empty result")
                return {"index": i, "result": {"entities": [], "relationships": []}, "success": True}
                
        except Exception as e:
            print(f"❌ Chunk {i+1}: Error - {str(e)[:50]}")
            return {"index": i, "result": {"entities": [], "relationships": []}, "success": False}
    
    def extract(self, document: str, chunk_size: int = 600, chunk_overlap: int = 100) -> Dict[str, Any]:
        """
        Extract knowledge graph from document using parallel processing
        
        Args:
            document: The text document to extract from
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            Dictionary with 'entities' and 'relationships' lists
        """
        try:
            start_time = time.time()
            MAX_WORKERS = 3  # 동시 처리 청크 수
            
            # --- [Step 1: Smart Chunking] ---
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(document)
            
            # 예상 시간 계산 (병렬 처리 고려)
            estimated_time = (len(chunks) * 13) / MAX_WORKERS / 60
            print(f"\n📝 Document split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
            print(f"🚀 Parallel processing with {MAX_WORKERS} workers")
            print(f"⏱️ Estimated time: {estimated_time:.1f} minutes\n")
            
            # --- [Step 2: Parallel Extraction] ---
            prompt_chain = self.prompt | self.llm
            chunk_results = []
            failed_count = 0
            
            # Together AI 초기 연결 테스트
            print("🔌 Testing Together AI connection...")
            try:
                test_response = invoke_together_ai_with_retry(prompt_chain, {"document": "test"}, max_retries=2)
                print("✅ Together AI connection verified\n")
            except Exception as e:
                print(f"⚠️ Initial connection test failed: {str(e)}")
                print("   Continuing anyway...\n")
            
            # 청크 데이터 준비
            chunk_data_list = [(i, chunk, prompt_chain) for i, chunk in enumerate(chunks)]
            
            # 병렬 처리 실행
            print(f"🔄 Processing {len(chunks)} chunks in parallel...\n")
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(self._process_single_chunk, data): data[0] for data in chunk_data_list}
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        chunk_results.append(result)
                        if not result["success"]:
                            failed_count += 1
                    except Exception as e:
                        print(f"❌ Future error: {str(e)[:50]}")
                        failed_count += 1
                    
                    # 진행률 표시 (10개마다)
                    if len(chunk_results) % 10 == 0:
                        print(f"📊 Progress: {len(chunk_results)}/{len(chunks)} chunks completed")
            
            # 결과를 인덱스 순서로 정렬
            chunk_results.sort(key=lambda x: x["index"])
            
            # --- [Step 3: Entity Deduplication & Merging] ---
            merged_entities = {}
            all_relationships = []
            
            for item in chunk_results:
                chunk_result = item["result"]
                
                if 'relationships' in chunk_result:
                    all_relationships.extend(chunk_result['relationships'])
                
                if 'entities' in chunk_result:
                    for entity in chunk_result['entities']:
                        name = entity.get('name', '').strip()
                        if not name:
                            continue
                        name_key = name.lower()
                        if name_key not in merged_entities:
                            merged_entities[name_key] = entity
            
            final_entities = list(merged_entities.values())
            final_relationships = all_relationships
            
            # 최종 보고
            elapsed = time.time() - start_time
            print(f"\n🎉 Extraction complete!")
            print(f"   - Total entities: {len(final_entities)}")
            print(f"   - Total relationships: {len(final_relationships)}")
            print(f"   - Failed chunks: {failed_count}/{len(chunks)}")
            print(f"   - Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            
            # Post-process
            result = self._post_process({
                "entities": final_entities,
                "relationships": final_relationships
            })
            
            return {
                "entities": result.get("entities", []),
                "relationships": result.get("relationships", [])
            }
            
        except Exception as e:
            print(f"\n🚨 Fatal error: {str(e)}")
            traceback.print_exc()
            return {
                "entities": list(merged_entities.values()) if 'merged_entities' in locals() else [],
                "relationships": all_relationships if 'all_relationships' in locals() else []
            }
    
    def _post_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process extraction results
        - Remove duplicates
        - Validate relationships
        - Clean up entity names
        """
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        
        # Create entity name set for validation
        entity_names = {e.get("name", "").lower() for e in entities if e.get("name")}
        
        # Filter relationships to only include those with valid entities
        valid_relationships = []
        for rel in relationships:
            source = rel.get("source", "").lower()
            target = rel.get("target", "").lower()
            if source in entity_names and target in entity_names:
                valid_relationships.append(rel)
        
        # Remove duplicate relationships
        seen_rels = set()
        unique_relationships = []
        for rel in valid_relationships:
            rel_key = (rel.get("source", "").lower(), rel.get("target", "").lower(), rel.get("type", "").lower())
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                unique_relationships.append(rel)
        
        return {
            "entities": entities,
            "relationships": unique_relationships
        }


# Singleton instance
_extractor_instance = None


def get_extractor() -> GraphExtractor:
    """Get or create the singleton GraphExtractor instance"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = GraphExtractor()
    return _extractor_instance


def create_extractor_with_config(api_key: str, model: str, base_url: str = None) -> GraphExtractor:
    """Create a new GraphExtractor with custom configuration"""
    return GraphExtractor(api_key=api_key, model=model, base_url=base_url)


def extract_knowledge_graph(
    document: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract knowledge graph from document (convenience function)
    
    Args:
        document: The text document to extract from
        api_key: Optional custom API key
        model: Optional custom model name
        base_url: Optional custom API base URL
        
    Returns:
        Dictionary with 'entities' and 'relationships' lists
    """
    if api_key and model:
        # Use custom configuration
        extractor = create_extractor_with_config(api_key, model, base_url)
    else:
        # Use default configuration
        extractor = get_extractor()
    
    return extractor.extract(document, chunk_size=600, chunk_overlap=100)
