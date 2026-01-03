"""
Neo4j Service Module - Handles all Neo4j database operations
Using LangChain's Neo4j integration
"""
from typing import Any, Dict, List, Optional, Tuple
from neo4j import GraphDatabase, Driver
from contextlib import contextmanager
from config import get_settings


class Neo4jService:
    """Service class for Neo4j database operations"""
    
    _driver: Optional[Driver] = None
    
    @classmethod
    def get_driver(cls) -> Driver:
        """Get or create Neo4j driver instance"""
        if cls._driver is None:
            settings = get_settings()
            cls._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password)
            )
        return cls._driver
    
    @classmethod
    def close(cls):
        """Close the driver connection"""
        if cls._driver:
            cls._driver.close()
            cls._driver = None
    
    @classmethod
    @contextmanager
    def session(cls):
        """Context manager for Neo4j session"""
        driver = cls.get_driver()
        session = driver.session()
        try:
            yield session
        finally:
            session.close()
    
    @classmethod
    def verify_connection(cls) -> Dict[str, Any]:
        """Verify Neo4j connection status"""
        try:
            driver = cls.get_driver()
            driver.verify_connectivity()
            return {"connected": True}
        except Exception as e:
            return {"connected": False, "error": str(e)}
    
    @classmethod
    def get_stats(cls) -> Dict[str, int]:
        """Get graph statistics (node and relationship counts)"""
        with cls.session() as session:
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()["count"]
            
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]
            
            return {
                "nodeCount": node_count,
                "relationshipCount": rel_count
            }
    
    @classmethod
    def run_query(cls, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a Cypher query and return results with nodes and relationships"""
        params = params or {}
        
        with cls.session() as session:
            result = session.run(query, params)
            records = []
            nodes_map = {}  # id -> node dict
            relationships_map = {}  # unique key -> relationship dict
            
            for record in result:
                record_dict = {}
                for key in record.keys():
                    value = record[key]
                    serialized = cls._serialize_value(value)
                    record_dict[key] = serialized
                    
                    # Extract nodes and relationships from the value
                    cls._extract_graph_elements(value, nodes_map, relationships_map)
                
                records.append(record_dict)
            
            summary = result.consume()
            
            return {
                "records": records,
                "nodes": list(nodes_map.values()),
                "relationships": list(relationships_map.values()),
                "summary": {
                    "nodesCreated": summary.counters.nodes_created,
                    "relationshipsCreated": summary.counters.relationships_created
                }
            }
    
    @classmethod
    def _extract_graph_elements(cls, value: Any, nodes_map: Dict, relationships_map: Dict):
        """Extract nodes and relationships from a value recursively"""
        if value is None:
            return
        
        # Check if it's a Node
        if hasattr(value, 'labels') and hasattr(value, 'id') and hasattr(value, 'items'):
            node_id = value.id
            if node_id not in nodes_map:
                nodes_map[node_id] = {
                    "id": node_id,
                    "labels": list(value.labels),
                    "properties": dict(value)
                }
        
        # Check if it's a Relationship
        elif hasattr(value, 'type') and hasattr(value, 'start_node') and hasattr(value, 'end_node'):
            rel_key = f"{value.start_node.id}-{value.type}-{value.end_node.id}"
            if rel_key not in relationships_map:
                relationships_map[rel_key] = {
                    "id": value.id,
                    "type": value.type,
                    "startNodeId": value.start_node.id,
                    "endNodeId": value.end_node.id,
                    "properties": dict(value)
                }
            # Also extract the connected nodes
            cls._extract_graph_elements(value.start_node, nodes_map, relationships_map)
            cls._extract_graph_elements(value.end_node, nodes_map, relationships_map)
        
        # Check if it's a Path
        elif hasattr(value, 'nodes') and hasattr(value, 'relationships'):
            for node in value.nodes:
                cls._extract_graph_elements(node, nodes_map, relationships_map)
            for rel in value.relationships:
                cls._extract_graph_elements(rel, nodes_map, relationships_map)
        
        # Check if it's a list
        elif isinstance(value, list):
            for item in value:
                cls._extract_graph_elements(item, nodes_map, relationships_map)
    
    @classmethod
    def _serialize_value(cls, value: Any) -> Any:
        """Serialize Neo4j values to JSON-compatible format"""
        if value is None:
            return None
        
        # Check if it's a Node
        if hasattr(value, 'labels') and hasattr(value, 'id'):
            return {
                "_type": "node",
                "id": value.id,
                "labels": list(value.labels),
                "properties": dict(value)
            }
        
        # Check if it's a Relationship
        if hasattr(value, 'type') and hasattr(value, 'start_node') and hasattr(value, 'end_node'):
            return {
                "_type": "relationship",
                "id": value.id,
                "type": value.type,
                "startNodeId": value.start_node.id,
                "endNodeId": value.end_node.id,
                "properties": dict(value)
            }
        
        # Check if it's a list
        if isinstance(value, list):
            return [cls._serialize_value(v) for v in value]
        
        # Check if it's a dict
        if isinstance(value, dict):
            return {k: cls._serialize_value(v) for k, v in value.items()}
        
        return value
    
    @classmethod
    def create_knowledge_graph(
        cls,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Create nodes and relationships in Neo4j"""
        nodes_created = 0
        relationships_created = 0
        
        with cls.session() as session:
            # Create nodes
            for entity in entities:
                entity_type = entity.get("type", "Entity") or "Entity"
                name = entity.get("name", "") or ""
                props = entity.get("properties") or {}
                
                # Ensure props is a valid dict
                if props is None:
                    props = {}
                
                # Skip if name is empty
                if not name:
                    continue
                
                # Sanitize entity_type for Cypher label
                entity_type = entity_type.replace(" ", "_").replace("-", "_")
                
                query = f"""
                    MERGE (n:`{entity_type}` {{name: $name}})
                    SET n.type = $entity_type
                    RETURN n
                """
                result = session.run(query, {"name": name, "entity_type": entity_type})
                summary = result.consume()
                nodes_created += summary.counters.nodes_created
            
            # Create relationships
            for rel in relationships:
                source = rel.get("source", "") or ""
                target = rel.get("target", "") or ""
                rel_type = rel.get("type", "RELATED_TO") or "RELATED_TO"
                
                # Skip if source or target is empty
                if not source or not target:
                    continue
                
                # Sanitize rel_type for Cypher relationship type
                rel_type = rel_type.replace(" ", "_").replace("-", "_")
                
                query = f"""
                    MATCH (a {{name: $source}})
                    MATCH (b {{name: $target}})
                    MERGE (a)-[r:`{rel_type}`]->(b)
                    RETURN r
                """
                result = session.run(query, {"source": source, "target": target})
                summary = result.consume()
                relationships_created += summary.counters.relationships_created
        
        return {
            "nodesCreated": nodes_created,
            "relationshipsCreated": relationships_created
        }
    
    @classmethod
    def clear_graph(cls) -> bool:
        """Clear all data from the graph"""
        with cls.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        return True
    
    @classmethod
    def find_nodes_by_name(cls, name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find nodes by name (case-insensitive, partial match)"""
        with cls.session() as session:
            # Strategy 1: Exact match
            result = session.run(
                "MATCH (n) WHERE toLower(n.name) = toLower($name) RETURN n LIMIT $limit",
                {"name": name, "limit": limit}
            )
            records = list(result)
            
            if not records:
                # Strategy 2: Contains match
                result = session.run(
                    "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($name) RETURN n LIMIT $limit",
                    {"name": name, "limit": limit}
                )
                records = list(result)
            
            if not records:
                # Strategy 3: Reverse contains
                result = session.run(
                    """MATCH (n) 
                    WHERE toLower($name) CONTAINS toLower(n.name) 
                    AND size(n.name) >= 2 
                    RETURN n LIMIT $limit""",
                    {"name": name, "limit": limit}
                )
                records = list(result)
            
            nodes = []
            for record in records:
                node = record["n"]
                nodes.append({
                    "id": node.id,
                    "labels": list(node.labels),
                    "properties": dict(node)
                })
            
            return nodes
    
    @classmethod
    def get_neighbors(cls, node_name: str, hops: int = 2, limit: int = 10) -> Tuple[List[Dict], List[Dict]]:
        """Get neighboring nodes and relationships for a given node with variable hop depth"""
        with cls.session() as session:
            # Dynamic hop query using variable-length relationship pattern
            query = f"""
                MATCH path = (start {{name: $name}})-[*1..{hops}]-(neighbor)
                WHERE start <> neighbor
                UNWIND nodes(path) as node
                UNWIND relationships(path) as rel
                RETURN DISTINCT node, rel
                LIMIT $limit
            """
            result = session.run(query, {"name": node_name, "limit": limit * 2})
            
            nodes_map = {}
            relationships_map = {}
            
            for record in result:
                # Process node
                node = record["node"]
                if node.id not in nodes_map:
                    nodes_map[node.id] = {
                        "id": node.id,
                        "labels": list(node.labels),
                        "properties": dict(node)
                    }
                
                # Process relationship
                rel = record["rel"]
                rel_key = f"{rel.start_node.id}-{rel.type}-{rel.end_node.id}"
                if rel_key not in relationships_map:
                    relationships_map[rel_key] = {
                        "id": rel.id,
                        "type": rel.type,
                        "startNodeId": rel.start_node.id,
                        "endNodeId": rel.end_node.id,
                        "properties": dict(rel)
                    }
            
            return list(nodes_map.values()), list(relationships_map.values())
