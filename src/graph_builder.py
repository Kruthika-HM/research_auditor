import json
import os
import networkx as nx
from rapidfuzz import fuzz
from src.persistence import save_graph

def normalize_entity_name(name: str, canonical_entities: set, threshold: int = 85) -> str:
    """
    Normalizes an incoming entity name using rapidfuzz token_sort_ratio 
    to match against existing canonical entities, preventing graph fragmentation.
    """
    if not name:
        return "Unknown"
        
    cleaned_name = name.strip()
    
    # If registry is empty, return the name as its own canonical reference
    if not canonical_entities:
        return cleaned_name
        
    # Check against existing canonical entities using fuzzy string matching
    for canonical in canonical_entities:
        score = fuzz.token_sort_ratio(cleaned_name.lower(), canonical.lower())
        if score >= threshold:
            return canonical
            
    # If no match exceeds the threshold, treat it as a new unique entity
    return cleaned_name

def build_knowledge_graph(json_path: str, similarity_threshold: int = 85) -> nx.DiGraph:
    """
    Reads extracted triplets from JSON, resolves entity name variations, 
    and constructs a directed NetworkX graph.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[ERROR] Triplet file not found at {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        triplets = json.load(f)
        
    G = nx.DiGraph()
    canonical_entities = set()
    
    print(f"[INFO] Processing {len(triplets)} raw triplets for graph construction...")
    
    for item in triplets:
        raw_subj = item.get("subject", "")
        predicate = item.get("predicate", "relates_to")
        raw_obj = item.get("object", "")
        
        if not raw_subj or not raw_obj:
            continue
            
        # Apply entity resolution and normalization
        subj = normalize_entity_name(raw_subj, canonical_entities, similarity_threshold)
        canonical_entities.add(subj)
        
        obj = normalize_entity_name(raw_obj, canonical_entities, similarity_threshold)
        canonical_entities.add(obj)
        
        # Add edge with the relational predicate attribute
        G.add_edge(subj, obj, relation=predicate)
        
    return G

if __name__ == "__main__":
    triplets_file = "data/extracted_triplets.json"
    graph_output_path = "data/knowledge_graph.graphml"
    
    # 1. Build the normalized NetworkX graph
    G = build_knowledge_graph(json_path=triplets_file, similarity_threshold=85)
    
    print(f"[SUCCESS] Knowledge Graph built successfully!")
    print(f"          - Total Nodes: {G.number_of_nodes()}")
    print(f"          - Total Edges: {G.number_of_edges()}")
    
    # 2. Persist the graph to disk for downstream tasks and GraphRAG
    save_graph(G, filepath=graph_output_path)