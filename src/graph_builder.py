import json
import os
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from src.persistence import save_graph

# Initialize a lightweight embedding model once globally
# (Runs locally, fast, and completely offline)
print("[INFO] Loading semantic embedding model for entity resolution...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize_entity_name_semantic(name: str, canonical_entities: list, threshold: float = 0.85) -> str:
    """
    Normalizes an incoming entity name using semantic embedding similarity 
    to automatically match synonyms (e.g., 'break' and 'fracture').
    """
    if not name:
        return "Unknown"
        
    cleaned_name = name.strip()
    
    # If the registry is empty, this is our first canonical entity
    if not canonical_entities:
        return cleaned_name
        
    # Generate embedding for the incoming entity
    incoming_embedding = embed_model.encode(cleaned_name, convert_to_tensor=True)
    
    best_match = cleaned_name
    highest_score = 0.0
    
    # Compare against existing canonical entities using cosine similarity
    for canonical in canonical_entities:
        canonical_embedding = embed_model.encode(canonical, convert_to_tensor=True)
        score = util.cos_sim(incoming_embedding, canonical_embedding).item()
        
        if score > highest_score:
            highest_score = score
            best_match = canonical
            
    # If similarity exceeds threshold, merge into the existing canonical node
    if highest_score >= threshold:
        return best_match
        
    # Otherwise, treat it as a brand new unique entity
    return cleaned_name

def build_knowledge_graph(json_path: str, similarity_threshold: float = 0.85) -> nx.DiGraph:
    """
    Reads extracted triplets, resolves semantic entity variations, 
    and constructs a directed NetworkX graph.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[ERROR] Triplet file not found at {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        triplets = json.load(f)
        
    G = nx.DiGraph()
    canonical_entities = [] # Using a list/set to track canonical nodes
    
    print(f"[INFO] Processing {len(triplets)} raw triplets with semantic resolution...")
    
    for item in triplets:
        raw_subj = item.get("subject", "")
        predicate = item.get("predicate", "relates_to")
        raw_obj = item.get("object", "")
        
        if not raw_subj or not raw_obj:
            continue
            
        # Apply semantic entity resolution
        subj = normalize_entity_name_semantic(raw_subj, canonical_entities, similarity_threshold)
        if subj not in canonical_entities:
            canonical_entities.append(subj)
            
        obj = normalize_entity_name_semantic(raw_obj, canonical_entities, similarity_threshold)
        if obj not in canonical_entities:
            canonical_entities.append(obj)
            
        # Add edge with relational predicate attribute
        G.add_edge(subj, obj, relation=predicate)
        
    return G

if __name__ == "__main__":
    triplets_file = "data/extracted_triplets.json"
    graph_output_path = "data/knowledge_graph.graphml"
    
    # 1. Build the semantically resolved graph
    G = build_knowledge_graph(json_path=triplets_file, similarity_threshold=0.85)
    
    print(f"[SUCCESS] Semantic Knowledge Graph built successfully!")
    print(f"          - Total Canonical Nodes: {G.number_of_nodes()}")
    print(f"          - Total Edges: {G.number_of_edges()}")
    
    # 2. Persist to disk
    save_graph(G, filepath=graph_output_path)