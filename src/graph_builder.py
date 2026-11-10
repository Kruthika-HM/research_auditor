import json
from pathlib import Path
import networkx as nx
from rapidfuzz import fuzz

def normalize_entity_name(name: str, existing_entities: set, threshold: int = 85) -> str:
    """
    Compares a new entity name against already accepted entities using fuzzy matching.
    If a close match is found, it maps the new name to the existing canonical name.
    """
    if not name:
        return ""
        
    cleaned_name = name.strip().lower()
    
    for canonical in existing_entities:
        # Compare strings using token sort ratio
        score = fuzz.token_sort_ratio(cleaned_name, canonical.lower())
        if score >= threshold:
            return canonical # Return the existing canonical name to merge them
            
    return name.strip() # If no match, return the original name as a new canonical entity

def build_knowledge_graph(json_path: str, similarity_threshold: int = 85) -> nx.DiGraph:
    """
    Loads extracted JSON triplets, normalizes entities to fix duplicates, 
    and constructs a directed graph using NetworkX.
    """
    print(f"Loading triplets from: {json_path}...")
    
    with open(json_path, "r", encoding="utf-8") as f:
        triplets = json.load(f)
        
    G = nx.DiGraph()
    canonical_entities = set()
    
    for item in triplets:
        raw_subject = item.get("subject")
        predicate = item.get("predicate")
        raw_object = item.get("object")
        
        if raw_subject and raw_object and predicate:
            # Normalize subject and object to prevent duplicates
            subject = normalize_entity_name(raw_subject, canonical_entities, similarity_threshold)
            canonical_entities.add(subject)
            
            object_node = normalize_entity_name(raw_object, canonical_entities, similarity_threshold)
            canonical_entities.add(object_node)
            
            # Add edge with the normalized nodes
            G.add_edge(subject, object_node, relation=predicate)
            
    print(f"Successfully built normalized graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def analyze_graph(G: nx.DiGraph):
    """
    Runs basic dependency audits and prints graph metrics.
    """
    print("\n--- Graph Dependency Audit ---")
    print(f"Total Components (Nodes): {G.number_of_nodes()}")
    print(f"Total Relationships (Edges): {G.number_of_edges()}")
    
    print("\nTop Connected Components:")
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    for node, deg in degrees[:5]:
        print(f" - '{node}' (Connections: {deg})")

if __name__ == "__main__":
    json_path = "data/processed_markdown/extracted_triplets.json"
    
    if Path(json_path).exists():
        graph = build_knowledge_graph(json_path)
        analyze_graph(graph)
    else:
        print(f"Triplet JSON file not found at {json_path}. Run extractor.py first!")