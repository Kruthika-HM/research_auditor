import networkx as nx
import os

def save_graph(G: nx.DiGraph, filepath: str = "data/knowledge_graph.graphml") -> None:
    """
    Saves a NetworkX directed graph to disk in GraphML format.
    Automatically creates destination directories if they don't exist.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nx.write_graphml(G, filepath)
        print(f"[SUCCESS] Graph successfully saved to {filepath}")
        print(f"          - Nodes: {G.number_of_nodes()}")
        print(f"          - Edges: {G.number_of_edges()}")
    except Exception as e:
        print(f"[ERROR] Failed to save graph: {e}")
        raise

def load_graph(filepath: str = "data/knowledge_graph.graphml") -> nx.DiGraph:
    """
    Loads a NetworkX directed graph from disk from a GraphML file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] No saved graph found at target path: {filepath}")
    
    try:
        G = nx.read_graphml(filepath)
        print(f"[SUCCESS] Graph loaded from {filepath}")
        print(f"          - Nodes: {G.number_of_nodes()}")
        print(f"          - Edges: {G.number_of_edges()}")
        return G
    except Exception as e:
        print(f"[ERROR] Failed to load graph: {e}")
        raise