from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
from src.persistence import load_graph


class KnowledgeGraphQuery:
    """Symbolic traversal and subgraph query interface over a NetworkX DiGraph."""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        # Map lowercase node label -> canonical node ID in graph
        self._lower_index: Dict[str, str] = {
            str(node).lower(): str(node) for node in self.graph.nodes()
        }

    @classmethod
    def from_graphml(cls, filepath: str) -> "KnowledgeGraphQuery":
        graph = load_graph(filepath)
        return cls(graph)

    def resolve_node(self, query_term: str) -> Optional[str]:
        """Resolves an exact or case-insensitive node match in the graph."""
        if not query_term:
            return None
        term = query_term.strip()
        if term in self.graph:
            return str(term)
        return self._lower_index.get(term.lower())

    def find_matching_nodes(self, keyword: str, limit: int = 10) -> List[str]:
        """Finds nodes whose label contains the keyword (case-insensitive)."""
        kw = keyword.strip().lower()
        if not kw:
            return []
        matches = [
            str(node) for node in self.graph.nodes() if kw in str(node).lower()
        ]
        return matches[:limit]

    def get_neighbors(
        self, entity: str, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Returns outgoing, incoming, or both neighbor relationships for an entity.
        direction: 'out' | 'in' | 'both'
        """
        node = self.resolve_node(entity)
        if node is None:
            return []

        edges: List[Dict[str, Any]] = []

        if direction in ("out", "both"):
            for _, target, data in self.graph.out_edges(node, data=True):
                edges.append({
                    "source": str(node),
                    "relation": data.get("relation", "relates_to"),
                    "target": str(target),
                    "direction": "outgoing",
                })

        if direction in ("in", "both"):
            for source, _, data in self.graph.in_edges(node, data=True):
                edges.append({
                    "source": str(source),
                    "relation": data.get("relation", "relates_to"),
                    "target": str(node),
                    "direction": "incoming",
                })

        return edges

    def extract_subgraph(
        self, entity: str, hops: int = 2, max_nodes: int = 50
    ) -> nx.DiGraph:
        """Extracts a bounded N-hop ego subgraph around a target entity."""
        node = self.resolve_node(entity)
        if node is None:
            return nx.DiGraph()

        undirected = self.graph.to_undirected()
        visited = {node}
        frontier = {node}

        for _ in range(max(1, hops)):
            next_frontier = set()
            for curr in frontier:
                for nbr in undirected.neighbors(curr):
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.add(nbr)
                        if len(visited) >= max_nodes:
                            break
                if len(visited) >= max_nodes:
                    break
            frontier = next_frontier
            if len(visited) >= max_nodes or not frontier:
                break

        return self.graph.subgraph(visited).copy()

    def find_paths(
        self, source_entity: str, target_entity: str, max_hops: int = 4
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Finds directed paths (up to max_hops edges) connecting source_entity
        to target_entity. Returns paths as lists of (u, relation, v) triples.
        """
        src = self.resolve_node(source_entity)
        dst = self.resolve_node(target_entity)
        if src is None or dst is None:
            return []

        paths_as_triples: List[List[Tuple[str, str, str]]] = []
        try:
            for path in nx.all_simple_paths(self.graph, source=src, target=dst, cutoff=max_hops):
                triple_path: List[Tuple[str, str, str]] = []
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    rel = self.graph[u][v].get("relation", "relates_to")
                    triple_path.append((str(u), str(rel), str(v)))
                paths_as_triples.append(triple_path)
        except nx.NetworkXNoPath:
            return []

        return paths_as_triples