from typing import Any, Dict, List, Optional
from src.graph_query import KnowledgeGraphQuery


class ResearchAuditor:
    """Audits claims and entity relationships against the built Knowledge Graph."""

    def __init__(self, kg_query: KnowledgeGraphQuery):
        self.kg = kg_query

    def audit_relationship(
        self,
        subject: str,
        target: str,
        expected_relation: Optional[str] = None,
        max_hops: int = 3,
    ) -> Dict[str, Any]:
        """
        Checks whether a relationship between `subject` and `target` is directly
        supported or connected via multi-hop paths in the graph.
        """
        resolved_subj = self.kg.resolve_node(subject)
        resolved_target = self.kg.resolve_node(target)

        if resolved_subj is None or resolved_target is None:
            return {
                "status": "UNVERIFIED_ENTITY",
                "subject_found": resolved_subj is not None,
                "target_found": resolved_target is not None,
                "resolved_subject": resolved_subj,
                "resolved_target": resolved_target,
                "direct_match": False,
                "paths": [],
                "summary": (
                    f"Missing node(s) in graph: "
                    f"subject='{subject}' ({resolved_subj}), target='{target}' ({resolved_target})."
                ),
            }

        paths = self.kg.find_paths(resolved_subj, resolved_target, max_hops=max_hops)
        direct_edges = [
            p[0] for p in paths if len(p) == 1
        ]

        relation_matched = False
        if expected_relation:
            exp_norm = expected_relation.strip().lower()
            relation_matched = any(
                exp_norm in rel.lower() for _, rel, _ in direct_edges
            )

        if direct_edges and (expected_relation is None or relation_matched):
            status = "SUPPORTED_DIRECT"
        elif paths:
            status = "SUPPORTED_INDIRECT"
        else:
            status = "NO_PATH_FOUND"

        return {
            "status": status,
            "subject_found": True,
            "target_found": True,
            "resolved_subject": resolved_subj,
            "resolved_target": resolved_target,
            "direct_match": bool(direct_edges),
            "relation_matched": relation_matched if expected_relation else None,
            "paths": paths,
            "summary": (
                f"[{status}] '{resolved_subj}' -> '{resolved_target}' "
                f"({len(direct_edges)} direct edge(s), {len(paths)} path(s) within {max_hops} hops)."
            ),
        }

    def audit_entity_coverage(self, entity: str, hops: int = 2) -> Dict[str, Any]:
        """Summarizes local evidence and neighborhood context around an entity."""
        resolved = self.kg.resolve_node(entity)
        if resolved is None:
            candidates = self.kg.find_matching_nodes(entity, limit=5)
            return {
                "status": "NOT_IN_GRAPH",
                "entity": entity,
                "similar_nodes": candidates,
                "neighbors": [],
                "subgraph_nodes": 0,
                "subgraph_edges": 0,
            }

        neighbors = self.kg.get_neighbors(resolved, direction="both")
        sub = self.kg.extract_subgraph(resolved, hops=hops)

        return {
            "status": "FOUND",
            "entity": resolved,
            "similar_nodes": [],
            "neighbors": neighbors,
            "subgraph_nodes": sub.number_of_nodes(),
            "subgraph_edges": sub.number_of_edges(),
        }

    def audit_triplets_batch(
        self, triplets: List[Dict[str, str]], max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """Audits a list of {'subject': ..., 'predicate': ..., 'object': ...} items."""
        reports = []
        for item in triplets:
            subj = item.get("subject", "").strip()
            pred = item.get("predicate", "").strip()
            obj = item.get("object", "").strip()
            if subj and obj:
                reports.append(
                    self.audit_relationship(
                        subject=subj,
                        target=obj,
                        expected_relation=pred or None,
                        max_hops=max_hops,
                    )
                )
        return reports