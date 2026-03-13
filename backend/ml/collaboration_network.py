"""
backend/ml/collaboration_network.py - Co-authorship Network Analysis using NetworkX
"""
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from itertools import combinations

import networkx as nx
from loguru import logger

from backend.database import get_papers_collection


class CollaborationNetwork:
    """
    Builds and analyzes an author co-authorship network.
    - Nodes = Authors
    - Edges = Co-authored papers (weighted by number of shared papers)
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.papers_count = 0

    def build_network(self) -> nx.Graph:
        """Build co-authorship graph from all papers in MongoDB."""
        col = get_papers_collection()
        papers = list(col.find({}, {"authors": 1, "title": 1, "department": 1, "paper_id": 1}))

        self.graph.clear()
        self.papers_count = len(papers)

        # Author metadata tracking
        author_papers = defaultdict(list)
        author_dept = {}

        for paper in papers:
            authors = paper.get("authors", [])
            dept = paper.get("department", "Unknown")
            title = paper.get("title", "")

            for author in authors:
                author_papers[author].append(title)
                if author not in author_dept:
                    author_dept[author] = dept

            # Add nodes
            for author in authors:
                if not self.graph.has_node(author):
                    self.graph.add_node(author, department=author_dept.get(author, "Unknown"))

            # Add edges for all pairs of co-authors
            for a1, a2 in combinations(authors, 2):
                if self.graph.has_edge(a1, a2):
                    self.graph[a1][a2]["weight"] += 1
                    self.graph[a1][a2]["papers"].append(title)
                else:
                    self.graph.add_edge(a1, a2, weight=1, papers=[title])

        # Add paper count to nodes
        for author, paper_list in author_papers.items():
            if self.graph.has_node(author):
                self.graph.nodes[author]["paper_count"] = len(paper_list)

        logger.info(f"Built co-authorship network: {self.graph.number_of_nodes()} authors, "
                    f"{self.graph.number_of_edges()} collaborations")
        return self.graph

    def get_network_stats(self) -> Dict:
        """Compute key network statistics."""
        if not self.graph.nodes():
            self.build_network()

        G = self.graph

        if G.number_of_nodes() == 0:
            return {"error": "No authors found in database"}

        # Degree centrality = collaboration breadth
        degree_centrality = nx.degree_centrality(G)

        # Betweenness centrality = bridge between communities (expensive for large graphs)
        if G.number_of_nodes() <= 500:
            betweenness = nx.betweenness_centrality(G, weight="weight")
        else:
            # Sample-based approximation for large graphs
            betweenness = nx.betweenness_centrality(G, k=100, weight="weight")

        # Top collaborators by degree
        top_collaborators = sorted(
            [(author, G.degree(author, weight="weight")) for author in G.nodes()],
            key=lambda x: x[1], reverse=True
        )[:10]

        # Top by paper count
        top_by_papers = sorted(
            [(author, G.nodes[author].get("paper_count", 0)) for author in G.nodes()],
            key=lambda x: x[1], reverse=True
        )[:10]

        # Connected components
        components = list(nx.connected_components(G))

        # Department collaboration matrix
        dept_collab = self._dept_collaboration_matrix()

        return {
            "total_authors": G.number_of_nodes(),
            "total_collaborations": G.number_of_edges(),
            "total_papers": self.papers_count,
            "connected_components": len(components),
            "largest_component_size": max(len(c) for c in components) if components else 0,
            "avg_collaborations_per_author": round(
                2 * G.number_of_edges() / max(G.number_of_nodes(), 1), 2
            ),
            "top_collaborators": [
                {"author": a, "collaboration_weight": w} for a, w in top_collaborators
            ],
            "top_researchers_by_papers": [
                {"author": a, "paper_count": c} for a, c in top_by_papers
            ],
            "top_connectors": [
                {"author": a, "betweenness": round(b, 4)}
                for a, b in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
            "department_collaborations": dept_collab,
        }

    def _dept_collaboration_matrix(self) -> List[Dict]:
        """Count cross-department and within-department collaborations."""
        collab_counts = defaultdict(int)
        for a1, a2, data in self.graph.edges(data=True):
            d1 = self.graph.nodes[a1].get("department", "Unknown")
            d2 = self.graph.nodes[a2].get("department", "Unknown")
            key = tuple(sorted([d1, d2]))
            collab_counts[key] += data.get("weight", 1)

        return [
            {"dept1": k[0], "dept2": k[1], "collaborations": v}
            for k, v in sorted(collab_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        ]

    def get_graph_for_visualization(self, max_nodes: int = 50) -> Dict:
        """
        Export graph in a format suitable for Plotly/frontend visualization.
        Limits to top N most-connected nodes for readability.
        """
        if not self.graph.nodes():
            self.build_network()

        G = self.graph

        # Select top nodes by degree for visualization
        degrees = dict(G.degree())
        top_nodes = set(
            sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)[:max_nodes]
        )
        subgraph = G.subgraph(top_nodes)

        # Use spring layout for positioning
        pos = nx.spring_layout(subgraph, k=2, seed=42)

        nodes = []
        for node in subgraph.nodes():
            x, y = pos[node]
            nodes.append({
                "id": node,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "degree": subgraph.degree(node),
                "paper_count": subgraph.nodes[node].get("paper_count", 0),
                "department": subgraph.nodes[node].get("department", "Unknown"),
            })

        edges = []
        for a1, a2, data in subgraph.edges(data=True):
            edges.append({
                "source": a1,
                "target": a2,
                "weight": data.get("weight", 1),
            })

        return {"nodes": nodes, "edges": edges}
