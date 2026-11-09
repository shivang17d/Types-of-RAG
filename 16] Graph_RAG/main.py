"""
16] Graph RAG
=============
Pipeline:
1. Graph Extraction: Uses OpenAI to extract (Subject, Relation, Object, Confidence) triples
   from raw documents to build a Knowledge Graph on-the-fly.
2. Community Detection & Summarization: Groups connected entities into communities and
   generates community-level summaries.
3. Graph-Augmented Retrieval: Matches query entities to graph nodes, expands 2-hop
   neighborhoods weighted by edge confidence scores, and combines them with community
   summaries to answer relational and global questions.
"""

import os
import json
from collections import defaultdict, deque
from typing import List, Dict, Set
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class GraphRAG:
    def __init__(self, chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chat_model = chat_model
        self.triples: List[Dict] = []
        self.adj: Dict[str, List[Dict]] = defaultdict(list)
        self.community_summary: str = ""

    def build_knowledge_graph(self, documents: List[str]):
        """Extracts entity-relation triples with confidence scores and builds the knowledge graph."""
        joined_docs = "\n".join([f"Doc {i+1}: {d}" for i, d in enumerate(documents)])
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract a Knowledge Graph of entity relationships from the documents. "
                        "Normalize entity names to Title Case. "
                        'Return JSON: {"triples": [{"source": str, "relation": str, "target": str, "confidence": float}], '
                        '"community_summary": str}'
                    ),
                },
                {"role": "user", "content": joined_docs},
            ],
            temperature=0.0,
        )
        parsed = json.loads(resp.choices[0].message.content)
        self.triples = parsed.get("triples", [])
        self.community_summary = parsed.get("community_summary", "")

        for t in self.triples:
            src, rel, tgt, conf = t["source"], t["relation"], t["target"], float(t.get("confidence", 0.9))
            self.adj[src].append({"neighbor": tgt, "relation": rel, "confidence": conf, "direction": "->"})
            self.adj[tgt].append({"neighbor": src, "relation": rel, "confidence": conf, "direction": "<-"})

    def traverse_subgraph(self, seed_entities: List[str], max_hops: int = 2, min_confidence: float = 0.6) -> List[str]:
        """BFS traversal over the knowledge graph filtered by confidence scores."""
        visited: Set[str] = set()
        queue = deque([(entity, 0) for entity in seed_entities])
        paths = []

        while queue:
            node, depth = queue.popleft()
            if node in visited or depth >= max_hops:
                continue
            visited.add(node)

            for edge in self.adj.get(node, []):
                if edge["confidence"] >= min_confidence:
                    if edge["direction"] == "->":
                        paths.append(f"({node}) -[{edge['relation']} (conf={edge['confidence']:.2f})]-> ({edge['neighbor']})")
                    else:
                        paths.append(f"({edge['neighbor']}) -[{edge['relation']} (conf={edge['confidence']:.2f})]-> ({node})")
                    if edge["neighbor"] not in visited:
                        queue.append((edge["neighbor"], depth + 1))

        return list(dict.fromkeys(paths))

    def query(self, user_query: str) -> Dict:
        all_nodes = list(self.adj.keys())
        # Match query to graph seed entities via OpenAI
        seed_resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": 'Select the graph entities most relevant to the user query. Return JSON: {"seed_entities": [str]}',
                },
                {"role": "user", "content": f"Available Entities: {all_nodes}\nQuery: {user_query}"},
            ],
            temperature=0.0,
        )
        seeds = json.loads(seed_resp.choices[0].message.content).get("seed_entities", [])
        subgraph_edges = self.traverse_subgraph(seeds, max_hops=2)

        graph_context = (
            f"Community Summary:\n{self.community_summary}\n\n"
            f"Traversed Subgraph Edges:\n" + "\n".join([f"- {e}" for e in subgraph_edges])
        )
        ans_resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Answer the question using the Knowledge Graph relationships and community summary."},
                {"role": "user", "content": f"{graph_context}\n\nQuestion: {user_query}"},
            ],
            temperature=0.0,
        )
        return {
            "seed_entities": seeds,
            "subgraph_edges": subgraph_edges,
            "community_summary": self.community_summary,
            "answer": ans_resp.choices[0].message.content.strip(),
        }


if __name__ == "__main__":
    docs = [
        "BioGenix acquired NeuroPulse Labs in 2024 to gain access to its SynapCure compound.",
        "SynapCure inhibits Protein-Kinase-Z, which is the primary driver of Inflammation Syndrome X.",
        "Dr. Marcus Vance invented SynapCure while working at NeuroPulse Labs.",
    ]

    grag = GraphRAG()
    grag.build_knowledge_graph(docs)
    res = grag.query("How is BioGenix connected to the treatment of Inflammation Syndrome X and Dr. Marcus Vance?")
    print("Seed Entities:", res["seed_entities"])
    print("\nTraversed Knowledge Graph Paths:")
    for edge in res["subgraph_edges"]:
        print(" ", edge)
    print(f"\nGraphRAG Answer:\n{res['answer']}")

