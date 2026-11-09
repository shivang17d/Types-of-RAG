"""
18] ConTReGen (Context-Driven Tree-Structured Retrieval)
========================================================
Pipeline:
Two-stage hierarchical framework for complex, multi-faceted queries:
1. Stage 1 (Top-Down Planning & Retrieval):
   - Analyzes the complex root query and decomposes it into a tree of facet sub-queries.
   - Retrieves and verifies passages for each leaf sub-query.
2. Stage 2 (Bottom-Up Synthesis & Generation):
   - Summarizes retrieved content at each leaf node.
   - Integrates summaries upwards from leaf nodes to the root to generate a coherent long-form response.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class ConTReGenRAG:
    def __init__(self, documents: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.doc_embeddings = [np.array(d.embedding) for d in resp.data]

    def top_down_decompose(self, root_query: str) -> List[Dict]:
        """Decomposes the root query into a 2-level tree of Main Facets -> Leaf Sub-Queries."""
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Decompose the complex query into 2 main facets, each containing 2 specific leaf sub-queries. "
                        'Return JSON: {"facets": [{"facet_title": str, "leaf_queries": [str, str]}]}'
                    ),
                },
                {"role": "user", "content": root_query},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content).get("facets", [])

    def retrieve_and_verify_leaf(self, leaf_query: str) -> str:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[leaf_query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.doc_embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        top_passage = self.documents[sims[0][1]]

        # Leaf node verification and summarization
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Verify and summarize the passage to answer the leaf sub-query concisely."},
                {"role": "user", "content": f"Leaf Query: {leaf_query}\nPassage: {top_passage}"},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    def run(self, root_query: str) -> Dict:
        # Stage 1: Top-Down Planning & Leaf Retrieval
        facet_tree = self.top_down_decompose(root_query)
        facet_summaries = []

        # Stage 2: Bottom-Up Synthesis (Leaves -> Facets -> Root)
        for facet in facet_tree:
            leaf_notes = []
            for lq in facet.get("leaf_queries", []):
                leaf_summary = self.retrieve_and_verify_leaf(lq)
                leaf_notes.append({"leaf_query": lq, "leaf_summary": leaf_summary})

            # Synthesize leaves into intermediate Facet Node summary
            joined_leaves = "\n".join([f"- {ln['leaf_summary']}" for ln in leaf_notes])
            facet_resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": f"Synthesize the leaf summaries into a section on '{facet['facet_title']}'."},
                    {"role": "user", "content": joined_leaves},
                ],
                temperature=0.0,
            )
            facet_summaries.append(
                {
                    "facet_title": facet["facet_title"],
                    "leaves": leaf_notes,
                    "facet_synthesis": facet_resp.choices[0].message.content.strip(),
                }
            )

        # Integrate Facet summaries into Root long-form response
        root_input = "\n\n".join([f"## {f['facet_title']}\n{f['facet_synthesis']}" for f in facet_summaries])
        root_resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Integrate the synthesized tree sections into a cohesive executive report."},
                {"role": "user", "content": f"Root Query: {root_query}\n\n{root_input}"},
            ],
            temperature=0.0,
        )

        return {
            "root_query": root_query,
            "tree": facet_summaries,
            "final_report": root_resp.choices[0].message.content.strip(),
        }


if __name__ == "__main__":
    corpus = [
        "Battery Chemistry: Solid-state lithium-metal cells achieve 450 Wh/kg energy density and eliminate flammable liquid electrolytes.",
        "Charging Speed: Silicon-anode fast-charge architecture reaches 10% to 80% state-of-charge in 9.5 minutes at 400kW.",
        "Supply Chain: Domestic lithium hydroxide refining capacity expanded by 60,000 metric tons in Nevada.",
        "Recycling: Hydrometallurgical closed-loop battery recycling recovers 96% of cobalt, nickel, and lithium.",
    ]

    contregen = ConTReGenRAG(corpus)
    res = contregen.run("Provide a technical and sustainability assessment of our next-generation EV battery program.")
    for f in res["tree"]:
        print(f"\n[Facet Node] {f['facet_title']}")
        for l in f["leaves"]:
            print(f"  ├── Leaf Query: {l['leaf_query']}")
            print(f"  │   Summary   : {l['leaf_summary']}")
    print(f"\n=== Bottom-Up Root Synthesis ===\n{res['final_report']}")

