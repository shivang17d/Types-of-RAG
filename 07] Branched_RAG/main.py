"""
07] Branched RAG
================
Pipeline:
1. Interprets an open-ended or multifaceted user query by generating multiple distinct
   lines of thought / interpretive branches (e.g., Technical, Financial, Risk/Compliance).
2. Retrieves top matching documents independently for each branch.
3. Generates branch-specific intermediate responses.
4. Compares and synthesizes all branch findings into a unified, well-structured final response.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class BranchedRAG:
    def __init__(self, documents: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.doc_embeddings = [np.array(d.embedding) for d in resp.data]

    def generate_branches(self, query: str, num_branches: int = 3) -> List[Dict[str, str]]:
        """Generates multiple analytical perspectives/sub-queries for the user query."""
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Break the user's question into {num_branches} distinct analytical branches "
                        '(e.g., Technical Specs, Financial/Cost Impact, Operational Risk). Return JSON: '
                        '{"branches": [{"perspective": str, "search_query": str}]}'
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content).get("branches", [])

    def retrieve_for_branch(self, search_query: str, top_k: int = 2) -> List[str]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[search_query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.doc_embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [self.documents[i] for _, i in sims[:top_k]]

    def run(self, query: str) -> Dict:
        branches = self.generate_branches(query, num_branches=3)
        branch_outputs = []

        for b in branches:
            docs = self.retrieve_for_branch(b["search_query"], top_k=2)
            ctx = "\n".join([f"- {d}" for d in docs])
            sub_resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Analyze the context strictly from the '{b['perspective']}' perspective.",
                    },
                    {"role": "user", "content": f"Context:\n{ctx}\n\nSub-Query: {b['search_query']}"},
                ],
                temperature=0.0,
            )
            branch_outputs.append(
                {
                    "perspective": b["perspective"],
                    "search_query": b["search_query"],
                    "retrieved_docs": docs,
                    "branch_analysis": sub_resp.choices[0].message.content.strip(),
                }
            )

        # Synthesize all branches
        synthesis_input = "\n\n".join(
            [f"### Branch ({bo['perspective']}):\n{bo['branch_analysis']}" for bo in branch_outputs]
        )
        final_resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "Synthesize the findings from all analytical branches into a balanced, comprehensive final answer.",
                },
                {"role": "user", "content": f"Original Question: {query}\n\n{synthesis_input}"},
            ],
            temperature=0.0,
        )

        return {
            "query": query,
            "branches": branch_outputs,
            "final_answer": final_resp.choices[0].message.content.strip(),
        }


if __name__ == "__main__":
    corpus = [
        "Technical Specs: Helios Solar Array uses bifacial perovskite cells with 26.8% conversion efficiency and IP68 weatherproofing.",
        "Financial Metrics: Helios Solar Array costs $1.4M upfront with an estimated ROI payback period of 4.2 years and 18% IRR.",
        "Compliance & Risk: Requires UL-1741 grid-tie certification and quarterly inverter thermal inspections to prevent arc faults.",
        "Customer Feedback: 92% satisfaction rating across 40 industrial deployments in arid climates.",
    ]

    rag = BranchedRAG(corpus)
    res = rag.run("Should we adopt the Helios Solar Array for our industrial facility?")
    for b in res["branches"]:
        print(f"\n[Branch: {b['perspective']}] Query: {b['search_query']}")
        print(f" -> {b['branch_analysis']}")
    print(f"\n=== Synthesized Final Answer ===\n{res['final_answer']}")

