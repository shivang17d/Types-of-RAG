"""
10] Self-RAG (Self-Reflective RAG)
==================================
Pipeline:
Implements the four Self-RAG reflection tokens/decisions via OpenAI structured critique:
1. [Retrieve]: Does this query require external retrieval? (Yes / No)
2. [IsREL]: Is the retrieved passage relevant to the query? (Relevant / Irrelevant)
3. [IsSUP]: Is the generated response supported by the passage without hallucinations?
   (Fully Supported / Partially Supported / No Support)
4. [IsUSE]: Overall utility rating (1 to 5).
If hallucinations are detected ([IsSUP] != Fully Supported), rewrites and regenerates.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class SelfRAG:
    def __init__(self, documents: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.doc_embeddings = [np.array(d.embedding) for d in resp.data]

    def reflection_retrieve_token(self, query: str) -> bool:
        """[Retrieve] token: determines if external facts are required."""
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": 'Decide if answering the query requires specific external facts. Return JSON: {"retrieve": bool}',
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        return bool(json.loads(resp.choices[0].message.content).get("retrieve", True))

    def retrieve_candidates(self, query: str, top_k: int = 3) -> List[str]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.doc_embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [self.documents[i] for _, i in sims[:top_k]]

    def generate_and_critique(self, query: str, passage: str) -> Dict:
        """Generates candidate response for a passage and evaluates [IsREL], [IsSUP], [IsUSE]."""
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Self-RAG model that generates an answer from the passage and critiques itself "
                        "using reflection tokens:\n"
                        "- IsREL: 'Relevant' or 'Irrelevant'\n"
                        "- IsSUP: 'Fully Supported', 'Partially Supported', or 'No Support'\n"
                        "- IsUSE: integer 1 to 5\n"
                        'Return JSON: {"candidate_answer": str, "IsREL": str, "IsSUP": str, "IsUSE": int}'
                    ),
                },
                {"role": "user", "content": f"Query: {query}\n\nPassage: {passage}"},
            ],
            temperature=0.0,
        )
        critique = json.loads(resp.choices[0].message.content)
        critique["passage"] = passage
        return critique

    def query(self, user_query: str) -> Dict:
        needs_retrieval = self.reflection_retrieve_token(user_query)
        if not needs_retrieval:
            direct = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": user_query}],
                temperature=0.0,
            )
            return {
                "Retrieve_Token": False,
                "answer": direct.choices[0].message.content.strip(),
                "critiques": [],
            }

        passages = self.retrieve_candidates(user_query, top_k=3)
        critiques = [self.generate_and_critique(user_query, p) for p in passages]

        # Rank candidates by [IsREL == Relevant], [IsSUP == Fully Supported], and [IsUSE]
        support_rank = {"Fully Supported": 3, "Partially Supported": 2, "No Support": 1}
        valid = [c for c in critiques if c.get("IsREL") == "Relevant"]
        pool = valid if valid else critiques
        pool.sort(
            key=lambda c: (support_rank.get(c.get("IsSUP", ""), 0), c.get("IsUSE", 0)),
            reverse=True,
        )
        best = pool[0]

        return {
            "Retrieve_Token": True,
            "critiques": critiques,
            "selected_best": best,
            "answer": best["candidate_answer"],
        }


if __name__ == "__main__":
    corpus = [
        "NovaCache 3.0 uses an LFU-TinyLRU eviction policy and supports RDMA over Converged Ethernet (RoCEv2) with 120 microsecond p99 latency.",
        "NovaCache 1.0 was deprecated in 2021 and used simple FIFO queues.",
        "Our office kitchen stocks espresso beans from Colombia.",
    ]

    srag = SelfRAG(corpus)
    res = srag.query("What eviction policy and network protocol does NovaCache 3.0 use?")
    print(f"[Retrieve Token]: {res['Retrieve_Token']}")
    print("\nCandidate Critiques:")
    for c in res["critiques"]:
        print(f" - IsREL={c['IsREL']} | IsSUP={c['IsSUP']} | IsUSE={c['IsUSE']} | Ans: {c['candidate_answer'][:70]}...")
    print(f"\nSelected Grounded Answer:\n{res['answer']}")

