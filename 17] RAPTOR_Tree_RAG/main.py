"""
17] RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
==========================================================================
Pipeline:
1. Leaf Nodes: Embed raw text chunks at Level 0.
2. Recursive Clustering & Summarization:
   - Group semantically similar nodes into clusters using cosine similarity K-means.
   - Summarize each cluster via OpenAI into a higher-level parent node (Level 1, Level 2...).
3. Collapsed Tree Retrieval:
   - Flatten all tree nodes (leaves + intermediate summaries + root summary) into a unified
     vector space so queries can simultaneously retrieve broad thematic summaries and specific details.
"""

import os
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class RAPTORTreeRAG:
    def __init__(self, emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.emb_model = emb_model
        self.chat_model = chat_model
        self.tree_nodes: List[Dict] = []  # Stores {'level': int, 'text': str, 'embedding': np.ndarray}

    def _embed(self, texts: List[str]) -> List[np.ndarray]:
        resp = self.client.embeddings.create(model=self.emb_model, input=texts)
        return [np.array(d.embedding) for d in resp.data]

    def _summarize_cluster(self, texts: List[str]) -> str:
        joined = "\n".join([f"- {t}" for t in texts])
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "Synthesize a cohesive, higher-level abstractive summary of the following cluster of passages.",
                },
                {"role": "user", "content": joined},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _simple_cluster(embeddings: List[np.ndarray], k: int) -> Dict[int, List[int]]:
        """Clusters vectors into k groups using greedy farthest-first centroids."""
        if len(embeddings) <= k:
            return {i: [i] for i in range(len(embeddings))}
        centroids = [embeddings[0]]
        while len(centroids) < k:
            dists = [
                min(float(np.dot(e, c) / (np.linalg.norm(e) * np.linalg.norm(c) + 1e-10)) for c in centroids)
                for e in embeddings
            ]
            # Pick vector with lowest similarity to existing centroids
            centroids.append(embeddings[int(np.argmin(dists))])

        clusters: Dict[int, List[int]] = {i: [] for i in range(k)}
        for idx, emb in enumerate(embeddings):
            sims = [
                float(np.dot(emb, c) / (np.linalg.norm(emb) * np.linalg.norm(c) + 1e-10))
                for c in centroids
            ]
            best_c = int(np.argmax(sims))
            clusters[best_c].append(idx)
        return {c_id: idxs for c_id, idxs in clusters.items() if idxs}

    def build_raptor_tree(self, leaf_chunks: List[str]):
        """Builds a 3-level RAPTOR tree (Level 0: Leaves -> Level 1: Cluster Summaries -> Level 2: Root Summary)."""
        current_texts = leaf_chunks
        current_embs = self._embed(current_texts)

        for t, emb in zip(current_texts, current_embs):
            self.tree_nodes.append({"level": 0, "text": t, "embedding": emb})

        level = 1
        while len(current_texts) > 1:
            num_clusters = max(1, len(current_texts) // 2)
            clusters = self._simple_cluster(current_embs, k=num_clusters)
            next_texts = []
            for _, member_idxs in clusters.items():
                cluster_passages = [current_texts[i] for i in member_idxs]
                summary = self._summarize_cluster(cluster_passages)
                next_texts.append(summary)

            next_embs = self._embed(next_texts)
            for t, emb in zip(next_texts, next_embs):
                self.tree_nodes.append({"level": level, "text": t, "embedding": emb})

            current_texts = next_texts
            current_embs = next_embs
            level += 1

    def query_collapsed_tree(self, query: str, top_k: int = 3) -> Dict:
        """Searches across all abstraction levels simultaneously (Collapsed Tree Retrieval)."""
        q_emb = self._embed([query])[0]
        scored = []
        for node in self.tree_nodes:
            sim = float(
                np.dot(q_emb, node["embedding"])
                / (np.linalg.norm(q_emb) * np.linalg.norm(node["embedding"]) + 1e-10)
            )
            scored.append({"level": node["level"], "text": node["text"], "score": sim})

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_nodes = scored[:top_k]

        context_str = "\n".join([f"[Tree Level {n['level']}] {n['text']}" for n in top_nodes])
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question using the multi-level RAPTOR tree context (combining broad themes and leaf details).",
                },
                {"role": "user", "content": f"RAPTOR Context:\n{context_str}\n\nQuestion: {query}"},
            ],
            temperature=0.0,
        )
        return {"retrieved_tree_nodes": top_nodes, "answer": resp.choices[0].message.content.strip()}


if __name__ == "__main__":
    leaves = [
        "Q1 Security Incident: Phishing campaign targeted contractor VPN credentials, mitigated in 4 hours.",
        "Q2 Security Incident: Misconfigured S3 bucket exposed non-sensitive staging logs; IAM policies were tightened.",
        "Q3 Infrastructure Upgrade: Migrated 120 microservices to zero-trust mTLS service mesh using Istio.",
        "Q4 Infrastructure Upgrade: Deployed hardware security modules (HSM) for automated root key rotation.",
    ]

    raptor = RAPTORTreeRAG()
    raptor.build_raptor_tree(leaves)
    print(f"Built RAPTOR Tree with {len(raptor.tree_nodes)} total nodes across levels 0, 1, and 2.")

    res = raptor.query_collapsed_tree(
        "What is the overarching theme of our security evolution this year, and what specific incident happened in Q1?"
    )
    print("\nRetrieved Tree Nodes:")
    for n in res["retrieved_tree_nodes"]:
        print(f" - [Level {n['level']} | Score: {n['score']:.4f}] {n['text']}")
    print(f"\nRAPTOR Answer:\n{res['answer']}")

