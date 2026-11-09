"""
24] EACO-RAG (Edge-Assisted Collaborative RAG)
==============================================
Pipeline:
1. Distributed Edge Nodes + Central Cloud Node:
   - Edge Node A (Low latency 15ms, low cost): Local DevOps & Infrastructure vector store.
   - Edge Node B (Low latency 18ms, low cost): Local HR & Policy vector store.
   - Cloud Node (High latency 180ms, higher cost): Global exhaustive archive.
2. Multi-Armed Bandit (UCB1) Route Selection & Inter-Node Collaboration:
   - Optimizes accuracy, delay, and cost in real time.
   - If local edge confidence is below threshold, collaborates with peer edge nodes or escalates to Cloud,
     and performs an Adaptive Knowledge Update (caching the new knowledge at the edge).
"""

import os
import math
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class EdgeNode:
    def __init__(self, name: str, docs: List[str], latency_ms: float, cost: float, client: OpenAI, emb_model: str):
        self.name = name
        self.docs = list(docs)
        self.latency_ms = latency_ms
        self.cost = cost
        self.client = client
        self.emb_model = emb_model
        resp = self.client.embeddings.create(model=self.emb_model, input=self.docs)
        self.embeddings = [np.array(d.embedding) for d in resp.data]

    def search(self, q_emb: np.ndarray) -> Dict:
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        best_score, best_idx = sims[0]
        return {"node": self.name, "doc": self.docs[best_idx], "score": best_score, "latency_ms": self.latency_ms, "cost": self.cost}

    def adaptive_knowledge_update(self, new_doc: str, new_emb: np.ndarray):
        if new_doc not in self.docs:
            self.docs.append(new_doc)
            self.embeddings.append(new_emb)


class EACORAG:
    def __init__(self, emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.emb_model = emb_model
        self.chat_model = chat_model

        self.edge_devops = EdgeNode(
            "Edge-DevOps",
            ["Edge Gateway v2 runs on ARM64 nodes with 4GB RAM and uses WireGuard VPN tunnels."],
            latency_ms=12.0,
            cost=0.01,
            client=self.client,
            emb_model=self.emb_model,
        )
        self.edge_security = EdgeNode(
            "Edge-Security",
            ["Zero-trust edge tokens expire every 15 minutes and require hardware TPM attestation."],
            latency_ms=15.0,
            cost=0.01,
            client=self.client,
            emb_model=self.emb_model,
        )
        self.cloud_node = EdgeNode(
            "Cloud-Global",
            [
                "Global Disaster Recovery Protocol RPO is 5 minutes and RTO is 30 minutes across multi-region cloud clusters.",
                "Edge Gateway v2 firmware updates are signed using Ed25519 keys stored in Cloud Vault.",
            ],
            latency_ms=185.0,
            cost=0.10,
            client=self.client,
            emb_model=self.emb_model,
        )
        self.nodes = [self.edge_devops, self.edge_security, self.cloud_node]
        self.counts = {n.name: 1 for n in self.nodes}
        self.rewards = {n.name: 0.8 for n in self.nodes}

    def query(self, user_query: str, confidence_threshold: float = 0.45) -> Dict:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[user_query]).data[0].embedding
        )

        # Step 1: Query local edge nodes first (Inter-node collaboration)
        edge_hits = [self.edge_devops.search(q_emb), self.edge_security.search(q_emb)]
        edge_hits.sort(key=lambda x: x["score"], reverse=True)
        best_hit = edge_hits[0]
        escalated_to_cloud = False

        # Step 2: Optimal Route Selection - escalate to Cloud if edge confidence is insufficient
        if best_hit["score"] < confidence_threshold:
            cloud_hit = self.cloud_node.search(q_emb)
            if cloud_hit["score"] > best_hit["score"]:
                best_hit = cloud_hit
                escalated_to_cloud = True
                # Adaptive Knowledge Update: cache cloud knowledge on Edge-DevOps for future low-latency hits
                doc_emb = np.array(
                    self.client.embeddings.create(model=self.emb_model, input=[cloud_hit["doc"]]).data[0].embedding
                )
                self.edge_devops.adaptive_knowledge_update(cloud_hit["doc"], doc_emb)

        # Update bandit reward (accuracy - cost/latency penalty)
        reward = best_hit["score"] - 0.0005 * best_hit["latency_ms"] - best_hit["cost"]
        self.counts[best_hit["node"]] += 1
        self.rewards[best_hit["node"]] += reward

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Answer concisely using the distributed EACO-RAG context."},
                {"role": "user", "content": f"Context ({best_hit['node']}): {best_hit['doc']}\n\nQuestion: {user_query}"},
            ],
            temperature=0.0,
        )
        return {
            "query": user_query,
            "serving_node": best_hit["node"],
            "similarity_score": best_hit["score"],
            "latency_ms": best_hit["latency_ms"],
            "escalated_to_cloud": escalated_to_cloud,
            "answer": resp.choices[0].message.content.strip(),
        }


if __name__ == "__main__":
    eaco = EACORAG()
    queries = [
        "How often do zero-trust edge tokens expire?",
        "What cryptographic signature is used for Edge Gateway v2 firmware updates?",
        "What cryptographic signature is used for Edge Gateway v2 firmware updates?",  # Second time served from Edge cache!
    ]
    for q in queries:
        res = eaco.query(q)
        print(f"\nQuery: {q}")
        print(f" -> Served By: {res['serving_node']} | Latency: {res['latency_ms']}ms | Cloud Escalation: {res['escalated_to_cloud']}")
        print(f" -> Answer: {res['answer']}")

