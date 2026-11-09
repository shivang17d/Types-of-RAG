"""
22] AutoRAG (Automated RAG Pipeline Optimization)
=================================================
Pipeline:
Implements a node-based greedy hyperparameter & module optimizer for RAG:
- Node 1: Query Expansion Strategy (Passthrough vs LLM Synonym Expansion)
- Node 2: Retrieval Top-K Configuration (top_k=1 vs top_k=2)
- Node 3: Passage Reranking Strategy (None vs LLM Reranker)
Evaluates each candidate node configuration on a benchmark Q&A pair using an OpenAI
Judge and greedily selects the highest-scoring configuration at each node.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class AutoRAGOptimizer:
    def __init__(self, documents: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.doc_embeddings = [np.array(d.embedding) for d in resp.data]

    def _expand_query(self, query: str, mode: str) -> str:
        if mode == "passthrough":
            return query
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Expand the query with technical terms for better retrieval."},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    def _retrieve(self, query: str, top_k: int) -> List[str]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.doc_embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [self.documents[i] for _, i in sims[:top_k]]

    def _rerank(self, query: str, passages: List[str], mode: str) -> List[str]:
        if mode == "none" or len(passages) <= 1:
            return passages
        scored = []
        for p in passages:
            resp = self.client.chat.completions.create(
                model=self.chat_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": 'Score passage relevance 0-10. Return JSON: {"score": float}'},
                    {"role": "user", "content": f"Query: {query}\nPassage: {p}"},
                ],
                temperature=0.0,
            )
            scored.append((float(json.loads(resp.choices[0].message.content).get("score", 0)), p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]

    def evaluate_pipeline(self, config: Dict, benchmark_q: str, ground_truth: str) -> float:
        q_exp = self._expand_query(benchmark_q, config["query_expansion"])
        docs = self._retrieve(q_exp, top_k=config["top_k"])
        docs = self._rerank(benchmark_q, docs, config["reranker"])
        ctx = "\n".join(docs)

        ans = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Answer concisely using the context."},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {benchmark_q}"},
            ],
            temperature=0.0,
        ).choices[0].message.content.strip()

        judge = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": 'Score how accurately the candidate answer matches the ground truth from 0.0 to 1.0. Return JSON: {"score": float}',
                },
                {"role": "user", "content": f"Ground Truth: {ground_truth}\nCandidate Answer: {ans}"},
            ],
            temperature=0.0,
        )
        return float(json.loads(judge.choices[0].message.content).get("score", 0.0))

    def greedy_optimize(self, benchmark_q: str, ground_truth: str) -> Dict:
        best_config = {"query_expansion": "passthrough", "top_k": 1, "reranker": "none"}
        search_space = {
            "query_expansion": ["passthrough", "llm_expand"],
            "top_k": [1, 2],
            "reranker": ["none", "llm_rerank"],
        }

        history = []
        best_score = -1.0
        for node_name, options in search_space.items():
            node_best_opt = best_config[node_name]
            for opt in options:
                trial_config = {**best_config, node_name: opt}
                score = self.evaluate_pipeline(trial_config, benchmark_q, ground_truth)
                history.append({"node": node_name, "config": trial_config, "score": score})
                if score >= best_score:
                    best_score = score
                    node_best_opt = opt
            best_config[node_name] = node_best_opt

        return {"optimal_config": best_config, "best_score": best_score, "trials": history}


if __name__ == "__main__":
    corpus = [
        "ZephyrOS kernel scheduler uses Earliest Eligible Virtual Deadline First (EEVDF) to guarantee latency bounds.",
        "ZephyrOS memory allocator uses slab caches with per-CPU magazines to prevent lock contention.",
        " ZephyrOS supports POSIX threads and real-time priority inheritance.",
    ]

    autorag = AutoRAGOptimizer(corpus)
    res = autorag.greedy_optimize(
        benchmark_q="How does ZephyrOS schedule tasks and prevent memory lock contention?",
        ground_truth="ZephyrOS uses the EEVDF scheduler for latency bounds and slab caches with per-CPU magazines to avoid lock contention.",
    )
    print("Greedy AutoRAG Optimization Trials:")
    for t in res["trials"]:
        print(f" Node={t['node']:<16} | Config={t['config']} -> Score={t['score']:.2f}")
    print(f"\nDiscovered Optimal RAG Configuration: {res['optimal_config']} (Score: {res['best_score']:.2f})")

