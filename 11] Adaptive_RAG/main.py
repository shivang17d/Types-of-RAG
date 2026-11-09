"""
11] Adaptive RAG
================
Pipeline:
Dynamically assesses query complexity and internal model confidence (honesty probe)
to route the query to one of three strategies:
1. NO_RETRIEVAL: Simple general knowledge query the LLM knows with high confidence.
2. SINGLE_HOP_RAG: Specific factual lookup requiring a single retrieval step.
3. MULTI_HOP_RAG: Complex analytical query requiring query decomposition & multi-step retrieval.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class AdaptiveRAG:
    def __init__(self, documents: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.doc_embeddings = [np.array(d.embedding) for d in resp.data]

    def classify_and_probe(self, query: str) -> Dict:
        """Routes query to NO_RETRIEVAL, SINGLE_HOP, or MULTI_HOP based on complexity & honesty probe."""
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Adaptive RAG Router with an internal honesty probe.\n"
                        "Classify the query into one of:\n"
                        "- 'NO_RETRIEVAL': Universal general knowledge (e.g., math, common capitals) where internal confidence >= 0.95.\n"
                        "- 'SINGLE_HOP': Requires looking up a single proprietary/external fact.\n"
                        "- 'MULTI_HOP': Requires comparing or chaining multiple facts across documents.\n"
                        'Return JSON: {"route": str, "internal_confidence": float, "sub_queries": [str]}'
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    def retrieve(self, query: str, top_k: int = 1) -> List[str]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.doc_embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [self.documents[i] for _, i in sims[:top_k]]

    def run(self, query: str) -> Dict:
        routing = self.classify_and_probe(query)
        route = routing.get("route", "SINGLE_HOP")
        retrieved_context = []

        if route == "NO_RETRIEVAL":
            ans = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": query}],
                temperature=0.0,
            ).choices[0].message.content.strip()
            return {"query": query, "route": route, "confidence": routing["internal_confidence"], "context": [], "answer": ans}

        elif route == "SINGLE_HOP":
            retrieved_context = self.retrieve(query, top_k=2)

        else:  # MULTI_HOP
            sub_queries = routing.get("sub_queries") or [query]
            for sq in sub_queries:
                for doc in self.retrieve(sq, top_k=1):
                    if doc not in retrieved_context:
                        retrieved_context.append(doc)

        ctx_str = "\n".join([f"- {c}" for c in retrieved_context])
        ans = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Answer using the retrieved context."},
                {"role": "user", "content": f"Context:\n{ctx_str}\n\nQuestion: {query}"},
            ],
            temperature=0.0,
        ).choices[0].message.content.strip()

        return {
            "query": query,
            "route": route,
            "confidence": routing.get("internal_confidence"),
            "context": retrieved_context,
            "answer": ans,
        }


if __name__ == "__main__":
    corpus = [
        "Service Alpha has a monthly SLA uptime target of 99.95% and processes 12,000 req/s.",
        "Service Beta has a monthly SLA uptime target of 99.99% and processes 45,000 req/s.",
        "Outage credits of 15% are issued whenever a service misses its monthly SLA target.",
    ]

    arag = AdaptiveRAG(corpus)
    test_queries = [
        "What is the square root of 144?",
        "What is the SLA uptime target of Service Alpha?",
        "Compare the throughput and SLA targets of Service Alpha and Service Beta, and explain when outage credits apply.",
    ]

    for q in test_queries:
        res = arag.run(q)
        print(f"\nQuery: {q}")
        print(f" -> Selected Route: {res['route']} (Internal Confidence: {res['confidence']})")
        print(f" -> Answer: {res['answer']}")

