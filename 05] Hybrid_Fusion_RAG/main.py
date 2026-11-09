"""
05] Hybrid / Fusion RAG
=======================
Pipeline:
1. Sparse Retrieval (BM25): Captures exact keyword, SKU, and error code matches.
2. Dense Retrieval (OpenAI Embeddings): Captures semantic similarity.
3. Reciprocal Rank Fusion (RRF): Combines rankings from both retrievers:
   RRF_Score(d) = sum_{m} 1 / (k + rank_m(d))
4. OpenAI Generator: Synthesizes the answer from the top fused documents.
"""

import os
import math
import re
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class BM25Retriever:
    """Pure Python BM25 implementation for sparse lexical search."""

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [self._tokenize(d) for d in documents]
        self.doc_freqs = Counter()
        for tokens in self.tokenized_docs:
            for token in set(tokens):
                self.doc_freqs[token] += 1
        self.avgdl = sum(len(t) for t in self.tokenized_docs) / max(1, len(documents))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+(?:-\w+)*", text.lower())

    def search(self, query: str, top_k: int = 4) -> List[int]:
        q_tokens = self._tokenize(query)
        scores = []
        n_docs = len(self.documents)

        for idx, doc_tokens in enumerate(self.tokenized_docs):
            dl = len(doc_tokens)
            tf_counts = Counter(doc_tokens)
            score = 0.0
            for token in q_tokens:
                df = self.doc_freqs.get(token, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                tf = tf_counts[token]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                score += idf * ((tf * (self.k1 + 1)) / denom)
            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in scores[:top_k]]


class HybridFusionRAG:
    def __init__(self, documents: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.emb_model = emb_model
        self.chat_model = chat_model

        self.bm25 = BM25Retriever(documents)
        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.embeddings = [np.array(item.embedding) for item in resp.data]

    def dense_search(self, query: str, top_k: int = 4) -> List[int]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in sims[:top_k]]

    @staticmethod
    def reciprocal_rank_fusion(rank_lists: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
        rrf_scores: Dict[int, float] = {}
        for rank_list in rank_lists:
            for rank, doc_id in enumerate(rank_list):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def query(self, user_query: str, top_k: int = 2) -> Dict:
        sparse_ranks = self.bm25.search(user_query, top_k=len(self.documents))
        dense_ranks = self.dense_search(user_query, top_k=len(self.documents))
        fused = self.reciprocal_rank_fusion([sparse_ranks, dense_ranks])[:top_k]

        top_docs = [{"doc": self.documents[doc_id], "rrf_score": score} for doc_id, score in fused]
        context = "\n".join([f"- {d['doc']}" for d in top_docs])

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Answer accurately using the fused hybrid search context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"},
            ],
            temperature=0.0,
        )
        return {
            "sparse_top": [self.documents[i] for i in sparse_ranks[:2]],
            "dense_top": [self.documents[i] for i in dense_ranks[:2]],
            "fused_top": top_docs,
            "answer": resp.choices[0].message.content.strip(),
        }


if __name__ == "__main__":
    from typing import Tuple

    corpus = [
        "Error code ERR-9042-X occurs when the TLS handshake times out on port 8443 due to expired client certificates.",
        "Network timeouts during secure connections can be resolved by renewing SSL/TLS certificates.",
        "Error code ERR-1008-B indicates a disk quota overflow on the primary write volume.",
        "Billing invoices are generated on the 1st of every month.",
    ]

    rag = HybridFusionRAG(corpus)
    res = rag.query("How do I fix ERR-9042-X SSL connection timeout?")
    print("Fused RRF Top Documents:")
    for d in res["fused_top"]:
        print(f" [RRF: {d['rrf_score']:.5f}] {d['doc']}")
    print(f"\nAnswer:\n{res['answer']}")
