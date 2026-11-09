"""
03] Advanced RAG
================
Pipeline:
1. Pre-Retrieval:
   - Query Expansion / Rewriting
   - Metadata Filter Extraction
2. Retrieval:
   - Vector similarity search with metadata pre-filtering
3. Post-Retrieval:
   - LLM Cross-Scoring / Reranking
   - Context Compression (extracting only relevant sentences)
   - Lost-in-the-Middle Reordering (placing highest-ranked chunks at the boundaries)
4. Generation:
   - Final grounded answer synthesis via OpenAI.
"""

import os
import json
from typing import List, Dict, Optional
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class AdvancedRAG:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.records: List[Dict] = []

    def index_documents(self, docs: List[Dict]):
        """Each doc is {'text': str, 'metadata': {'category': str, 'year': int}}."""
        texts = [d["text"] for d in docs]
        emb_resp = self.client.embeddings.create(model=self.embedding_model, input=texts)
        for i, doc in enumerate(docs):
            self.records.append(
                {
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "embedding": emb_resp.data[i].embedding,
                }
            )

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        va, vb = np.array(a), np.array(b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))

    def pre_retrieval_rewrite(self, query: str) -> str:
        """Rewrites user query to maximize semantic retrieval recall."""
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite the user query into a clear, keyword-rich search query for technical retrieval. Return only the rewritten query.",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    def retrieve(self, rewritten_query: str, category_filter: Optional[str] = None, top_k: int = 4) -> List[Dict]:
        q_emb = self.client.embeddings.create(
            model=self.embedding_model, input=[rewritten_query]
        ).data[0].embedding

        candidates = []
        for rec in self.records:
            if category_filter and rec["metadata"].get("category") != category_filter:
                continue
            sim = self.cosine_similarity(q_emb, rec["embedding"])
            candidates.append({**rec, "vector_score": sim})

        candidates.sort(key=lambda x: x["vector_score"], reverse=True)
        return candidates[:top_k]

    def post_retrieval_rerank_and_compress(self, query: str, candidates: List[Dict], top_n: int = 3) -> List[Dict]:
        """Reranks candidates using LLM relevance scoring and compresses each chunk."""
        reranked = []
        for cand in candidates:
            resp = self.client.chat.completions.create(
                model=self.chat_model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Score the relevance of the passage to the query from 0.0 to 10.0, and extract ONLY "
                            "the key sentences needed to answer the query. Return JSON: "
                            '{"relevance_score": float, "compressed_text": str}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nPassage: {cand['text']}",
                    },
                ],
                temperature=0.0,
            )
            parsed = json.loads(resp.choices[0].message.content)
            reranked.append(
                {
                    "original_text": cand["text"],
                    "compressed_text": parsed.get("compressed_text", cand["text"]),
                    "rerank_score": float(parsed.get("relevance_score", 0.0)),
                }
            )

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]

    @staticmethod
    def lost_in_the_middle_reorder(chunks: List[Dict]) -> List[Dict]:
        """Places highest-ranked chunks at the beginning and end of the context window."""
        if len(chunks) <= 2:
            return chunks
        left, right = [], []
        for i, item in enumerate(chunks):
            if i % 2 == 0:
                left.append(item)
            else:
                right.insert(0, item)
        return left + right

    def run(self, query: str, category_filter: Optional[str] = None) -> Dict:
        rewritten = self.pre_retrieval_rewrite(query)
        candidates = self.retrieve(rewritten, category_filter=category_filter, top_k=4)
        reranked = self.post_retrieval_rerank_and_compress(query, candidates, top_n=3)
        reordered = self.lost_in_the_middle_reorder(reranked)

        context_str = "\n".join([f"- {c['compressed_text']}" for c in reordered])
        ans_resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Answer accurately using the provided compressed context."},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
            ],
            temperature=0.0,
        )
        return {
            "original_query": query,
            "rewritten_query": rewritten,
            "reordered_context": reordered,
            "answer": ans_resp.choices[0].message.content.strip(),
        }


if __name__ == "__main__":
    docs = [
        {
            "text": "In 2025, Project Titan reduced GPU cluster power consumption by 38% using liquid immersion cooling. Unrelated cafeteria menus were also updated in Q2.",
            "metadata": {"category": "infrastructure", "year": 2025},
        },
        {
            "text": "Project Titan's networking fabric upgraded to 800Gbps InfiniBand interconnects, cutting all-reduce collective communication overhead by 52%.",
            "metadata": {"category": "infrastructure", "year": 2025},
        },
        {
            "text": "The Titan marketing campaign launched across 14 European cities with billboard ads.",
            "metadata": {"category": "marketing", "year": 2025},
        },
        {
            "text": "Storage tiering on Project Titan uses NVMe-oF caches to sustain 25 GB/s checkpoint writes.",
            "metadata": {"category": "infrastructure", "year": 2025},
        },
    ]

    rag = AdvancedRAG()
    rag.index_documents(docs)

    res = rag.run(
        "How did Project Titan improve power efficiency and network speed?",
        category_filter="infrastructure",
    )
    print(f"Original Query : {res['original_query']}")
    print(f"Rewritten Query: {res['rewritten_query']}")
    print("\nCompressed & Lost-in-the-Middle Reordered Chunks:")
    for c in res["reordered_context"]:
        print(f" [Score: {c['rerank_score']}] {c['compressed_text']}")
    print(f"\nFinal Answer:\n{res['answer']}")

