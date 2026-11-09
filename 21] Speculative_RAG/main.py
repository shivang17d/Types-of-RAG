"""
21] Speculative RAG
===================
Pipeline:
1. Document Retrieval & Clustering: Fetches candidate documents and partitions them into
   distinct subsets so each subset offers a focused perspective without context bloat.
2. Parallel Specialist Drafting (Small Fast Model: gpt-4o-mini):
   - Generates multiple candidate drafts + rationales in parallel threads, one per document subset.
3. Generalist Verification & Selection (Larger Verifier Model: gpt-4o):
   - Evaluates all parallel drafts against their supporting subsets and selects/verifies
     the highest-accuracy final response.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class SpeculativeRAG:
    def __init__(
        self,
        documents: List[str],
        emb_model: str = "text-embedding-3-small",
        drafter_model: str = "gpt-4o-mini",
        verifier_model: str = "gpt-4o",
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.emb_model = emb_model
        self.drafter_model = drafter_model
        self.verifier_model = verifier_model

        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.doc_embeddings = [np.array(d.embedding) for d in resp.data]

    def retrieve_subsets(self, query: str, top_k: int = 4, subset_size: int = 2) -> List[List[str]]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.doc_embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        top_docs = [self.documents[i] for _, i in sims[:top_k]]
        return [top_docs[i : i + subset_size] for i in range(0, len(top_docs), subset_size)]

    def _draft_worker(self, draft_id: int, query: str, doc_subset: List[str]) -> Dict:
        """Small Specialist Drafter generates a candidate answer + rationale for a document subset."""
        ctx = "\n".join([f"- {d}" for d in doc_subset])
        resp = self.client.chat.completions.create(
            model=self.drafter_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fast Specialist RAG Drafter. Draft an answer and a self-consistency rationale "
                        'strictly from the provided document subset. Return JSON: {"draft_answer": str, "rationale": str}'
                    ),
                },
                {"role": "user", "content": f"Subset Context:\n{ctx}\n\nQuestion: {query}"},
            ],
            temperature=0.2,
        )
        parsed = json.loads(resp.choices[0].message.content)
        return {
            "draft_id": draft_id,
            "subset": doc_subset,
            "draft_answer": parsed.get("draft_answer", ""),
            "rationale": parsed.get("rationale", ""),
        }

    def verify_and_select(self, query: str, drafts: List[Dict]) -> Dict:
        """Larger Generalist Verifier evaluates all parallel drafts and picks the best one."""
        drafts_summary = "\n\n".join(
            [
                f"Draft #{d['draft_id']}:\nAnswer: {d['draft_answer']}\nRationale: {d['rationale']}\nEvidence: {d['subset']}"
                for d in drafts
            ]
        )
        resp = self.client.chat.completions.create(
            model=self.verifier_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the Generalist Verifier LLM in Speculative RAG. Evaluate the parallel drafts "
                        "for factual grounding, completeness, and rationale strength. Select the best draft and "
                        'provide the final verified answer. Return JSON: {"selected_draft_id": int, '
                        '"verification_reason": str, "final_result": str}'
                    ),
                },
                {"role": "user", "content": f"Query: {query}\n\nCandidate Drafts:\n{drafts_summary}"},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    def run(self, query: str) -> Dict:
        subsets = self.retrieve_subsets(query, top_k=4, subset_size=2)
        with ThreadPoolExecutor(max_workers=len(subsets)) as executor:
            futures = [
                executor.submit(self._draft_worker, idx + 1, query, sub)
                for idx, sub in enumerate(subsets)
            ]
            drafts = [f.result() for f in futures]

        verification = self.verify_and_select(query, drafts)
        return {"query": query, "parallel_drafts": drafts, "verification": verification}


if __name__ == "__main__":
    corpus = [
        "VectorEngine v3 uses HNSW indexing with efConstruction=200 and m=16, achieving 98.4% recall@10 at 1.1ms query latency.",
        "Quantization in VectorEngine v3 supports Scalar Quantization (SQ8) and Product Quantization (PQ), reducing RAM footprint by 75%.",
        "Legacy VectorEngine v1 used flat brute-force L2 scan with 45ms latency.",
        "Customer support hours are 9am to 6pm EST Monday through Friday.",
    ]

    spec_rag = SpeculativeRAG(corpus)
    res = spec_rag.run("How does VectorEngine v3 achieve sub-2ms latency and 75% memory reduction?")
    print("Parallel Specialist Drafts:")
    for d in res["parallel_drafts"]:
        print(f" [Draft #{d['draft_id']}]: {d['draft_answer']}")
    print(f"\nVerifier Selected Draft #{res['verification']['selected_draft_id']}: {res['verification']['verification_reason']}")
    print(f"\nFinal Verified Result:\n{res['verification']['final_result']}")

