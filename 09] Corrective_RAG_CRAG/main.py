"""
09] Corrective RAG (CRAG)
=========================
Pipeline:
1. Retrieve candidate documents from local vector DB.
2. Retrieval Evaluator grades each document's relevance to the query:
   - CORRECT: High confidence match -> Decompose-then-Recompose knowledge refinement.
   - AMBIGUOUS: Partial match -> Combine refined local knowledge + Fallback Web Search.
   - INCORRECT: Low confidence match -> Discard local docs, rewrite query, and run Web Search.
3. Generate final answer using the corrected evidence.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class CorrectiveRAG:
    def __init__(self, local_docs: List[str], web_corpus: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.local_docs = local_docs
        self.web_corpus = web_corpus
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=local_docs)
        self.local_embeddings = [np.array(d.embedding) for d in resp.data]

        web_resp = self.client.embeddings.create(model=self.emb_model, input=web_corpus)
        self.web_embeddings = [np.array(d.embedding) for d in web_resp.data]

    def _vector_search(self, query: str, docs: List[str], embeddings: List[np.ndarray], top_k: int = 2) -> List[str]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [docs[i] for _, i in sims[:top_k]]

    def evaluate_retrieval(self, query: str, retrieved_docs: List[str]) -> Dict:
        """Grades retrieved documents as CORRECT, AMBIGUOUS, or INCORRECT."""
        joined = "\n".join([f"- {d}" for d in retrieved_docs])
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Corrective RAG Retrieval Evaluator. Assess if the retrieved documents "
                        "contain accurate, sufficient information to answer the query. "
                        'Return JSON: {"grade": "CORRECT" | "AMBIGUOUS" | "INCORRECT", '
                        '"confidence": float, "refined_facts": str, "rewritten_web_query": str}'
                    ),
                },
                {"role": "user", "content": f"Query: {query}\n\nRetrieved Local Docs:\n{joined}"},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    def query(self, user_query: str) -> Dict:
        local_hits = self._vector_search(user_query, self.local_docs, self.local_embeddings, top_k=2)
        eval_res = self.evaluate_retrieval(user_query, local_hits)
        grade = eval_res.get("grade", "INCORRECT")

        evidence_sources = []
        if grade == "CORRECT":
            evidence_sources.append(f"[Local Refined Knowledge]: {eval_res['refined_facts']}")
        elif grade == "AMBIGUOUS":
            evidence_sources.append(f"[Local Partial Knowledge]: {eval_res['refined_facts']}")
            web_hits = self._vector_search(eval_res["rewritten_web_query"], self.web_corpus, self.web_embeddings, top_k=2)
            evidence_sources.extend([f"[Web Fallback]: {w}" for w in web_hits])
        else:  # INCORRECT
            web_hits = self._vector_search(eval_res["rewritten_web_query"], self.web_corpus, self.web_embeddings, top_k=2)
            evidence_sources.extend([f"[Web Fallback]: {w}" for w in web_hits])

        context_str = "\n".join(evidence_sources)
        ans_resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Answer accurately using the verified corrective context."},
                {"role": "user", "content": f"Verified Evidence:\n{context_str}\n\nQuestion: {user_query}"},
            ],
            temperature=0.0,
        )

        return {
            "query": user_query,
            "local_retrieved": local_hits,
            "evaluator_grade": grade,
            "rewritten_web_query": eval_res.get("rewritten_web_query"),
            "final_evidence": evidence_sources,
            "answer": ans_resp.choices[0].message.content.strip(),
        }


if __name__ == "__main__":
    # Local DB only knows about Python 3.10
    local_kb = [
        "Python 3.10 introduced structural pattern matching (match/case statements) in October 2021.",
        "Python virtual environments isolate package dependencies per project.",
    ]
    # Simulated external Web Search corpus has newer info
    web_kb = [
        "Python 3.13 introduced an experimental free-threaded mode (PEP 703) that disables the Global Interpreter Lock (GIL) and a basic Just-In-Time (JIT) compiler.",
        "PEP 703 allows Python threads to run in true parallel on multi-core CPUs.",
    ]

    crag = CorrectiveRAG(local_kb, web_kb)
    res = crag.query("What major concurrency and compiler features were added in Python 3.13?")
    print(f"Evaluator Grade: {res['evaluator_grade']}")
    print(f"Rewritten Web Query: {res['rewritten_web_query']}")
    print("Final Evidence Used:")
    for ev in res["final_evidence"]:
        print(f"  {ev}")
    print(f"\nFinal Answer:\n{res['answer']}")

