"""
20] Memo RAG
============
Pipeline:
1. Global Memory Model (Light LLM: gpt-4o-mini):
   - Maintains a compressed global memory synopsis of the entire database.
   - Given an ambiguous or complex user query, generates draft answers and specific
     "Retrieval Clues" that bridge the user's intent to exact database terminology.
2. Clue-Based Retriever:
   - Uses the generated clues to fetch high-precision passages from the vector store.
3. Heavy Answer Generator (Heavy LLM: gpt-4o):
   - Synthesizes a comprehensive final answer from the retrieved evidence.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class MemoRAG:
    def __init__(
        self,
        documents: List[str],
        emb_model: str = "text-embedding-3-small",
        light_memory_model: str = "gpt-4o-mini",
        heavy_generator_model: str = "gpt-4o",
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.emb_model = emb_model
        self.light_model = light_memory_model
        self.heavy_model = heavy_generator_model

        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.doc_embeddings = [np.array(d.embedding) for d in resp.data]
        self.global_memory_state = self._build_global_memory(documents)

    def _build_global_memory(self, documents: List[str]) -> str:
        """Compresses the corpus into a compact global memory state using the Light LLM."""
        joined = "\n".join([f"- {d}" for d in documents])
        resp = self.client.chat.completions.create(
            model=self.light_model,
            messages=[
                {
                    "role": "system",
                    "content": "Create a compact global memory index summarizing key entities, code names, and topics in these documents.",
                },
                {"role": "user", "content": joined},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    def generate_memory_clues(self, query: str) -> Dict:
        """Light Memory LLM generates a draft answer and explicit search clues from global memory."""
        resp = self.client.chat.completions.create(
            model=self.light_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the MemoRAG Light Memory Model. Using the Global Memory State, interpret the "
                        "ambiguous user query, produce a memory draft answer, and generate 2 specific search clues "
                        'to retrieve exact passages. Return JSON: {"memory_draft": str, "clues": [str, str]}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Global Memory State:\n{self.global_memory_state}\n\nAmbiguous Query: {query}",
                },
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    def retrieve_from_clues(self, clues: List[str], top_k_per_clue: int = 1) -> List[str]:
        retrieved = []
        for clue in clues:
            q_emb = np.array(
                self.client.embeddings.create(model=self.emb_model, input=[clue]).data[0].embedding
            )
            sims = [
                (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
                for i, d_emb in enumerate(self.doc_embeddings)
            ]
            sims.sort(key=lambda x: x[0], reverse=True)
            for _, idx in sims[:top_k_per_clue]:
                if self.documents[idx] not in retrieved:
                    retrieved.append(self.documents[idx])
        return retrieved

    def query(self, user_query: str) -> Dict:
        clue_data = self.generate_memory_clues(user_query)
        clues = clue_data.get("clues", [user_query])
        retrieved_docs = self.retrieve_from_clues(clues, top_k_per_clue=1)

        ctx_str = "\n".join([f"- {d}" for d in retrieved_docs])
        final_resp = self.client.chat.completions.create(
            model=self.heavy_model,
            messages=[
                {"role": "system", "content": "You are the Heavy LLM Generator in MemoRAG. Produce a precise, comprehensive final answer."},
                {"role": "user", "content": f"Retrieved Context:\n{ctx_str}\n\nUser Question: {user_query}"},
            ],
            temperature=0.0,
        )
        return {
            "query": user_query,
            "memory_draft": clue_data.get("memory_draft"),
            "generated_clues": clues,
            "retrieved_context": retrieved_docs,
            "final_answer": final_resp.choices[0].message.content.strip(),
        }


if __name__ == "__main__":
    corpus = [
        "Project Obsidian migrated legacy COBOL mainframe ledgers to a distributed CockroachDB cluster in Q2 2025, cutting batch settlement time from 6 hours to 40 seconds.",
        "Incident Report #881: A BGP route leak in the Frankfurt data center caused a 14-minute packet loss spike during the Project Obsidian cutover.",
        "Employee Wellness Program offers a $600 annual stipend for gym memberships.",
    ]

    memorag = MemoRAG(corpus)
    res = memorag.query("What went wrong in Europe when we modernized the old mainframe banking system, and how much faster did settlements get?")
    print("Memory Model Draft:", res["memory_draft"])
    print("Generated Retrieval Clues:", res["generated_clues"])
    print(f"\nFinal Answer (Heavy LLM):\n{res['final_answer']}")

