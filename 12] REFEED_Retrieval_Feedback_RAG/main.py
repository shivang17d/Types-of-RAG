"""
12] REFEED (Retrieval Feedback) RAG
===================================
Pipeline:
Refines model outputs using retrieval feedback WITHOUT fine-tuning:
1. Initial Generation: Generate multiple initial answers from the LLM's parametric memory.
2. Query-Passage Combination Retrieval: Use [Original Query + Initial Answers] to retrieve
   relevant passages from the vector store.
3. Post-Retrieval Refinement: Generate updated answers conditioned on the retrieved passages.
4. Ensemble Ranking Evaluation: Rank pre- and post-retrieval outputs to produce the final answer.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class REFEEDRAG:
    def __init__(self, documents: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.doc_embeddings = [np.array(d.embedding) for d in resp.data]

    def generate_initial_drafts(self, query: str, n: int = 2) -> List[str]:
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Provide a concise initial answer to the question."},
                {"role": "user", "content": query},
            ],
            n=n,
            temperature=0.6,
        )
        return [c.message.content.strip() for c in resp.choices]

    def retrieve_with_feedback(self, query: str, initial_drafts: List[str], top_k: int = 2) -> List[str]:
        combined_search_input = query + " " + " ".join(initial_drafts)
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[combined_search_input]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.doc_embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [self.documents[i] for _, i in sims[:top_k]]

    def ensemble_refine_and_rank(self, query: str, initial_drafts: List[str], passages: List[str]) -> Dict:
        passages_str = "\n".join([f"- {p}" for p in passages])
        drafts_str = "\n".join([f"Draft {i+1}: {d}" for i, d in enumerate(initial_drafts)])

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the REFEED Retrieval Feedback Evaluator. Compare the initial pre-retrieval drafts "
                        "against the retrieved passages, refine the response to fix any inaccuracies, and output JSON: "
                        '{"corrections_made": str, "final_refined_answer": str}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\nInitial Pre-Retrieval Drafts:\n{drafts_str}\n\n"
                        f"Retrieved Feedback Passages:\n{passages_str}"
                    ),
                },
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    def run(self, query: str) -> Dict:
        initial_drafts = self.generate_initial_drafts(query, n=2)
        retrieved_passages = self.retrieve_with_feedback(query, initial_drafts, top_k=2)
        refined = self.ensemble_refine_and_rank(query, initial_drafts, retrieved_passages)

        return {
            "query": query,
            "initial_drafts": initial_drafts,
            "retrieved_passages": retrieved_passages,
            "corrections_made": refined.get("corrections_made"),
            "final_answer": refined.get("final_refined_answer"),
        }


if __name__ == "__main__":
    corpus = [
        "The Hyperion-X7 chip uses a 3nm gate-all-around (GAA) process node and features 144 tensor cores with 192GB HBM3e memory.",
        "Hyperion-X7 thermal design power (TDP) is 650W at peak boost clock of 2.8 GHz.",
    ]

    refeed = REFEEDRAG(corpus)
    res = refeed.run("What are the memory specs, process node, and TDP of the Hyperion-X7 chip?")
    print("Initial Pre-Retrieval Drafts:")
    for d in res["initial_drafts"]:
        print(f" - {d}")
    print(f"\nCorrections Made via Retrieval Feedback:\n {res['corrections_made']}")
    print(f"\nFinal REFEED Answer:\n {res['final_answer']}")

