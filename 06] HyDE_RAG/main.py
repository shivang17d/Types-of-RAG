"""
06] HyDE (Hypothetical Document Embeddings) RAG
===============================================
Pipeline:
1. Instead of embedding the raw short/ambiguous query directly, prompt OpenAI to generate N
   hypothetical answer documents that look like ideal reference passages.
2. Compute OpenAI embeddings for each hypothetical document and average them into a single
   centroid query vector.
3. Search the real vector database using this HyDE centroid vector.
4. Generate the final grounded response from the retrieved real documents.
"""

import os
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class HyDERAG:
    def __init__(self, documents: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.doc_embeddings = [np.array(d.embedding) for d in resp.data]

    def generate_hypothetical_documents(self, query: str, n_hypotheses: int = 3) -> List[str]:
        """Generates N hypothetical technical passages answering the query."""
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write a concise, detailed technical documentation paragraph that directly "
                        "answers the user's question. Use domain-specific terminology."
                    ),
                },
                {"role": "user", "content": query},
            ],
            n=n_hypotheses,
            temperature=0.7,
        )
        return [choice.message.content.strip() for choice in resp.choices]

    def retrieve_via_hyde(self, query: str, top_k: int = 2) -> Dict:
        hypotheses = self.generate_hypothetical_documents(query, n_hypotheses=3)
        hyp_emb_resp = self.client.embeddings.create(model=self.emb_model, input=hypotheses)
        hyp_vectors = np.array([d.embedding for d in hyp_emb_resp.data])

        # Average the hypothetical embeddings into a single HyDE vector
        hyde_centroid = np.mean(hyp_vectors, axis=0)

        scored = []
        for idx, doc_vec in enumerate(self.doc_embeddings):
            sim = float(
                np.dot(hyde_centroid, doc_vec)
                / (np.linalg.norm(hyde_centroid) * np.linalg.norm(doc_vec) + 1e-10)
            )
            scored.append({"doc": self.documents[idx], "score": sim})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"hypotheses": hypotheses, "retrieved": scored[:top_k]}

    def query(self, user_query: str, top_k: int = 2) -> Dict:
        hyde_res = self.retrieve_via_hyde(user_query, top_k=top_k)
        context = "\n".join([f"- {d['doc']}" for d in hyde_res["retrieved"]])

        ans_resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the user's question using ONLY the real retrieved documents below.",
                },
                {"role": "user", "content": f"Real Documents:\n{context}\n\nQuestion: {user_query}"},
            ],
            temperature=0.0,
        )
        return {
            "query": user_query,
            "hypothetical_docs": hyde_res["hypotheses"],
            "retrieved_real_docs": hyde_res["retrieved"],
            "answer": ans_resp.choices[0].message.content.strip(),
        }


if __name__ == "__main__":
    corpus = [
        "Mitigating gradient explosion in deep recurrent architectures requires clipping the L2 norm of parameter gradients at threshold tau=1.0 prior to the optimizer step.",
        "Batch normalization stabilizes layer activations by normalizing mini-batch mean and variance.",
        "Cosine annealing schedules decay the learning rate smoothly following a half-cosine curve.",
    ]

    rag = HyDERAG(corpus)
    result = rag.query("How to stop weights from blowing up to NaN during training?")
    print("Generated Hypothetical Documents:")
    for i, h in enumerate(result["hypothetical_docs"], 1):
        print(f"  [Hyp {i}] {h}")
    print("\nRetrieved Real Documents:")
    for d in result["retrieved_real_docs"]:
        print(f"  ({d['score']:.4f}) {d['doc']}")
    print(f"\nFinal Grounded Answer:\n{result['answer']}")

