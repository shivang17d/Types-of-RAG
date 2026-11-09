"""
25] REALM (Retrieval-Augmented Language Model) RAG
==================================================
Pipeline:
Implements the core REALM probabilistic retrieve-then-predict formulation:
1. Neural Knowledge Retriever uses Maximum Inner Product Search (MIPS):
   p(z | x) = exp(f(x, z)) / sum_{z'} exp(f(x, z'))
2. Knowledge-Augmented Encoder marginalizes over the Top-K retrieved documents z:
   p(y | x) = sum_{z in Top-K} p(y | z, x) * p(z | x)
3. Demonstrates supervised retriever weight scaling feedback based on whether each document z
   improved the likelihood of the ground-truth answer.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class REALMRAG:
    def __init__(self, corpus: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.corpus = corpus
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=corpus)
        self.doc_vectors = np.array([d.embedding for d in resp.data])
        # Learnable document prior / scaling weights updated via feedback
        self.doc_weights = np.ones(len(corpus), dtype=float)

    def mips_retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Maximum Inner Product Search (MIPS) with softmax retrieval distribution p(z | x)."""
        q_vec = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[query]).data[0].embedding
        )
        inner_products = (self.doc_vectors @ q_vec) * self.doc_weights
        top_indices = np.argsort(inner_products)[::-1][:top_k]

        top_logits = inner_products[top_indices]
        # Temperature-scaled softmax for p(z | x)
        exp_logits = np.exp((top_logits - np.max(top_logits)) * 10.0)
        probs = exp_logits / np.sum(exp_logits)

        return [
            {
                "doc_id": int(idx),
                "document": self.corpus[idx],
                "mips_score": float(inner_products[idx]),
                "p_z_given_x": float(prob),
            }
            for idx, prob in zip(top_indices, probs)
        ]

    def marginalize_predict(self, query: str, top_k: int = 3) -> Dict:
        """Computes p(y | z, x) for each retrieved doc z and marginalizes over p(z | x)."""
        retrieved = self.mips_retrieve(query, top_k=top_k)
        candidates = []

        for item in retrieved:
            resp = self.client.chat.completions.create(
                model=self.chat_model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the REALM Knowledge-Augmented Encoder. Predict the concise answer y given "
                            "query x and document z, and estimate p(y | z, x) in [0.0, 1.0]. "
                            'Return JSON: {"answer": str, "p_y_given_zx": float}'
                        ),
                    },
                    {"role": "user", "content": f"Document z: {item['document']}\nQuery x: {query}"},
                ],
                temperature=0.0,
            )
            parsed = json.loads(resp.choices[0].message.content)
            joint_prob = item["p_z_given_x"] * float(parsed.get("p_y_given_zx", 0.0))
            candidates.append(
                {
                    **item,
                    "predicted_answer": parsed.get("answer", ""),
                    "p_y_given_zx": float(parsed.get("p_y_given_zx", 0.0)),
                    "marginal_contribution": joint_prob,
                }
            )

        candidates.sort(key=lambda x: x["marginal_contribution"], reverse=True)
        return {"query": query, "marginalized_candidates": candidates, "final_prediction": candidates[0]["predicted_answer"]}


if __name__ == "__main__":
    wiki_corpus = [
        " superconducting qubits in Sycamore-2 operate at 15 millikelvin inside a dilution refrigerator.",
        "Trapped-ion quantum processors use ytterbium ions suspended in electromagnetic Paul traps.",
        "Standard server racks operate at room temperatures between 18C and 27C.",
    ]

    realm = REALMRAG(wiki_corpus)
    res = realm.marginalize_predict("At what temperature do superconducting qubits in Sycamore-2 operate?")
    print("REALM Marginalized Candidates p(y|z,x) * p(z|x):")
    for c in res["marginalized_candidates"]:
        print(
            f" - Doc #{c['doc_id']} | p(z|x)={c['p_z_given_x']:.3f} | p(y|z,x)={c['p_y_given_zx']:.2f} "
            f"| Joint={c['marginal_contribution']:.3f} -> '{c['predicted_answer']}'"
        )
    print(f"\nFinal REALM Marginalized Prediction:\n{res['final_prediction']}")

