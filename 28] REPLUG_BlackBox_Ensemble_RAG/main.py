"""
28] REPLUG (Retrieval-Augmented Black-Box Language Models)
==========================================================
Pipeline:
1. Treats the OpenAI Language Model strictly as a frozen "Black Box".
2. Retrieves Top-K documents d_1, ..., d_k and computes retrieval likelihoods:
   P(d_i | x) = exp(sim(x, d_i) / tau) / sum_j exp(sim(x, d_j) / tau)
3. Input Reformulation & Parallel LM Predictions:
   Prepends each document d_i individually to the input query [d_i ; x] and queries
   the black-box LLM in parallel.
4. Ensemble Predictions:
   Combines the parallel black-box predictions weighted by their retrieval probability P(d_i | x)
   and model confidence.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class REPLUGBlackBoxRAG:
    def __init__(self, documents: List[str], temperature_tau: float = 0.1, emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.tau = temperature_tau
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.doc_embeddings = [np.array(d.embedding) for d in resp.data]

    def retrieve_with_probabilities(self, query: str, top_k: int = 3) -> List[Dict]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[query]).data[0].embedding
        )
        sims = np.array(
            [
                float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10))
                for d_emb in self.doc_embeddings
            ]
        )
        top_idx = np.argsort(sims)[::-1][:top_k]
        top_sims = sims[top_idx]

        # Softmax with hyperparameter tau
        logits = (top_sims - np.max(top_sims)) / self.tau
        probs = np.exp(logits) / np.sum(np.exp(logits))

        return [
            {"doc_id": int(idx), "doc": self.documents[idx], "sim": float(s), "p_d_given_x": float(p)}
            for idx, s, p in zip(top_idx, top_sims, probs)
        ]

    def parallel_blackbox_predict(self, query: str, retrieved: List[Dict]) -> List[Dict]:
        """Prepends each retrieved doc individually to the black-box LLM input."""
        predictions = []
        for item in retrieved:
            resp = self.client.chat.completions.create(
                model=self.chat_model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a frozen black-box LLM. Given the prepended context document and query, "
                            'output your predicted answer and confidence in [0.0, 1.0]. Return JSON: {"answer": str, "lm_confidence": float}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"[Prepended Document]: {item['doc']}\n\n[Input Query]: {query}",
                    },
                ],
                temperature=0.0,
            )
            parsed = json.loads(resp.choices[0].message.content)
            ensemble_weight = item["p_d_given_x"] * float(parsed.get("lm_confidence", 0.5))
            predictions.append(
                {
                    **item,
                    "candidate_answer": parsed.get("answer", ""),
                    "lm_confidence": float(parsed.get("lm_confidence", 0.5)),
                    "ensemble_weight": ensemble_weight,
                }
            )
        return predictions

    def query(self, user_query: str, top_k: int = 3) -> Dict:
        retrieved = self.retrieve_with_probabilities(user_query, top_k=top_k)
        parallel_preds = self.parallel_blackbox_predict(user_query, retrieved)
        parallel_preds.sort(key=lambda x: x["ensemble_weight"], reverse=True)

        # Ensemble aggregation over weighted parallel predictions
        summary_table = "\n".join(
            [
                f"- Weight={p['ensemble_weight']:.3f} | Prediction: {p['candidate_answer']}"
                for p in parallel_preds
            ]
        )
        final_resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "Combine the probability-weighted REPLUG ensemble predictions into a single final answer.",
                },
                {"role": "user", "content": f"Query: {user_query}\n\nWeighted Predictions:\n{summary_table}"},
            ],
            temperature=0.0,
        )
        return {
            "parallel_predictions": parallel_preds,
            "ensembled_answer": final_resp.choices[0].message.content.strip(),
        }


if __name__ == "__main__":
    corpus = [
        "Protocol Chimera uses a 512-bit Blake3 hash function for block header verification.",
        "Protocol Chimera finality takes 2.4 seconds using optimistic pipelined BFT.",
        "Unrelated note: Tea is brewed at 95 degrees Celsius.",
    ]

    replug = REPLUGBlackBoxRAG(corpus)
    res = replug.query("What hash function and finality time does Protocol Chimera have?")
    print("REPLUG Parallel Black-Box Predictions:")
    for p in res["parallel_predictions"]:
        print(f" - Doc #{p['doc_id']} | P(d|x)={p['p_d_given_x']:.3f} | EnsembleWeight={p['ensemble_weight']:.3f} -> {p['candidate_answer']}")
    print(f"\nFinal Ensembled Output:\n{res['ensembled_answer']}")

