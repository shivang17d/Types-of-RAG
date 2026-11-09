"""
27] Attention-based RAG (ATLAS / Fusion-in-Decoder)
===================================================
Pipeline:
Implements the ATLAS (Meta AI) Fusion-in-Decoder (FiD) architecture with an updatable
document index:
1. Dual-Encoder Retriever identifies Top-K passages from an index that can be hot-updated
   without retraining the generator.
2. Independent Encoder Pass: Each (Query + Passage_i) pair is encoded independently to
   extract query-focused representations and attention weights.
3. Fusion-in-Decoder (FiD) Step: Fuses all encoded representations weighted by their
   cross-attention salience to generate the final answer.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class ATLASAttentionRAG:
    def __init__(self, initial_docs: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.emb_model = emb_model
        self.chat_model = chat_model
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.hot_update_index(initial_docs)

    def hot_update_index(self, new_docs: List[str]):
        """Updatable document index without retraining."""
        resp = self.client.embeddings.create(model=self.emb_model, input=new_docs)
        for doc, item in zip(new_docs, resp.data):
            self.documents.append(doc)
            self.embeddings.append(np.array(item.embedding))

    def dual_encoder_retrieve(self, query: str, top_k: int = 3) -> List[str]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [self.documents[i] for _, i in sims[:top_k]]

    def encode_query_passage_pair(self, query: str, passage: str) -> Dict:
        """Encodes (Query + Passage_i) independently and computes attention salience weight."""
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an ATLAS FiD Encoder head. Encode the (Query, Passage) pair into a concise "
                        "representation of relevant facts and assign a cross-attention weight in [0.0, 1.0]. "
                        'Return JSON: {"encoded_representation": str, "attention_weight": float}'
                    ),
                },
                {"role": "user", "content": f"Query: {query}\nPassage: {passage}"},
            ],
            temperature=0.0,
        )
        parsed = json.loads(resp.choices[0].message.content)
        return {
            "passage": passage,
            "encoded_representation": parsed.get("encoded_representation", ""),
            "attention_weight": float(parsed.get("attention_weight", 0.0)),
        }

    def fusion_in_decoder_generate(self, query: str, top_k: int = 3) -> Dict:
        passages = self.dual_encoder_retrieve(query, top_k=top_k)
        encoded_pairs = [self.encode_query_passage_pair(query, p) for p in passages]
        encoded_pairs.sort(key=lambda x: x["attention_weight"], reverse=True)

        fused_context = "\n".join(
            [
                f"[Attention Weight: {ep['attention_weight']:.2f}] {ep['encoded_representation']}"
                for ep in encoded_pairs
            ]
        )
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are the ATLAS Fusion-in-Decoder (FiD) reader. Synthesize the final answer by attending to the weighted encoder representations.",
                },
                {"role": "user", "content": f"Fused Representations:\n{fused_context}\n\nQuery: {query}"},
            ],
            temperature=0.0,
        )
        return {
            "query": query,
            "encoded_passages": encoded_pairs,
            "answer": resp.choices[0].message.content.strip(),
        }


if __name__ == "__main__":
    initial_kb = [
        "AetherNet v1 supported 10Gbps optical transceivers over single-mode fiber up to 10km.",
        "AetherNet uses Forward Error Correction (RS-FEC) to maintain bit error rates below 1e-15.",
    ]
    atlas = ATLASAttentionRAG(initial_kb)

    # Hot-update index with latest v2 release notes without retraining
    atlas.hot_update_index(
        ["AetherNet v2 (released 2026) upgrades optical links to 800Gbps coherent pluggable optics over 80km."]
    )

    res = atlas.fusion_in_decoder_generate("What is the link speed and reach of AetherNet v2, and what error correction does AetherNet use?")
    print("Fusion-in-Decoder Encoded Representations:")
    for ep in res["encoded_passages"]:
        print(f" - [Attn: {ep['attention_weight']:.2f}] {ep['encoded_representation']}")
    print(f"\nATLAS FiD Answer:\n{res['answer']}")

