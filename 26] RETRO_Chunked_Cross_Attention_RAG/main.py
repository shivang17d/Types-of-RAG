"""
26] RETRO (Retrieval-Enhanced Transformer with Chunked Cross-Attention)
=======================================================================
Pipeline:
Demonstrates DeepMind's RETRO chunk-wise auto-regressive retrieval mechanism:
1. Splits the ongoing text generation stream into fixed-size token chunks C_1, C_2, ...
2. For each active chunk C_u, retrieves k-Nearest Neighbor chunks N(C_u) AND their
   immediate continuation chunks F(C_u) from the knowledge base.
3. Integrates [Input Chunk C_u + Retrieved Neighbor Pairs (N, F)] via Chunked Cross-Attention
   to predict the next chunk C_{u+1} with high factual precision.
"""

import os
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class RETRORAG:
    def __init__(self, knowledge_sequences: List[List[str]], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        """
        knowledge_sequences: list of multi-chunk sequences, where each chunk N_i is paired
        with its continuation chunk F_i.
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.emb_model = emb_model
        self.chat_model = chat_model
        self.db_pairs: List[Dict[str, str]] = []

        for seq in knowledge_sequences:
            for i in range(len(seq) - 1):
                self.db_pairs.append({"neighbor_chunk": seq[i], "continuation_chunk": seq[i + 1]})

        neighbor_texts = [p["neighbor_chunk"] for p in self.db_pairs]
        resp = self.client.embeddings.create(model=self.emb_model, input=neighbor_texts)
        self.db_embeddings = [np.array(d.embedding) for d in resp.data]

    def retrieve_knn_with_continuations(self, active_chunk: str, k: int = 2) -> List[Dict]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[active_chunk]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.db_embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "similarity": score,
                "neighbor": self.db_pairs[idx]["neighbor_chunk"],
                "continuation": self.db_pairs[idx]["continuation_chunk"],
            }
            for score, idx in sims[:k]
        ]

    def generate_chunk_by_chunk(self, prompt_prefix: str, num_chunks_to_generate: int = 2) -> Dict:
        generated_chunks = [prompt_prefix]
        retrieval_trace = []

        for step in range(num_chunks_to_generate):
            active_chunk = generated_chunks[-1]
            neighbors = self.retrieve_knn_with_continuations(active_chunk, k=2)
            retrieval_trace.append({"step": step + 1, "active_chunk": active_chunk, "retrieved_knn": neighbors})

            ca_context = "\n".join(
                [
                    f"- Matched Neighbor: '{n['neighbor']}' -> Continuation: '{n['continuation']}'"
                    for n in neighbors
                ]
            )
            sequence_so_far = " ".join(generated_chunks)

            resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a RETRO Chunked Cross-Attention decoder. Given the sequence so far and the "
                            "k-NN retrieved neighbor+continuation chunks, generate ONLY the next single sentence chunk."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Sequence So Far:\n{sequence_so_far}\n\nChunked Cross-Attention Neighbors:\n{ca_context}",
                    },
                ],
                temperature=0.0,
            )
            next_chunk = resp.choices[0].message.content.strip()
            generated_chunks.append(next_chunk)

        return {
            "initial_chunk": prompt_prefix,
            "trace": retrieval_trace,
            "full_output": " ".join(generated_chunks),
        }


if __name__ == "__main__":
    kb_sequences = [
        [
            "The Voyager-X deep space probe entered the heliosheath in November 2024.",
            "Upon crossing the termination shock, its plasma wave subsystem recorded a 3.4x jump in electron density.",
            "Mission control subsequently switched telemetry to the high-gain X-band antenna at 1.4 kbps.",
        ],
        [
            "Solar flare activity peaked during the 25th solar cycle.",
            "Geomagnetic storms disrupted HF radio communications across polar aviation routes.",
        ],
    ]

    retro = RETRORAG(kb_sequences)
    res = retro.generate_chunk_by_chunk(
        "The Voyager-X probe crossed into the heliosheath region in late 2024.",
        num_chunks_to_generate=2,
    )
    for t in res["trace"]:
        print(f"\n[Chunk Step {t['step']}] Active Chunk: {t['active_chunk']}")
        for n in t["retrieved_knn"]:
            print(f"  -> k-NN ({n['similarity']:.3f}) Continuation: {n['continuation']}")
    print(f"\nFull RETRO Generated Sequence:\n{res['full_output']}")

