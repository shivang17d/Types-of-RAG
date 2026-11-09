"""
30] REVEAL (Retrieval-Augmented Visual-Language Model)
======================================================
Pipeline:
Implements the REVEAL architecture for Visual-Language reasoning:
1. Query Encoding (Vision Transformer + Text Encoder via OpenAI Vision):
   Encodes visual input + task query into a unified visual-semantic query representation.
2. Memory Encoding & Multi-Source Knowledge Retrieval:
   Retrieves real-world factual entries from an external structured knowledge base.
3. Knowledge Fusion with Gated Attention:
   Fuses visual observations with retrieved external facts and produces transparent,
   step-by-step grounded reasoning.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SAMPLE_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkWPjfDwAEgQH45628yAAAAABJRU5ErkJggg=="
)


class REVEALVisualRAG:
    def __init__(self, knowledge_base: List[str], emb_model: str = "text-embedding-3-small", vlm_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.knowledge_base = knowledge_base
        self.emb_model = emb_model
        self.vlm_model = vlm_model

        resp = self.client.embeddings.create(model=self.emb_model, input=knowledge_base)
        self.kb_embeddings = [np.array(d.embedding) for d in resp.data]

    def encode_visual_query(self, image_b64: str, visual_scene_description: str, user_question: str) -> str:
        """Encodes visual input + text query into a unified query representation."""
        resp = self.client.chat.completions.create(
            model=self.vlm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Visual Scene Telemetry: {visual_scene_description}\n"
                                f"Task Question: {user_question}\n"
                                "Formulate a dense factual search query combining the visual features and the question."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip()

    def retrieve_memory(self, encoded_query: str, top_k: int = 2) -> List[str]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[encoded_query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.kb_embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [self.knowledge_base[i] for _, i in sims[:top_k]]

    def knowledge_fusion_and_generate(self, visual_scene: str, question: str, retrieved_facts: List[str]) -> Dict:
        facts_str = "\n".join([f"- {f}" for f in retrieved_facts])
        resp = self.client.chat.completions.create(
            model=self.vlm_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the REVEAL Knowledge Fusion Generator. Fuse the visual observation with "
                        "the retrieved external knowledge to provide human-like step-by-step reasoning and "
                        'a grounded output. Return JSON: {"reasoning_steps": [str], "final_output": str}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Visual Observation: {visual_scene}\n"
                        f"Retrieved Knowledge Base Facts:\n{facts_str}\n"
                        f"Question: {question}"
                    ),
                },
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    def run(self, image_b64: str, visual_scene: str, question: str) -> Dict:
        encoded_query = self.encode_visual_query(image_b64, visual_scene, question)
        retrieved_facts = self.retrieve_memory(encoded_query, top_k=2)
        fused = self.knowledge_fusion_and_generate(visual_scene, question, retrieved_facts)
        return {
            "encoded_visual_query": encoded_query,
            "retrieved_facts": retrieved_facts,
            "reasoning_steps": fused.get("reasoning_steps", []),
            "final_output": fused.get("final_output", ""),
        }


if __name__ == "__main__":
    external_kb = [
        "Turbine Blade Anomaly Type-C (blue thermal halo at leading edge root) indicates cooling channel micro-blockage.",
        "Maintenance Protocol MP-44 requires ultrasonic flushing within 48 operating hours when Type-C thermal halos appear.",
        "Landing gear hydraulic pressure nominal range is 2,950 to 3,050 PSI.",
    ]

    reveal = REVEALVisualRAG(external_kb)
    res = reveal.run(
        image_b64=SAMPLE_IMAGE_B64,
        visual_scene="Infrared borescope image showing a distinct blue thermal halo at the leading edge root of high-pressure turbine blade #7.",
        question="What anomaly is shown in this inspection image and what maintenance action is required?",
    )
    print(f"Encoded Visual Query: {res['encoded_visual_query']}")
    print("\nStep-by-Step Grounded Reasoning:")
    for idx, step in enumerate(res["reasoning_steps"], 1):
        print(f"  {idx}. {step}")
    print(f"\nREVEAL Output:\n{res['final_output']}")

