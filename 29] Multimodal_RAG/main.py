"""
29] Multimodal RAG
==================
Pipeline:
Indexes and retrieves across heterogeneous modalities (Text Documents, Visual Images/Charts
encoded via OpenAI Vision, and Structured Markdown Tables) in a unified embedding space:
1. Multimodal Indexing:
   - Images/Diagrams (base64 PNG/SVG data URLs) are captioned & semantically indexed using OpenAI Vision (gpt-4o-mini).
   - Tables and Text are indexed with modality tags.
2. Unified Cross-Modal Retrieval:
   - Searches across all modalities simultaneously.
3. Multimodal Generation:
   - Passes retrieved text, tables, AND raw image payloads to OpenAI Vision to synthesize the final answer.
"""

import os
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# A tiny 1x1 blue pixel PNG encoded in base64 to demonstrate real OpenAI Vision API multimodal input
SAMPLE_CHART_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkWPjfDwAEgQH45628yAAAAABJRU5ErkJggg=="
)


class MultimodalRAG:
    def __init__(self, emb_model: str = "text-embedding-3-small", vision_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.emb_model = emb_model
        self.vision_model = vision_model
        self.items: List[Dict] = []
        self.embeddings: List[np.ndarray] = []

    def index_multimodal_items(self, raw_items: List[Dict]):
        """
        raw_items can be:
        - {'modality': 'text', 'content': str}
        - {'modality': 'table', 'content': str}
        - {'modality': 'image', 'image_b64': str, 'metadata_hint': str}
        """
        searchable_texts = []
        for item in raw_items:
            if item["modality"] == "image":
                # Use OpenAI Vision to convert the image + chart metadata into a rich semantic description
                resp = self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Summarize this visual chart/image for search indexing. Context hint: {item.get('metadata_hint', '')}",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{item['image_b64']}"},
                                },
                            ],
                        }
                    ],
                    max_tokens=120,
                )
                caption = resp.choices[0].message.content.strip()
                enriched = {**item, "searchable_summary": f"[IMAGE/CHART]: {caption}"}
            else:
                enriched = {**item, "searchable_summary": f"[{item['modality'].upper()}]: {item['content']}"}

            self.items.append(enriched)
            searchable_texts.append(enriched["searchable_summary"])

        emb_resp = self.client.embeddings.create(model=self.emb_model, input=searchable_texts)
        self.embeddings = [np.array(d.embedding) for d in emb_resp.data]

    def query(self, user_query: str, top_k: int = 3) -> Dict:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[user_query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        retrieved = [self.items[i] for _, i in sims[:top_k]]

        # Build multimodal prompt containing both text/tables and any retrieved image
        content_blocks = [
            {
                "type": "text",
                "text": (
                    "Answer the user question by combining insights from the retrieved text, "
                    "structured tables, and visual chart items below.\n\n"
                    f"Question: {user_query}\n\nRetrieved Multimodal Context:\n"
                    + "\n".join([r["searchable_summary"] for r in retrieved])
                ),
            }
        ]
        for r in retrieved:
            if r["modality"] == "image":
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{r['image_b64']}"},
                    }
                )

        resp = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[{"role": "user", "content": content_blocks}],
            temperature=0.0,
        )
        return {
            "retrieved_modalities": [r["modality"] for r in retrieved],
            "retrieved_summaries": [r["searchable_summary"] for r in retrieved],
            "answer": resp.choices[0].message.content.strip(),
        }


if __name__ == "__main__":
    multimodal_kb = [
        {
            "modality": "text",
            "content": "Q4 Server Fleet Report: Cold-plate liquid cooling was deployed across Datacenter-East in October.",
        },
        {
            "modality": "table",
            "content": "| Datacenter | PUE Before | PUE After | Annual Savings |\n| East | 1.48 | 1.12 | $2.4M |",
        },
        {
            "modality": "image",
            "image_b64": SAMPLE_CHART_BASE64,
            "metadata_hint": "Thermal telemetry chart showing GPU junction temperatures dropping from 84C to 59C under full load in Datacenter-East.",
        },
    ]

    mm_rag = MultimodalRAG()
    mm_rag.index_multimodal_items(multimodal_kb)
    res = mm_rag.query("How did liquid cooling impact PUE, cost savings, and GPU temperatures in Datacenter-East?")
    print("Retrieved Modalities:", res["retrieved_modalities"])
    for s in res["retrieved_summaries"]:
        print(f" - {s}")
    print(f"\nMultimodal Answer:\n{res['answer']}")

