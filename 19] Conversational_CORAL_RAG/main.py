"""
19] Conversational RAG (CORAL)
==============================
Pipeline:
Implements the CORAL multi-turn conversational RAG architecture:
1. Title / Entity Extraction & Conversation Flow Tracking.
2. Multi-Turn Contextualization (resolving topic shifts, ellipses, and coreferences).
3. Passage Retrieval from indexed articles with titles.
4. Citation-Labeled Response Generation (explicitly citing [Doc ID] inline).
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class ConversationalCORALRAG:
    def __init__(self, articles: List[Dict[str, str]], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        """Each article is {'id': 'Doc1', 'title': str, 'content': str}."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.articles = articles
        self.emb_model = emb_model
        self.chat_model = chat_model
        self.dialogue_turns: List[Dict[str, str]] = []

        inputs = [f"{a['title']}: {a['content']}" for a in articles]
        resp = self.client.embeddings.create(model=self.emb_model, input=inputs)
        self.embeddings = [np.array(d.embedding) for d in resp.data]

    def contextualize_and_extract_titles(self, user_utterance: str) -> Dict:
        history = "\n".join(
            [f"Turn {i+1} - User: {t['user']} | Assistant: {t['assistant']}" for i, t in enumerate(self.dialogue_turns)]
        )
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a CORAL Conversational Contextualizer. Given the multi-turn dialogue flow "
                        "and the latest user utterance, extract target entity titles and produce a de-contextualized "
                        'search query. Return JSON: {"extracted_titles": [str], "contextualized_query": str}'
                    ),
                },
                {"role": "user", "content": f"Dialogue History:\n{history}\n\nLatest Utterance: {user_utterance}"},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    def retrieve_passages(self, query: str, top_k: int = 2) -> List[Dict]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [self.articles[i] for _, i in sims[:top_k]]

    def respond_with_citations(self, user_utterance: str) -> Dict:
        ctx_info = self.contextualize_and_extract_titles(user_utterance)
        passages = self.retrieve_passages(ctx_info["contextualized_query"], top_k=2)

        formatted_passages = "\n".join(
            [f"[{p['id']}] Title: {p['title']} | Content: {p['content']}" for p in passages]
        )
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question in a natural conversational tone and label every factual "
                        "claim with its inline citation bracket, e.g., [Doc1], [Doc2]."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Retrieved Passages:\n{formatted_passages}\n\nUser Question: {user_utterance}",
                },
            ],
            temperature=0.0,
        )
        answer = resp.choices[0].message.content.strip()
        self.dialogue_turns.append({"user": user_utterance, "assistant": answer})
        return {
            "utterance": user_utterance,
            "extracted_titles": ctx_info.get("extracted_titles", []),
            "contextualized_query": ctx_info.get("contextualized_query", ""),
            "cited_answer": answer,
        }


if __name__ == "__main__":
    wiki_articles = [
        {
            "id": "Doc1",
            "title": "James Webb Space Telescope",
            "content": "Launched on December 25, 2021, JWST orbits the Sun-Earth L2 Lagrange point and observes in near- and mid-infrared wavelengths.",
        },
        {
            "id": "Doc2",
            "title": "JWST Primary Mirror",
            "content": "The JWST primary mirror consists of 18 hexagonal gold-coated beryllium segments with a combined diameter of 6.5 meters.",
        },
        {
            "id": "Doc3",
            "title": "Hubble Space Telescope",
            "content": "Hubble was launched in 1990 into low Earth orbit with a 2.4-meter glass mirror.",
        },
    ]

    coral = ConversationalCORALRAG(wiki_articles)
    for turn_q in [
        "When was the James Webb Space Telescope launched and where does it orbit?",
        "What is its primary mirror made of and how wide is it compared to Hubble's?",
    ]:
        out = coral.respond_with_citations(turn_q)
        print(f"\nUser: {out['utterance']}")
        print(f" -> Extracted Titles     : {out['extracted_titles']}")
        print(f" -> Contextualized Query : {out['contextualized_query']}")
        print(f" -> Cited Response       : {out['cited_answer']}")

