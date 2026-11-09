"""
08] Iterative / Multi-Hop RAG
=============================
Pipeline:
Models multi-step retrieval as a stateful Markov Decision Process (MDP):
- At iteration t, the stateful retriever inspects the original question and the
  accumulated knowledge state S_t.
- Determines whether the question can be answered or what bridge fact is missing.
- Formulates a refined sub-query q_{t+1} to fetch the next hop of evidence.
- Repeats until the full chain of evidence is assembled.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class IterativeMultiHopRAG:
    def __init__(self, documents: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.doc_embeddings = [np.array(d.embedding) for d in resp.data]

    def retrieve_step(self, sub_query: str, top_k: int = 1) -> List[str]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[sub_query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.doc_embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [self.documents[i] for _, i in sims[:top_k]]

    def policy_step(self, original_query: str, accumulated_facts: List[str]) -> Dict:
        """Stateful MDP policy: decides whether we have enough facts or what next sub-query to issue."""
        facts_str = "\n".join([f"- {f}" for f in accumulated_facts]) if accumulated_facts else "None yet."
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an iterative multi-hop retrieval policy agent. Given the main question and "
                        "accumulated facts so far, decide if you have all facts needed to answer completely. "
                        'Return JSON: {"is_complete": bool, "next_search_query": str, "reasoning": str}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Main Question: {original_query}\n\nAccumulated Facts:\n{facts_str}",
                },
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    def answer(self, query: str, max_hops: int = 4) -> Dict:
        accumulated_facts: List[str] = []
        trace: List[Dict] = []

        for hop in range(1, max_hops + 1):
            decision = self.policy_step(query, accumulated_facts)
            if decision.get("is_complete") and accumulated_facts:
                trace.append({"hop": hop, "status": "Complete", "reasoning": decision["reasoning"]})
                break

            sub_query = decision.get("next_search_query", query)
            new_docs = self.retrieve_step(sub_query, top_k=1)
            for d in new_docs:
                if d not in accumulated_facts:
                    accumulated_facts.append(d)

            trace.append(
                {
                    "hop": hop,
                    "sub_query": sub_query,
                    "reasoning": decision.get("reasoning", ""),
                    "retrieved": new_docs,
                }
            )

        context_str = "\n".join([f"- {f}" for f in accumulated_facts])
        final_resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Answer the multi-hop question using the accumulated facts."},
                {"role": "user", "content": f"Accumulated Facts:\n{context_str}\n\nQuestion: {query}"},
            ],
            temperature=0.0,
        )
        return {
            "query": query,
            "hops": trace,
            "accumulated_facts": accumulated_facts,
            "final_answer": final_resp.choices[0].message.content.strip(),
        }


if __name__ == "__main__":
    corpus = [
        "Project Zephyr is led by the Chief Architect of CloudCore Systems.",
        "The Chief Architect of CloudCore Systems is Dr. Elena Rostova.",
        "Dr. Elena Rostova earned her PhD in Distributed Consensus from ETH Zurich in 2016.",
        "ETH Zurich is located in Zurich, Switzerland.",
    ]

    rag = IterativeMultiHopRAG(corpus)
    res = rag.answer("From which university and in what year did the leader of Project Zephyr earn their PhD?")
    for step in res["hops"]:
        print(f"Hop {step['hop']}: {step}")
    print(f"\nFinal Answer:\n{res['final_answer']}")

