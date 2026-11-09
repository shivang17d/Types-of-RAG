"""
14] ReAct (Reason + Act) RAG
============================
Pipeline:
Implements the explicit ReAct reasoning loop:
  Thought -> Action (Search[query] | Finish[answer]) -> Observation
Maintains situational awareness by updating the prompt context with past Thoughts,
Actions, and Observations at every step until the task is completed.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class ReActRAG:
    def __init__(self, documents: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.doc_embeddings = [np.array(d.embedding) for d in resp.data]

    def search_environment(self, query: str) -> str:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(self.doc_embeddings)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return self.documents[sims[0][1]]

    def run(self, question: str, max_steps: int = 5) -> Dict:
        trajectory: List[Dict] = []
        history_log = f"Question: {question}\n"

        for step in range(1, max_steps + 1):
            resp = self.client.chat.completions.create(
                model=self.chat_model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a ReAct (Reason + Act) agent. At each step, generate a logical 'thought', "
                            "then choose an 'action_type' ('SEARCH' or 'FINISH') and 'action_input'.\n"
                            'Return JSON: {"thought": str, "action_type": "SEARCH" | "FINISH", "action_input": str}'
                        ),
                    },
                    {"role": "user", "content": history_log},
                ],
                temperature=0.0,
            )
            parsed = json.loads(resp.choices[0].message.content)
            thought = parsed.get("thought", "")
            action_type = parsed.get("action_type", "FINISH")
            action_input = parsed.get("action_input", "")

            if action_type == "FINISH":
                trajectory.append({"step": step, "thought": thought, "action": f"Finish[{action_input}]"})
                return {"question": question, "trajectory": trajectory, "answer": action_input}

            observation = self.search_environment(action_input)
            trajectory.append(
                {
                    "step": step,
                    "thought": thought,
                    "action": f"Search[{action_input}]",
                    "observation": observation,
                }
            )
            history_log += (
                f"\nStep {step}:\nThought: {thought}\n"
                f"Action: Search[{action_input}]\nObservation: {observation}\n"
            )

        return {"question": question, "trajectory": trajectory, "answer": "Unable to finish within max steps."}


if __name__ == "__main__":
    kb = [
        "The flagship database engine developed by Chronos Labs is called ChronoStore.",
        "ChronoStore was written in Rust by lead systems engineer Kenji Takahashi.",
        "Kenji Takahashi previously worked at CERN on high-throughput particle telemetry systems.",
    ]

    react = ReActRAG(kb)
    res = react.run("Where did the creator of Chronos Labs' flagship database engine previously work?")
    for step in res["trajectory"]:
        print(f"\n--- Step {step['step']} ---")
        print(f"Thought    : {step['thought']}")
        print(f"Action     : {step['action']}")
        if "observation" in step:
            print(f"Observation: {step['observation']}")
    print(f"\nFinal Answer:\n{res['answer']}")

