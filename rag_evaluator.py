"""
RAG Triad Evaluator (OpenAI LLM-as-a-Judge)
===========================================
Evaluates any RAG pipeline output across the three core pillars of the RAG Triad:
1. Context Relevance  : Are the retrieved chunks focused and relevant to the user query?
2. Faithfulness       : Is the generated answer strictly grounded in the retrieved context without hallucination?
3. Answer Relevance   : Does the generated answer directly and completely address the user's question?
"""

import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class RAGTriadEvaluator:
    def __init__(self, judge_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.judge_model = judge_model

    def evaluate(self, query: str, retrieved_context: List[str], generated_answer: str) -> Dict:
        context_block = "\n".join([f"- {c}" for c in retrieved_context])
        resp = self.client.chat.completions.create(
            model=self.judge_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an impartial RAG Triad Judge. Score the RAG execution on three metrics from 0.0 to 1.0:\n"
                        "1. context_relevance: How relevant and noise-free is the retrieved context for the query?\n"
                        "2. faithfulness: Is every claim in the generated answer supported by the retrieved context?\n"
                        "3. answer_relevance: How well does the answer address the user's original query?\n"
                        'Return JSON: {"context_relevance": float, "faithfulness": float, '
                        '"answer_relevance": float, "critique": str}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User Query:\n{query}\n\n"
                        f"Retrieved Context:\n{context_block}\n\n"
                        f"Generated Answer:\n{generated_answer}"
                    ),
                },
            ],
            temperature=0.0,
        )
        scores = json.loads(resp.choices[0].message.content)
        triad_mean = (
            float(scores.get("context_relevance", 0.0))
            + float(scores.get("faithfulness", 0.0))
            + float(scores.get("answer_relevance", 0.0))
        ) / 3.0
        scores["overall_triad_score"] = round(triad_mean, 4)
        return scores


if __name__ == "__main__":
    evaluator = RAGTriadEvaluator()
    sample_eval = evaluator.evaluate(
        query="What encryption does AuroraDB v4.2 use and what is its write latency?",
        retrieved_context=[
            "AuroraDB v4.2 introduces Quantum-Safe Encryption using lattice-based cryptography. It reduces write latency to 1.4 milliseconds."
        ],
        generated_answer="AuroraDB v4.2 uses Quantum-Safe Encryption based on lattice-based cryptography and achieves a write latency of 1.4 milliseconds.",
    )
    print("RAG Triad Evaluation Result:")
    print(json.dumps(sample_eval, indent=2))

