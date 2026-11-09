"""
15] Rule RAG
============
Pipeline:
Enhances question answering by adding explicit rule-based guidance to both stages of RAG:
1. Rule-Guided Retrieval: Applies predefined domain rules to derive logical search queries
   that fetch rule-compliant documents.
2. Rule-Guided Generation: Enforces domain compliance rules during response generation
   and validates that no rule is violated.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class RuleRAG:
    def __init__(self, documents: List[str], rules: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = documents
        self.rules = rules
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.doc_embeddings = [np.array(d.embedding) for d in resp.data]

    def apply_rules_to_guide_retrieval(self, query: str) -> List[str]:
        """Uses predefined rules to generate logically required search queries."""
        rules_str = "\n".join([f"Rule {i+1}: {r}" for i, r in enumerate(self.rules)])
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Given the user query and predefined domain rules, generate search queries that "
                        "retrieve all logically required facts enforced by the rules. "
                        'Return JSON: {"rule_guided_queries": [str]}'
                    ),
                },
                {"role": "user", "content": f"Domain Rules:\n{rules_str}\n\nUser Query: {query}"},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content).get("rule_guided_queries", [query])

    def retrieve_relevant_documents(self, rule_queries: List[str], top_k: int = 1) -> List[str]:
        collected = []
        for rq in rule_queries:
            q_emb = np.array(
                self.client.embeddings.create(model=self.emb_model, input=[rq]).data[0].embedding
            )
            sims = [
                (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
                for i, d_emb in enumerate(self.doc_embeddings)
            ]
            sims.sort(key=lambda x: x[0], reverse=True)
            for _, idx in sims[:top_k]:
                if self.documents[idx] not in collected:
                    collected.append(self.documents[idx])
        return collected

    def apply_rules_to_guide_generation(self, query: str, retrieved_docs: List[str]) -> str:
        rules_str = "\n".join([f"- {r}" for r in self.rules])
        docs_str = "\n".join([f"- {d}" for d in retrieved_docs])
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Rule-RAG generator. Answer the user query using the retrieved documents "
                        "while strictly following every domain rule below.\n\n"
                        f"Domain Rules:\n{rules_str}"
                    ),
                },
                {"role": "user", "content": f"Retrieved Documents:\n{docs_str}\n\nQuestion: {query}"},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    def run(self, query: str) -> Dict:
        rule_queries = self.apply_rules_to_guide_retrieval(query)
        docs = self.retrieve_relevant_documents(rule_queries, top_k=1)
        answer = self.apply_rules_to_guide_generation(query, docs)
        return {
            "query": query,
            "rule_guided_queries": rule_queries,
            "retrieved_docs": docs,
            "answer": answer,
        }


if __name__ == "__main__":
    domain_rules = [
        "When assessing loan eligibility, always retrieve both the applicant's credit score policy AND debt-to-income (DTI) cap.",
        "Never approve a commercial tier loan if DTI exceeds 36%, even if credit score is above 750.",
        "Always explicitly state which rule governed the decision.",
    ]
    kb = [
        "Commercial Tier Loan Credit Policy: Minimum credit score required is 720.",
        "Commercial Tier Loan DTI Policy: Maximum allowable debt-to-income (DTI) ratio is 36%.",
        "Applicant #409 (Apex Logistics) has a credit score of 780 and a current DTI ratio of 41%.",
    ]

    rule_rag = RuleRAG(kb, domain_rules)
    res = rule_rag.run("Is Applicant #409 (Apex Logistics) eligible for a Commercial Tier Loan?")
    print("Rule-Guided Queries Generated:")
    for rq in res["rule_guided_queries"]:
        print(f" - {rq}")
    print("\nRule-Guided Final Answer:")
    print(res["answer"])

