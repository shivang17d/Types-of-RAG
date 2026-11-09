"""
13] Agentic RAG
===============
Pipeline:
Uses an OpenAI Function-Calling / Tool-Use Agent with a Task Planner to dynamically
route across multiple specialized knowledge bases (Technical Docs Vector Store,
Financial Metrics DB, Regulatory Compliance Store) and synthesize multi-source findings.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class AgenticRAG:
    def __init__(self, emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.emb_model = emb_model
        self.chat_model = chat_model

        # Specialized Knowledge Bases
        self.tech_kb = [
            "QuantumVault uses AES-256-GCM combined with ML-KEM-1024 post-quantum key encapsulation.",
            "QuantumVault cluster replication throughput is 18 GB/sec across 3 availability zones.",
        ]
        self.finance_kb = [
            "QuantumVault Enterprise license costs $48,000/year per cluster with a 99.999% uptime SLA.",
            "Q3 revenue from QuantumVault grew 64% YoY to $112M.",
        ]
        self.compliance_kb = [
            "QuantumVault is certified for SOC2 Type II, HIPAA, and FIPS 140-3 Level 3 compliance.",
        ]

        self.stores = {}
        for name, docs in [("tech", self.tech_kb), ("finance", self.finance_kb), ("compliance", self.compliance_kb)]:
            resp = self.client.embeddings.create(model=self.emb_model, input=docs)
            self.stores[name] = {
                "docs": docs,
                "embeddings": [np.array(d.embedding) for d in resp.data],
            }

    def search_store(self, store_name: str, query: str) -> str:
        store = self.stores.get(store_name)
        if not store:
            return "Store not found."
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[query]).data[0].embedding
        )
        sims = [
            (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
            for i, d_emb in enumerate(store["embeddings"])
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        return store["docs"][sims[0][1]]

    def run(self, user_query: str) -> Dict:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Searches a specialized knowledge base ('tech', 'finance', or 'compliance').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "store_name": {
                                "type": "string",
                                "enum": ["tech", "finance", "compliance"],
                            },
                            "query": {"type": "string"},
                        },
                        "required": ["store_name", "query"],
                    },
                },
            }
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an Agentic RAG Planner. Break down complex questions, invoke the appropriate "
                    "knowledge base tools ('tech', 'finance', 'compliance'), and synthesize a complete answer."
                ),
            },
            {"role": "user", "content": user_query},
        ]

        tool_calls_log = []
        for _ in range(4):
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
            )
            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                return {"query": user_query, "tool_calls": tool_calls_log, "answer": msg.content.strip()}

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = self.search_store(args["store_name"], args["query"])
                tool_calls_log.append({"store": args["store_name"], "query": args["query"], "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        return {"query": user_query, "tool_calls": tool_calls_log, "answer": "Max iterations reached."}


if __name__ == "__main__":
    agent = AgenticRAG()
    res = agent.run(
        "What cryptography does QuantumVault use, how much does the Enterprise license cost, and is it FIPS compliant?"
    )
    print("Agent Tool Calls:")
    for tc in res["tool_calls"]:
        print(f" -> [{tc['store'].upper()}] Query='{tc['query']}' | Found: {tc['result']}")
    print(f"\nFinal Agentic Answer:\n{res['answer']}")

