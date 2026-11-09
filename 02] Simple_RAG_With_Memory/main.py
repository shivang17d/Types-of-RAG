"""
02] Simple RAG with Memory
==========================
Pipeline:
1. Maintains a rolling conversation history of (user_query, assistant_answer) turns.
2. Uses OpenAI to rewrite/contextualize follow-up queries that contain pronouns or implicit references
   (e.g., "What is its population?" -> "What is the population of Paris?").
3. Searches the vector store using the standalone contextualized query.
4. Generates a response incorporating both retrieved documents and conversation memory.
"""

import os
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class SimpleRAGWithMemory:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.docs: List[str] = []
        self.embeddings: List[List[float]] = []
        self.chat_history: List[Dict[str, str]] = []

    def index_documents(self, documents: List[str]):
        self.docs = documents
        resp = self.client.embeddings.create(model=self.embedding_model, input=documents)
        self.embeddings = [item.embedding for item in resp.data]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        va, vb = np.array(a), np.array(b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))

    def contextualize_query(self, latest_query: str) -> str:
        """Rewrites a follow-up query into a standalone search query using chat history."""
        if not self.chat_history:
            return latest_query

        history_str = "\n".join(
            [f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in self.chat_history[-4:]]
        )
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Given the conversation history and a follow-up question, rewrite the follow-up "
                        "question into a standalone search query that resolves all pronouns (it, they, its, there). "
                        "Return ONLY the standalone query text."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Chat History:\n{history_str}\n\nFollow-up Question: {latest_query}",
                },
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def retrieve(self, standalone_query: str, top_k: int = 2) -> List[str]:
        q_emb = self.client.embeddings.create(
            model=self.embedding_model, input=[standalone_query]
        ).data[0].embedding

        scored = [
            (self.cosine_similarity(q_emb, doc_emb), self.docs[i])
            for i, doc_emb in enumerate(self.embeddings)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def chat(self, user_query: str, top_k: int = 2) -> Dict:
        standalone_query = self.contextualize_query(user_query)
        retrieved_docs = self.retrieve(standalone_query, top_k=top_k)
        context_block = "\n".join([f"- {d}" for d in retrieved_docs])

        messages = [
            {
                "role": "system",
                "content": "You are a helpful conversational assistant. Use the retrieved context and conversation history to answer accurately.",
            }
        ]
        for turn in self.chat_history[-4:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        messages.append(
            {
                "role": "user",
                "content": f"Retrieved Context:\n{context_block}\n\nQuestion: {user_query}",
            }
        )

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        self.chat_history.append({"user": user_query, "assistant": answer})

        return {
            "original_query": user_query,
            "standalone_query": standalone_query,
            "retrieved_docs": retrieved_docs,
            "answer": answer,
        }


if __name__ == "__main__":
    knowledge_base = [
        "Paris is the capital and most populous city of France.",
        "Paris has an estimated city-proper population of 2.1 million residents, and 11.2 million in its metropolitan area.",
        "The Eiffel Tower in Paris was constructed in 1889 by Gustave Eiffel.",
        "Berlin is the capital of Germany with a population of 3.7 million.",
    ]

    rag = SimpleRAGWithMemory()
    rag.index_documents(knowledge_base)

    turns = [
        "What is the capital of France?",
        "What is its population?",
        "Who built the famous tower there and when?",
    ]

    for q in turns:
        res = rag.chat(q)
        print(f"\nUser: {res['original_query']}")
        print(f" -> Rewritten Standalone Query: {res['standalone_query']}")
        print(f" -> Assistant: {res['answer']}")

