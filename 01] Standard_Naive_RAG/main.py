"""
01] Standard / Naive / Simple RAG
=================================
Pipeline:
1. Offline Indexing: Split documents into chunks -> Compute embeddings via OpenAI -> Store in Vector Index.
2. Online Retrieval: Embed user query -> Cosine similarity search -> Fetch Top-K chunks.
3. Generation: Pass [Retrieved Chunks + User Query] to OpenAI LLM for grounded answer generation.
"""

import os
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class StandardNaiveRAG:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.chunks: List[str] = []
        self.embeddings: List[List[float]] = []

    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Splits raw text into overlapping character-based chunks."""
        words = text.split()
        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates vector embeddings using the OpenAI Embeddings API."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def index_documents(self, documents: List[str], chunk_size: int = 60, overlap: int = 10):
        """Chunks and indexes documents into the in-memory vector store."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_text(doc, chunk_size=chunk_size, overlap=overlap))
        self.chunks = all_chunks
        self.embeddings = self.get_embeddings(self.chunks)
        print(f"[Indexed] {len(self.chunks)} chunks into the vector store.")

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict]:
        """Embeds the query and returns the top-k most similar chunks."""
        query_embedding = self.get_embeddings([query])[0]
        scored_chunks = []
        for idx, chunk_emb in enumerate(self.embeddings):
            score = self.cosine_similarity(query_embedding, chunk_emb)
            scored_chunks.append({"chunk": self.chunks[idx], "score": score})

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def generate(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generates an answer grounded strictly in the retrieved context."""
        context_str = "\n\n".join(
            [f"[Chunk {i+1} | Score: {d['score']:.4f}]\n{d['chunk']}" for i, d in enumerate(retrieved_docs)]
        )
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise assistant. Answer the user's question using ONLY "
                        "the provided context. If the answer is not in the context, say 'I don't know'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nQuestion: {query}",
                },
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def query(self, user_query: str, top_k: int = 2) -> Dict:
        retrieved = self.retrieve(user_query, top_k=top_k)
        answer = self.generate(user_query, retrieved)
        return {"query": user_query, "retrieved_context": retrieved, "answer": answer}


if __name__ == "__main__":
    sample_docs = [
        "AuroraDB v4.2 introduces Quantum-Safe Encryption using lattice-based cryptography. "
        "It reduces write latency to 1.4 milliseconds and supports up to 128TB per cluster.",
        "To configure automatic failover in AuroraDB, set ENABLE_AUTO_FAILOVER=true in cluster.yaml "
        "and assign at least two standby nodes in separate availability zones.",
        "The company refund policy allows full refunds within 30 days of purchase if usage is under 10 hours.",
    ]

    rag = StandardNaiveRAG()
    rag.index_documents(sample_docs)

    question = "What encryption does AuroraDB v4.2 use and what is its write latency?"
    result = rag.query(question, top_k=2)

    print(f"\nUser Query: {result['query']}")
    print("\nRetrieved Chunks:")
    for item in result["retrieved_context"]:
        print(f" - ({item['score']:.4f}) {item['chunk']}")
    print(f"\nGenerated Answer:\n{result['answer']}")

