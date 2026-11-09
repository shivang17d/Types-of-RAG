"""
04] Modular RAG
===============
Pipeline:
Implements a decoupled, plug-and-play RAG framework using abstract interfaces.
Each module (QueryTransformer, Retriever, Reranker, Generator) can be swapped
independently without changing the pipeline orchestrator.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# --- Abstract Module Interfaces ---
class BaseQueryTransformer(ABC):
    @abstractmethod
    def transform(self, query: str) -> str:
        pass


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[str]:
        pass


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: List[str], top_n: int) -> List[str]:
        pass


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, context: List[str]) -> str:
        pass


# --- Concrete Pluggable Implementations ---
class PassthroughQueryTransformer(BaseQueryTransformer):
    def transform(self, query: str) -> str:
        return query


class LLMExpansionQueryTransformer(BaseQueryTransformer):
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def transform(self, query: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Expand the query with relevant technical synonyms in one concise sentence."},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()


class OpenAIVectorRetriever(BaseRetriever):
    def __init__(self, client: OpenAI, documents: List[str], emb_model: str = "text-embedding-3-small"):
        self.client = client
        self.documents = documents
        self.emb_model = emb_model
        resp = self.client.embeddings.create(model=self.emb_model, input=documents)
        self.embeddings = [np.array(d.embedding) for d in resp.data]

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        q_emb = np.array(
            self.client.embeddings.create(model=self.emb_model, input=[query]).data[0].embedding
        )
        scores = [
            float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10))
            for d_emb in self.embeddings
        ]
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in ranked_idx]


class IdentityReranker(BaseReranker):
    def rerank(self, query: str, docs: List[str], top_n: int) -> List[str]:
        return docs[:top_n]


class OpenAIGenerator(BaseGenerator):
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def generate(self, query: str, context: List[str]) -> str:
        joined = "\n".join([f"- {c}" for c in context])
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer using only the provided context."},
                {"role": "user", "content": f"Context:\n{joined}\n\nQuestion: {query}"},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()


# --- Modular Orchestrator ---
class ModularRAGPipeline:
    def __init__(
        self,
        query_transformer: BaseQueryTransformer,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        generator: BaseGenerator,
    ):
        self.query_transformer = query_transformer
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def execute(self, query: str, retrieve_k: int = 3, final_n: int = 2) -> Dict:
        transformed_q = self.query_transformer.transform(query)
        retrieved_docs = self.retriever.retrieve(transformed_q, top_k=retrieve_k)
        reranked_docs = self.reranker.rerank(query, retrieved_docs, top_n=final_n)
        answer = self.generator.generate(query, reranked_docs)
        return {
            "original_query": query,
            "transformed_query": transformed_q,
            "context_used": reranked_docs,
            "answer": answer,
        }


if __name__ == "__main__":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    corpus = [
        "Kubernetes Horizontal Pod Autoscaler (HPA) scales pods based on CPU or custom metrics.",
        "Vertical Pod Autoscaler (VPA) adjusts CPU and memory resource requests for containers.",
        "Karpenter is a node provisioning system that launches right-sized compute nodes in seconds.",
    ]

    # Swap modules effortlessly
    pipeline = ModularRAGPipeline(
        query_transformer=LLMExpansionQueryTransformer(client),
        retriever=OpenAIVectorRetriever(client, corpus),
        reranker=IdentityReranker(),
        generator=OpenAIGenerator(client),
    )

    result = pipeline.execute("How do I automatically adjust pod count vs container resource limits?")
    print(f"Transformed Query: {result['transformed_query']}")
    print(f"Answer:\n{result['answer']}")

