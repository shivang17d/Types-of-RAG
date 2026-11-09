# 01] Standard / Naive / Simple RAG

### Overview
**Standard (Naive) RAG** is the foundational single-step retrieval-augmented generation architecture. It converts documents into fixed-size chunks, embeds them into a vector space, retrieves the top-$k$ nearest neighbors for a user query, and passes them directly to the LLM.

### Architecture
```
Documents ──► Chunking ──► OpenAI Embeddings ──► Vector Store
                                                      │
User Query ──► Query Embedding ──► Cosine Search ◄────┘
                                        │
                                        ▼
                            Top-K Relevant Chunks ──► OpenAI LLM ──► Final Answer
```

<!-- mermaid-01 -->
### Architecture Diagram

```mermaid
flowchart TD
    subgraph OFF["Offline Indexing Phase"]
        DOC["Raw Documents"]
        CH["Chunker: 60-word windows, 10-word overlap"]
        E1["OpenAI Embeddings: text-embedding-3-small"]
        VS[("In-Memory Vector Store: chunks + vectors")]
    end
    subgraph ON["Online Query Phase"]
        Q["User Query"]
        E2["Embed Query"]
        S["Cosine Similarity vs every chunk"]
        K["Top-K Chunks + scores"]
        P["Prompt: grounding rule + context + question"]
        L["gpt-4o-mini, temperature 0"]
        A["Grounded Answer or I don't know"]
    end
    DOC --> CH --> E1 --> VS
    Q --> E2 --> S
    VS --> S
    S --> K --> P
    Q --> P
    P --> L --> A
```

### Run
```bash
python main.py
```

