# 03] Advanced RAG

### Overview
**Advanced RAG** addresses the precision and noise problems of Naive RAG by adding structured **Pre-Retrieval** (Query Rewriting, Metadata Filtering) and **Post-Retrieval** (Reranking, Context Compression, and Lost-in-the-Middle Reordering) stages.

### Architecture
```
[Pre-Retrieval]   User Query ──► Query Rewriter + Metadata Filter
                                        │
[Retrieval]                             ▼
                              Filtered Vector Search
                                        │
[Post-Retrieval]                        ▼
                    Reranker ──► Context Compressor ──► Lost-in-the-Middle Reorder
                                                                │
[Generation]                                                    ▼
                                                          OpenAI Generator
```

<!-- mermaid-03 -->
### Architecture Diagram

```mermaid
flowchart TD
    Q["User Query"]
    subgraph PRE["1. Pre-Retrieval"]
        RW["LLM Query Rewriter: keyword-rich query"]
        MF["Metadata Filter: e.g. category = infrastructure"]
    end
    subgraph RET["2. Retrieval"]
        VS[("Vector Store with metadata")]
        FS["Filtered Cosine Search, top_k = 4"]
    end
    subgraph POST["3. Post-Retrieval"]
        RR["LLM Reranker: relevance score 0-10, JSON"]
        CC["Context Compressor: keep only key sentences"]
        LM["Lost-in-the-Middle Reorder: best chunks at the edges"]
    end
    subgraph GEN["4. Generation"]
        G["gpt-4o-mini Generator"]
        A["Final Answer"]
    end
    Q --> RW --> FS
    Q --> MF --> FS
    VS --> FS
    FS --> RR --> CC --> LM --> G --> A
    Q --> G
```

### Run
```bash
python main.py
```

