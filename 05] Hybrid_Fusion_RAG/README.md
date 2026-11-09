# 05] Hybrid / Fusion RAG

### Overview
**Hybrid / Fusion RAG** combines **Sparse Lexical Retrieval (BM25)** (ideal for exact product codes, SKUs, and error strings) with **Dense Vector Search (OpenAI Embeddings)** (ideal for conceptual similarity) using **Reciprocal Rank Fusion (RRF)**:

$$RRF(d) = \sum_{m \in \{\text{BM25}, \text{Vector}\}} \frac{1}{k + \text{rank}_m(d)}$$

<!-- mermaid-05 -->
### Architecture Diagram

```mermaid
flowchart TD
    Q["User Query: fix ERR-9042-X SSL timeout"]
    subgraph SP["Sparse Path - lexical"]
        T["Tokenizer: words + hyphenated codes"]
        BM["BM25: IDF x saturated TF, k1=1.5, b=0.75"]
        R1["Sparse Ranked List"]
    end
    subgraph DP["Dense Path - semantic"]
        E["OpenAI Query Embedding"]
        CS["Cosine Similarity"]
        R2["Dense Ranked List"]
    end
    Q --> T --> BM --> R1
    Q --> E --> CS --> R2
    R1 --> RRF["Reciprocal Rank Fusion: sum of 1 / (60 + rank)"]
    R2 --> RRF
    RRF --> TOP["Top-K Fused Documents"]
    TOP --> G["gpt-4o-mini Generator"]
    G --> A["Answer"]
```

### Run
```bash
python main.py
```

