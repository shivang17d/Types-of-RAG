# 06] HyDE (Hypothetical Document Embeddings) RAG

### Overview
**HyDE** bridges the semantic gap between short, vague user questions and dense technical documentation. It prompts the LLM to write hypothetical ideal documents first, averages their embeddings, and searches the real vector store in "document-to-document" embedding space.

### Architecture
```
User Query ──► OpenAI LLM ──► N Hypothetical Answers ──► Average Embedding (Centroid)
                                                                   │
                                                                   ▼
Real Answer ◄── OpenAI LLM ◄── Top-K Real Docs ◄── Vector Similarity Search
```

<!-- mermaid-06 -->
### Architecture Diagram

```mermaid
flowchart TD
    Q["Short / Vague Query"] --> GEN["LLM writes N=3 hypothetical docs, temperature 0.7"]
    GEN --> H1["Hypothetical Doc 1"]
    GEN --> H2["Hypothetical Doc 2"]
    GEN --> H3["Hypothetical Doc 3"]
    H1 & H2 & H3 --> EMB["Embed each hypothetical doc"]
    EMB --> CEN["Mean Centroid Vector"]
    CEN --> S["Cosine Search in document-to-document space"]
    VS[("Real Document Vectors")] --> S
    S --> TOP["Top-K Real Documents"]
    TOP --> G["Generator: answer ONLY from real docs"]
    Q --> G
    G --> A["Grounded Answer"]
```

### Run
```bash
python main.py
```

