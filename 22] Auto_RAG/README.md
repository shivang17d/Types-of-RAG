# 22] AutoRAG (Automated RAG Optimization)

### Overview
**AutoRAG** treats the RAG pipeline as a modular node graph (`Query Expansion Node` $\rightarrow$ `Retriever Node` $\rightarrow$ `Reranker Node`) and uses greedy optimization against a ground-truth evaluation dataset to automatically discover the best-performing module configuration.

<!-- mermaid-22 -->
### Architecture Diagram

**Search Space**

```mermaid
flowchart LR
    BM["Benchmark Question + Ground Truth"] --> N1
    subgraph N1["Node 1: Query Expansion"]
        A1["passthrough"]
        A2["llm_expand"]
    end
    subgraph N2["Node 2: Retrieval top_k"]
        B1["k = 1"]
        B2["k = 2"]
    end
    subgraph N3["Node 3: Reranker"]
        C1["none"]
        C2["llm_rerank"]
    end
    N1 -->|"greedy best"| N2
    N2 -->|"greedy best"| N3
    N3 --> OPT["Optimal Configuration"]
```

**Greedy Optimization Loop**

```mermaid
flowchart TD
    START["Start: passthrough, k=1, none"] --> NODE{"Next node in search space?"}
    NODE -->|Yes| OPTN["For each option of this node"]
    OPTN --> TRIAL["Trial config = best config + this option"]
    TRIAL --> RUN["Expand, Retrieve, Rerank, Generate"]
    RUN --> JUDGE["LLM Judge vs ground truth: score 0-1"]
    JUDGE --> KEEP["Keep option if score is the best so far"]
    KEEP --> NODE
    NODE -->|No| OUT["Optimal config + trial history"]
```

### Run
```bash
python main.py
```

