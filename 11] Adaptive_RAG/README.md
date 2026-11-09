# 11] Adaptive RAG

### Overview
**Adaptive RAG** dynamically selects the optimal retrieval depth (`NO_RETRIEVAL`, `SINGLE_HOP`, or `MULTI_HOP`) based on query complexity and an internal model honesty/confidence probe, avoiding unnecessary retrievals for simple queries while dedicating multi-hop budget to complex ones.

<!-- mermaid-11 -->
### Architecture Diagram

```mermaid
flowchart TD
    Q["User Query"] --> RTR{"Adaptive Router + Honesty Probe"}
    RTR -->|"NO_RETRIEVAL, confidence at least 0.95"| D["LLM Parametric Answer"]
    RTR -->|SINGLE_HOP| S["Single Retrieval top-2"]
    RTR -->|MULTI_HOP| M["Decompose into sub_queries"]
    M --> L["Per sub-query: retrieve top-1, dedupe"]
    S --> CTX["Retrieved Context"]
    L --> CTX
    CTX --> G["Generator LLM"]
    D --> A["Answer + route + confidence"]
    G --> A
```

### Run
```bash
python main.py
```

