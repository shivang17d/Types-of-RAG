# 09] Corrective RAG (CRAG)

### Overview
**Corrective RAG (CRAG)** introduces an explicit **Retrieval Evaluator** between retrieval and generation. It grades local retrieved documents as `CORRECT`, `AMBIGUOUS`, or `INCORRECT`. When local documents fail or are incomplete, it rewrites the query and falls back to external web search.

<!-- mermaid-09 -->
### Architecture Diagram

```mermaid
flowchart TD
    Q["User Query"] --> LR["Local Vector Retrieval top-2"]
    LR --> EV{"Retrieval Evaluator LLM: grade + confidence"}
    EV -->|CORRECT| KR["Knowledge Refinement: refined facts"]
    EV -->|AMBIGUOUS| KR
    EV -->|AMBIGUOUS| RW["Rewrite query for web"]
    EV -->|INCORRECT| DIS["Discard local docs"]
    DIS --> RW
    RW --> WS["Web Search fallback"]
    KR --> EVD["Verified Evidence"]
    WS --> EVD
    EVD --> G["Generator LLM"]
    G --> A["Corrected Answer"]
```

### Run
```bash
python main.py
```

