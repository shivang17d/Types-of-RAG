# 24] EACO-RAG (Edge-Assisted Collaborative RAG)

### Overview
**EACO-RAG** distributes vector datasets across low-latency **Edge Nodes** and a central **Cloud Node**. It collaborates across edge nodes first, escalates to the cloud only when local confidence is low, and performs **Adaptive Knowledge Updates** to cache newly retrieved cloud knowledge at the edge.

<!-- mermaid-24 -->
### Architecture Diagram

```mermaid
flowchart TD
    Q["Query"] --> EMB["Embed Query"]
    EMB --> E1["Edge-DevOps node, 12 ms"]
    EMB --> E2["Edge-Security node, 15 ms"]
    E1 & E2 --> BEST{"Best edge score at least 0.45?"}
    BEST -->|Yes| SERVE["Serve from Edge"]
    BEST -->|No| CL["Cloud-Global node, 185 ms"]
    CL --> CMP{"Cloud score better?"}
    CMP -->|No| SERVE
    CMP -->|Yes| CACHE["Adaptive Knowledge Update: cache doc on Edge-DevOps"]
    CACHE --> SERVE2["Serve from Cloud"]
    SERVE & SERVE2 --> BANDIT["Bandit reward = score - latency penalty - cost"]
    BANDIT --> G["Generator LLM"]
    G --> A["Answer + serving node + latency"]
```

### Run
```bash
python main.py
```

