# 20] Memo RAG

### Overview
**Memo RAG** pairs a **Lightweight Memory Model** (`gpt-4o-mini`) that maintains a global memory index and generates retrieval "clues" for vague/ambiguous queries with a **Heavy Generator Model** (`gpt-4o`) that synthesizes the final response from clue-retrieved documents.

<!-- mermaid-20 -->
### Architecture Diagram

```mermaid
flowchart TD
    subgraph INIT["Initialization"]
        CORP["Corpus"]
        GM["Light LLM builds Global Memory Index"]
        EMB[("Vector Store")]
    end
    CORP --> GM
    CORP --> EMB
    Q["Ambiguous Query"] --> CLUE["Light Memory Model gpt-4o-mini: draft + 2 clues"]
    GM --> CLUE
    CLUE --> DR["Memory Draft"]
    CLUE --> C1["Clue 1"] & C2["Clue 2"]
    C1 & C2 --> RET["Clue-based Retrieval top-1 each"]
    EMB --> RET
    RET --> HEAVY["Heavy Generator gpt-4o"]
    Q --> HEAVY
    HEAVY --> A["Final Answer"]
```

### Run
```bash
python main.py
```

