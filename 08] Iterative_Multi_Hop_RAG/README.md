# 08] Iterative / Multi-Hop RAG

### Overview
**Iterative RAG** handles multi-hop questions where the entity needed for the second search is only discovered after reading the result of the first search. It maintains an internal knowledge state and iteratively updates its search query across multiple hops.

<!-- mermaid-08 -->
### Architecture Diagram

**Architecture**

```mermaid
flowchart TD
    Q["Multi-hop Question"] --> POL{"Policy LLM: is evidence complete?"}
    POL -->|"No: emit next_search_query"| RET["Retrieve top-1 for sub-query"]
    RET --> ACC[("Accumulated Facts - state S_t")]
    ACC -->|"hop t+1"| POL
    POL -->|"Yes, or max_hops reached"| GEN["Generator: answer from accumulated facts"]
    ACC --> GEN
    GEN --> A["Final Answer"]
```

**Example Hop Sequence**

```mermaid
sequenceDiagram
    participant U as User
    participant P as Policy LLM
    participant V as Vector Store
    participant G as Generator
    U->>P: Where did the leader of Project Zephyr earn a PhD?
    P->>V: Hop 1 - Who leads Project Zephyr?
    V-->>P: Chief Architect of CloudCore Systems
    P->>V: Hop 2 - Who is the CloudCore Chief Architect?
    V-->>P: Dr. Elena Rostova
    P->>V: Hop 3 - Elena Rostova PhD
    V-->>P: ETH Zurich, 2016
    P->>G: is_complete = true, all facts
    G-->>U: ETH Zurich in 2016
```

### Run
```bash
python main.py
```

