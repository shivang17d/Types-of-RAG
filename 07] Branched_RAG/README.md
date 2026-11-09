# 07] Branched RAG

### Overview
**Branched RAG** explores multiple lines of thought simultaneously by splitting an open-ended query into distinct perspectives (e.g., Technical, Financial, Compliance), retrieving evidence for each branch in parallel, and synthesizing the multi-perspective results.

<!-- mermaid-07 -->
### Architecture Diagram

```mermaid
flowchart TD
    Q["Open-ended Query"] --> BG["Branch Generator LLM: JSON perspectives + sub-queries"]
    BG --> B1["Branch 1: Technical"]
    BG --> B2["Branch 2: Financial"]
    BG --> B3["Branch 3: Risk / Compliance"]
    B1 --> R1["Retrieve top-2"] --> A1["Branch Analysis LLM"]
    B2 --> R2["Retrieve top-2"] --> A2["Branch Analysis LLM"]
    B3 --> R3["Retrieve top-2"] --> A3["Branch Analysis LLM"]
    VS[("Vector Store")] --> R1 & R2 & R3
    A1 & A2 & A3 --> SYN["Synthesizer LLM: compare + merge perspectives"]
    SYN --> F["Balanced Final Answer"]
```

### Run
```bash
python main.py
```

