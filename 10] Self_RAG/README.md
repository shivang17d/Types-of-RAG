# 10] Self-RAG (Self-Reflective RAG)

### Overview
**Self-RAG** critiques its own retrieval and generation using four reflection decisions:
- `[Retrieve]`: Checks if retrieval is needed.
- `[IsREL]`: Checks if each passage is relevant.
- `[IsSUP]`: Verifies that the generated answer is grounded without hallucinations.
- `[IsUSE]`: Scores overall utility (1–5) to pick the best candidate.

<!-- mermaid-10 -->
### Architecture Diagram

```mermaid
flowchart TD
    Q["User Query"] --> RT{"Retrieve token: are external facts needed?"}
    RT -->|No| DIR["Direct LLM Answer"]
    RT -->|Yes| RET["Retrieve top-3 passages"]
    RET --> P1["Passage 1"] & P2["Passage 2"] & P3["Passage 3"]
    P1 --> C1["Generate + Critique: IsREL, IsSUP, IsUSE"]
    P2 --> C2["Generate + Critique: IsREL, IsSUP, IsUSE"]
    P3 --> C3["Generate + Critique: IsREL, IsSUP, IsUSE"]
    C1 & C2 & C3 --> F{"Keep IsREL = Relevant"}
    F --> RANK["Rank by IsSUP: Fully, Partially, No Support - then IsUSE 1-5"]
    RANK --> BEST["Best Grounded Candidate"]
    BEST --> A["Final Answer"]
    DIR --> A
```

### Run
```bash
python main.py
```

