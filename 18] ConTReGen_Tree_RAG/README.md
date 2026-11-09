# 18] ConTReGen (Context-Driven Tree-Structured Retrieval)

### Overview
**ConTReGen** solves information gaps in long-form generation via a two-stage tree workflow:
1. **Top-Down Planning & Retrieval**: Decomposes a complex query into hierarchical facets and leaf sub-queries.
2. **Bottom-Up Synthesis**: Summarizes verified passages at each leaf node and integrates them upwards from leaves to root.

<!-- mermaid-18 -->
### Architecture Diagram

```mermaid
flowchart TD
    ROOT["Root Query"] --> DEC["Stage 1 - Top-Down Decomposer LLM"]
    DEC --> F1["Facet 1"]
    DEC --> F2["Facet 2"]
    F1 --> L11["Leaf Query 1.1"] & L12["Leaf Query 1.2"]
    F2 --> L21["Leaf Query 2.1"] & L22["Leaf Query 2.2"]
    L11 --> S11["Retrieve + Verify + Summarize"]
    L12 --> S12["Retrieve + Verify + Summarize"]
    L21 --> S21["Retrieve + Verify + Summarize"]
    L22 --> S22["Retrieve + Verify + Summarize"]
    S11 & S12 --> FS1["Stage 2 - Facet 1 Synthesis"]
    S21 & S22 --> FS2["Stage 2 - Facet 2 Synthesis"]
    FS1 & FS2 --> RS["Root Integration LLM"]
    RS --> REP["Long-form Executive Report"]
```

### Run
```bash
python main.py
```

