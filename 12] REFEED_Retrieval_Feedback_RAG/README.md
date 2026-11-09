# 12] REFEED (Retrieval Feedback) RAG

### Overview
**REFEED** refines model outputs using retrieval feedback without fine-tuning. It generates initial draft answers first, concatenates `[Query + Initial Drafts]` to retrieve grounding passages, and uses an ensemble ranking pass to correct errors in the initial drafts.

<!-- mermaid-12 -->
### Architecture Diagram

```mermaid
flowchart TD
    Q["Query"] --> DR["Initial Drafts from parametric memory: n=2, temp 0.6"]
    DR --> CMB["Concatenate Query + Drafts"]
    Q --> CMB
    CMB --> R["Feedback Retrieval top-2"]
    VS[("Vector Store")] --> R
    R --> EVAL["REFEED Evaluator: compare drafts vs passages, fix errors"]
    DR --> EVAL
    EVAL --> COR["corrections_made"]
    EVAL --> F["final_refined_answer"]
```

### Run
```bash
python main.py
```

