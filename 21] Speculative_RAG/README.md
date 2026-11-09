# 21] Speculative RAG

### Overview
**Speculative RAG** partitions retrieved documents into smaller subsets and uses a lightweight **Specialist Drafter Model** (`gpt-4o-mini`) to generate multiple candidate drafts in parallel threads. A larger **Generalist Verifier Model** (`gpt-4o`) evaluates the drafts and rationales to select the most accurate final output.

<!-- mermaid-21 -->
### Architecture Diagram

```mermaid
flowchart TD
    Q["Query"] --> RET["Retrieve top-4 docs"]
    RET --> PART["Partition into subsets of 2"]
    PART --> S1["Subset 1"] & S2["Subset 2"]
    subgraph PAR["Parallel Specialist Drafting - ThreadPoolExecutor"]
        D1["Drafter gpt-4o-mini: draft + rationale"]
        D2["Drafter gpt-4o-mini: draft + rationale"]
    end
    S1 --> D1
    S2 --> D2
    D1 & D2 --> VER["Generalist Verifier gpt-4o: grounding, completeness, rationale"]
    VER --> SEL["selected_draft_id + verification_reason"]
    VER --> FIN["Final Verified Result"]
```

### Run
```bash
python main.py
```

