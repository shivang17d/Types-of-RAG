# 15] Rule RAG

### Overview
**Rule RAG** injects explicit deterministic rules into both stages of RAG:
1. **Apply Rules to Guide Retrieval**: Uses rules to identify all mandatory compliance facts to search for.
2. **Apply Rules to Guide Generation**: Constrains the LLM generator to obey logical policies.

<!-- mermaid-15 -->
### Architecture Diagram

```mermaid
flowchart TD
    Q["User Query"] --> RG["Rule-Guided Query Generator LLM"]
    RULES[("Domain Rule Base")] --> RG
    RG --> Q1["Rule Query 1: credit score policy"]
    RG --> Q2["Rule Query 2: DTI cap"]
    RG --> Q3["Rule Query 3: applicant facts"]
    Q1 & Q2 & Q3 --> RET["Retrieve top-1 each, dedupe"]
    VS[("Vector Store")] --> RET
    RET --> GEN["Rule-Constrained Generator: rules in system prompt"]
    RULES --> GEN
    GEN --> A["Compliant Decision citing the governing rule"]
```

### Run
```bash
python main.py
```

