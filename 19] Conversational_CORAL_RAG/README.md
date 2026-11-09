# 19] Conversational RAG (CORAL)

### Overview
**Conversational RAG (CORAL)** handles realistic multi-turn open-domain conversations through **Title Extraction**, **Conversation Flow Contextualization**, **Passage Retrieval**, and **Inline Citation Labeling** (`[Doc1]`, `[Doc2]`).

<!-- mermaid-19 -->
### Architecture Diagram

```mermaid
flowchart TD
    U["Latest Utterance"] --> CTX["CORAL Contextualizer LLM"]
    HIST[("Dialogue Turns")] --> CTX
    CTX --> TI["Extracted Titles"]
    CTX --> CQ["De-contextualized Query"]
    CQ --> RET["Retrieve top-2 titled articles"]
    IDX[("Index: title + content embeddings")] --> RET
    RET --> FMT["Format passages with Doc IDs"]
    FMT --> GEN["Citation Generator: label every claim with its Doc ID"]
    GEN --> ANS["Cited Conversational Answer"]
    ANS --> HIST
```

### Run
```bash
python main.py
```

