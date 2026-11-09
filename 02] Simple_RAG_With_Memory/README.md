# 02] Simple RAG with Memory

### Overview
**Simple RAG with Memory** extends standard RAG by storing conversation history and using an LLM query-rewriter to resolve pronouns and implicit context (`"its population"` $\rightarrow$ `"Paris population"`) prior to searching the vector store.

### Architecture
```
User Follow-up Query + Chat Memory ──► Query Contextualizer (LLM)
                                               │
                                    Standalone Search Query
                                               │
                                               ▼
                                      Vector DB Retrieval
                                               │
                                               ▼
                       Context + Chat History ──► OpenAI LLM ──► Response (Saved to Memory)
```

<!-- mermaid-02 -->
### Architecture Diagram

```mermaid
flowchart TD
    U["User Follow-up: What is its population?"] --> H{"Chat history empty?"}
    H -->|Yes| RAW["Use query as-is"]
    H -->|No| CQ["Query Contextualizer LLM: resolve pronouns from last 4 turns"]
    MEM[("Conversation Memory Buffer")] --> CQ
    CQ --> SQ["Standalone Query: population of Paris"]
    RAW --> R["Embed + Cosine Search"]
    SQ --> R
    VS[("Vector Store")] --> R
    R --> TOP["Top-K Documents"]
    TOP --> G["Generator LLM: history + context + question"]
    MEM --> G
    G --> ANS["Answer"]
    ANS --> W["Append user + assistant turn"]
    W --> MEM
```

### Run
```bash
python main.py
```

