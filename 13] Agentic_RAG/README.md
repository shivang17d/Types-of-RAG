# 13] Agentic RAG

### Overview
**Agentic RAG** uses an autonomous task-planning LLM agent with tool-calling capabilities to dynamically route sub-queries across specialized indices (`tech`, `finance`, `compliance`), inspect retrieved results in real time, and synthesize multi-source answers.

<!-- mermaid-13 -->
### Architecture Diagram

**Architecture**

```mermaid
flowchart TD
    Q["Complex Query"] --> AG["Agent Planner LLM with tools, tool_choice auto"]
    AG --> DEC{"Tool calls requested?"}
    DEC -->|Yes| TC["search_knowledge_base: store_name, query"]
    TC --> T1[("tech KB")]
    TC --> T2[("finance KB")]
    TC --> T3[("compliance KB")]
    T1 & T2 & T3 --> OBS["Tool result messages appended"]
    OBS -->|"loop, max 4 iterations"| AG
    DEC -->|No| FIN["Synthesized Final Answer"]
```

**Example Tool-Calling Sequence**

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent LLM
    participant T as Tech KB
    participant F as Finance KB
    participant C as Compliance KB
    U->>A: Cryptography, license cost, FIPS compliance?
    A->>T: search_knowledge_base tech
    T-->>A: AES-256-GCM + ML-KEM-1024
    A->>F: search_knowledge_base finance
    F-->>A: 48,000 USD per year per cluster
    A->>C: search_knowledge_base compliance
    C-->>A: FIPS 140-3 Level 3
    A-->>U: Synthesized multi-source answer
```

### Run
```bash
python main.py
```

