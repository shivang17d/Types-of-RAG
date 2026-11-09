# 14] ReAct (Reason + Act) RAG

### Overview
**ReAct RAG** interleaves explicit verbal reasoning (`Thought`) with environment interactions (`Action: Search[...]`) and feedback (`Observation`). Each observation updates the agent's situational context to guide the next logical thought.

<!-- mermaid-14 -->
### Architecture Diagram

**Architecture**

```mermaid
flowchart TD
    Q["Question"] --> LOG[("History Log: thoughts, actions, observations")]
    LOG --> LLM["ReAct LLM: JSON thought, action_type, action_input"]
    LLM --> T["Thought"]
    T --> ACT{"action_type"}
    ACT -->|SEARCH| S["Search environment top-1"]
    S --> O["Observation"]
    O -->|"append step"| LOG
    ACT -->|FINISH| F["Finish: final answer"]
```

**Example Trajectory**

```mermaid
sequenceDiagram
    participant A as ReAct Agent
    participant E as Search Environment
    Note over A: Thought 1 - find the flagship engine of Chronos Labs
    A->>E: Search Chronos Labs flagship database
    E-->>A: Observation - ChronoStore
    Note over A: Thought 2 - who created ChronoStore
    A->>E: Search ChronoStore creator
    E-->>A: Observation - Kenji Takahashi
    Note over A: Thought 3 - where did Kenji work before
    A->>E: Search Kenji Takahashi previous employer
    E-->>A: Observation - CERN
    Note over A: Finish - CERN
```

### Run
```bash
python main.py
```

