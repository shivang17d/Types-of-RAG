# 30] REVEAL (Visual-Language Model RAG)

### Overview
**REVEAL** grounds visual-language reasoning in real-world facts by combining **Visual-Text Query Encoding**, **Memory Retrieval**, and **Knowledge Fusion**, producing transparent, step-by-step task-solving steps.

<!-- mermaid-30 -->
### Architecture Diagram

```mermaid
flowchart TD
    IMG["Image, base64"] --> QENC["Visual-Text Query Encoder VLM"]
    SCN["Scene Description"] --> QENC
    QQ["Task Question"] --> QENC
    QENC --> EQ["Encoded Visual Query"]
    EQ --> MEM["Memory Retrieval top-2"]
    KB[("External Knowledge Base")] --> MEM
    MEM --> FUS["Knowledge Fusion Generator"]
    SCN --> FUS
    QQ --> FUS
    FUS --> STEPS["Step-by-step reasoning"]
    FUS --> OUT["Grounded Output"]
```

### Run
```bash
python main.py
```

