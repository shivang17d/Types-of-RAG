# 27] Attention-based RAG (ATLAS / Fusion-in-Decoder)

### Overview
**ATLAS** combines a **Dual-Encoder Retriever** over a hot-updatable document index with a **Fusion-in-Decoder (FiD)** architecture. Each `(Query, Passage_i)` pair is encoded independently with attention salience weights and fused in the decoder.

<!-- mermaid-27 -->
### Architecture Diagram

```mermaid
flowchart TD
    NEW["New Documents"] --> HOT["Hot Index Update - no retraining"]
    HOT --> IDX[("Document Index")]
    Q["Query"] --> DE["Dual-Encoder Retrieval top-3"]
    IDX --> DE
    DE --> P1["Passage 1"] & P2["Passage 2"] & P3["Passage 3"]
    P1 --> E1["Encoder: query + passage 1 to representation + attention weight"]
    P2 --> E2["Encoder: query + passage 2 to representation + attention weight"]
    P3 --> E3["Encoder: query + passage 3 to representation + attention weight"]
    E1 & E2 & E3 --> SORT["Sort by attention weight"]
    SORT --> FID["Fusion-in-Decoder Reader"]
    FID --> A["Answer"]
```

### Run
```bash
python main.py
```

