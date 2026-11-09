# 26] RETRO (Chunked Cross-Attention RAG)

### Overview
**RETRO** splits the input sequence into contiguous chunks. For each chunk $C_u$, it performs $k$-NN search to retrieve both the matching neighbor chunk $N_i$ **and its continuation chunk** $F_i$ from the database, conditioning the generation of the next chunk $C_{u+1}$ on those continuations.

<!-- mermaid-26 -->
### Architecture Diagram

```mermaid
flowchart TD
    subgraph DB["Retrieval Database"]
        PAIRS[("Neighbor chunk N_i paired with continuation F_i")]
    end
    PRE["Prompt Prefix C_1"] --> ACT["Active Chunk C_u"]
    ACT --> KNN["k-NN search on neighbor embeddings, k=2"]
    PAIRS --> KNN
    KNN --> NF["Retrieved pairs N, F"]
    NF --> CCA["Chunked Cross-Attention Decoder LLM"]
    SEQ[("Sequence so far")] --> CCA
    CCA --> NEXT["Next chunk C_u+1"]
    NEXT --> SEQ
    NEXT -->|"repeat num_chunks times"| ACT
    SEQ --> OUT["Full Generated Sequence"]
```

### Run
```bash
python main.py
```

