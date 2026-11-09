# 28] REPLUG (Black-Box Retrieval Plugin)

### Overview
**REPLUG** treats the LLM as a frozen black box. Instead of concatenating all retrieved passages into one giant prompt, REPLUG prepends each retrieved document $d_i$ **individually** to the query (`[d_i ; x]`), runs parallel passes, and ensembles the output probabilities weighted by retrieval likelihood $P(d_i \mid x)$.

<!-- mermaid-28 -->
### Architecture Diagram

```mermaid
flowchart TD
    X["Query x"] --> RET["Retrieve top-3 by cosine"]
    RET --> SMX["Softmax with tau = 0.1: P of d_i given x"]
    SMX --> D1["Prepend d1 to x"] & D2["Prepend d2 to x"] & D3["Prepend d3 to x"]
    D1 --> L1["Frozen Black-Box LLM pass 1: answer + confidence"]
    D2 --> L2["Frozen Black-Box LLM pass 2: answer + confidence"]
    D3 --> L3["Frozen Black-Box LLM pass 3: answer + confidence"]
    L1 --> W1["Weight = P of d1 given x times confidence"]
    L2 --> W2["Weight = P of d2 given x times confidence"]
    L3 --> W3["Weight = P of d3 given x times confidence"]
    W1 & W2 & W3 --> ENS["Ensemble Aggregator LLM"]
    ENS --> A["Ensembled Answer"]
```

### Run
```bash
python main.py
```

