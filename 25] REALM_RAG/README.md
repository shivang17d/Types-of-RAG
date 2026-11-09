# 25] REALM (Retrieval-Augmented Language Model)

### Overview
**REALM** uses **Maximum Inner Product Search (MIPS)** to compute a retrieval probability distribution $p(z \mid x)$ over the corpus, and marginalizes the generator's predictions across the top-$k$ retrieved documents:

$$p(y \mid x) = \sum_{z \in \text{Top-}K} p(y \mid z, x) \, p(z \mid x)$$

<!-- mermaid-25 -->
### Architecture Diagram

```mermaid
flowchart TD
    X["Query x"] --> QE["Embed x"]
    QE --> MIPS["MIPS: inner product x learnable doc weights"]
    DV[("Doc Vectors + weights")] --> MIPS
    MIPS --> TOPK["Top-K documents z"]
    TOPK --> SM["Temperature-scaled Softmax: p of z given x"]
    SM --> Z1["z1"] & Z2["z2"] & Z3["z3"]
    Z1 --> P1["Knowledge-Augmented Encoder: answer + p of y given z1, x"]
    Z2 --> P2["Knowledge-Augmented Encoder: answer + p of y given z2, x"]
    Z3 --> P3["Knowledge-Augmented Encoder: answer + p of y given z3, x"]
    P1 & P2 & P3 --> MARG["Marginalize: sum over z of p y given z,x times p z given x"]
    MARG --> Y["Highest-contribution prediction y"]
```

### Run
```bash
python main.py
```

