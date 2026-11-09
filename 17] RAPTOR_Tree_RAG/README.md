# 17] RAPTOR (Tree-Organized Retrieval)

### Overview
**RAPTOR** recursively clusters and summarizes text chunks bottom-up into a hierarchical abstraction tree:
- **Level 0 (Leaves)**: Fine-grained original chunks.
- **Level 1 (Internal Nodes)**: Cluster-level thematic summaries.
- **Level 2 (Root)**: Global corpus summary.

Using **Collapsed Tree Retrieval**, queries retrieve from all levels simultaneously to combine macro themes with micro details.

<!-- mermaid-17 -->
### Architecture Diagram

**Tree Build + Collapsed Retrieval**

```mermaid
flowchart TD
    CH["Leaf Chunks"] --> E0["Embed: Level 0 nodes"]
    E0 --> LOOP{"More than 1 node at current level?"}
    LOOP -->|Yes| CL["Farthest-first clustering, k = n/2"]
    CL --> SUM["LLM summarizes each cluster"]
    SUM --> EN["Embed summaries: Level L+1 nodes"]
    EN --> LOOP
    LOOP -->|No| TREE[("Collapsed Tree: every node, every level")]
    Q["Query"] --> QS["Cosine search over all tree nodes"]
    TREE --> QS
    QS --> TOP["Top-K mixed-level nodes"]
    TOP --> G["Generator LLM"]
    G --> A["Answer: broad themes + specific details"]
```

**Resulting Tree**

```mermaid
flowchart BT
    L1["Leaf: Q1 phishing incident"] --> C1["Level 1: Security incidents summary"]
    L2["Leaf: Q2 S3 bucket exposure"] --> C1
    L3["Leaf: Q3 mTLS service mesh"] --> C2["Level 1: Infrastructure upgrades summary"]
    L4["Leaf: Q4 HSM key rotation"] --> C2
    C1 --> ROOT["Level 2: Root summary - yearly security evolution"]
    C2 --> ROOT
```

### Run
```bash
python main.py
```

