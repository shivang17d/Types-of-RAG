# 16] Graph RAG

### Overview
**Graph RAG** extracts entities, relationships, and confidence scores from unstructured text to construct a **Knowledge Graph** alongside **Community Summaries**. During retrieval, it traverses multi-hop relational paths to uncover connections across documents that standard vector search misses.

<!-- mermaid-16 -->
### Architecture Diagram

**Architecture**

```mermaid
flowchart TD
    subgraph BUILD["Graph Construction - offline"]
        DOCS["Documents"]
        EX["LLM Triple Extractor: source, relation, target, confidence"]
        TR["Triples"]
        CS["Community Summary"]
        ADJ[("Bidirectional Adjacency List")]
    end
    subgraph QUERY["Query Time"]
        Q["User Query"]
        SEED["LLM Seed Entity Selector"]
        BFS["BFS Traversal: max 2 hops, confidence at least 0.6"]
        PATHS["Relational Paths"]
        CTX["Graph Context"]
        G["Generator LLM"]
        A["Relational Answer"]
    end
    DOCS --> EX
    EX --> TR --> ADJ
    EX --> CS
    Q --> SEED
    ADJ --> SEED
    SEED --> BFS
    ADJ --> BFS
    BFS --> PATHS --> CTX
    CS --> CTX
    CTX --> G --> A
```

**Example Extracted Knowledge Graph**

```mermaid
graph LR
    BG["BioGenix"] -->|acquired| NP["NeuroPulse Labs"]
    NP -->|owns| SC["SynapCure"]
    MV["Dr. Marcus Vance"] -->|invented| SC
    MV -->|worked at| NP
    SC -->|inhibits| PK["Protein-Kinase-Z"]
    PK -->|drives| IX["Inflammation Syndrome X"]
```

### Run
```bash
python main.py
```

