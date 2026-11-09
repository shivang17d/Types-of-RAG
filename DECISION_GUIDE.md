# RAG Architecture Decision & Selection Guide

This guide helps you choose the right RAG architecture from the 31 types implemented in this repository based on your business domain, data structure, latency requirements, and accuracy tolerances.

---

## 1. Quick Decision Tree

```
Are your queries simple lookups over static FAQs?
  ├── YES ──► [01] Standard Naive RAG
  └── NO
      │
      Does your data include part numbers, error codes, SKUs, or exact identifiers?
        ├── YES ──► [05] Hybrid / Fusion RAG (BM25 + Dense Vectors + RRF)
        └── NO
            │
            Are user queries vague, abstract, or significantly shorter than document text?
              ├── YES ──► [06] HyDE RAG (Hypothetical Document Embeddings)
              └── NO
                  │
                  Are you answering multi-hop questions ("Who leads the company that created X?")?
                    ├── YES ──► [08] Iterative Multi-Hop RAG or [14] ReAct RAG
                    └── NO
                        │
                        Are hallucinations completely unacceptable (medical, legal, finance)?
                          ├── YES ──► [09] Corrective RAG (CRAG) + [10] Self-RAG
                          └── NO
                              │
                              Does the question require multi-angle analysis (tech, finance, risk)?
                                ├── YES ──► [07] Branched RAG
                                └── NO
                                    │
                                    Do you need global summaries across thousands of interconnected docs?
                                      ├── YES ──► [16] Graph RAG or [17] RAPTOR Tree RAG
                                      └── NO
                                          │
                                          Are you constrained by strict token / API cost limits?
                                            ├── YES ──► [23] CORAG (Cost-Constrained RAG with MCTS)
                                            └── NO
                                                │
                                                Do documents contain charts, diagrams, or images?
                                                  ├── YES ──► [29] Multimodal RAG or [30] REVEAL
                                                  └── NO  ──► [03] Advanced RAG (Query Rewrite + Reranking)
```

---

## 2. Architecture Comparison by Engineering Tradeoffs

| Category | Recommended RAG Types | Retrieval Accuracy | Latency | Compute / Cost | Primary Benefit |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **High Speed / Low Cost** | `01` Standard, `11` Adaptive, `21` Speculative | ⭐⭐⭐ | ⚡ < 1.5s | 💲 Low | Fast response times; low API overhead. |
| **Keyword & Semantic Precision** | `05` Hybrid / Fusion, `03` Advanced | ⭐⭐⭐⭐⭐ | ⚡ 2–3s | 💲💲 Moderate | Resolves exact IDs, error codes, and semantic nuances. |
| **Hallucination Prevention** | `09` CRAG, `10` Self-RAG, `12` REFEED | ⭐⭐⭐⭐⭐ | ⏱️ 3–5s | 💲💲 Moderate | Self-evaluates relevance; falls back when unsure. |
| **Complex Multi-Step Reasoning** | `08` Iterative, `13` Agentic, `14` ReAct | ⭐⭐⭐⭐⭐ | 🐢 5–15s | 💲💲💲 High | Dynamic planning, tool usage, and iterative discovery. |
| **Global Corpus Summaries** | `16` Graph RAG, `17` RAPTOR, `18` ConTReGen | ⭐⭐⭐⭐⭐ | ⏱️ 4–8s | 💲💲💲 High (offline) | Connects distant relationships across documents. |
| **Regulated & Compliance** | `15` Rule RAG, `31` CRAT | ⭐⭐⭐⭐⭐ | ⏱️ 3–5s | 💲💲 Moderate | Enforces strict business policies and causal consistency. |
| **Distributed / Edge Computing** | `24` EACO-RAG | ⭐⭐⭐⭐ | ⚡ < 50ms (at edge)| 💲 Low (bandwidth) | Localized vector caching with cloud escalation. |
| **Multimodal (Images + Text)** | `29` Multimodal, `30` REVEAL | ⭐⭐⭐⭐⭐ | ⏱️ 3–6s | 💲💲💲 High | Simultaneously understands text, tables, and images. |

---

## 3. Recommended Production Roadmap

1. **Step 1 (PoC)**: Start with `[01] Standard Naive RAG` to establish baseline performance.
2. **Step 2 (The 80/20 Sweet Spot)**: Upgrade to `[05] Hybrid Fusion RAG` + Cross-Encoder Reranking (`[03] Advanced RAG`). This fixes ~80% of real-world retrieval failures.
3. **Step 3 (Safety Gate)**: Add `[09] Corrective RAG (CRAG)` and `[10] Self-RAG` reflection checks before passing context to users.
4. **Step 4 (Autonomous Scaling)**: Layer `[13] Agentic RAG` or `[16] Graph RAG` if your product requires complex multi-source tool calling or relationship mapping.

