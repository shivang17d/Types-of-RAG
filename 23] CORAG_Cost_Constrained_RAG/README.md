# 23] CORAG (Cost-Constrained RAG)

### Overview
**CORAG** models chunk selection as a combinatorial optimization problem under a strict token/cost budget. Because adding redundant or noisy chunks can reduce utility (non-monotonicity) while consuming budget, CORAG uses **Monte Carlo Tree Search (MCTS)** to find the optimal chunk combination.

<!-- mermaid-23 -->
### Architecture Diagram

```mermaid
flowchart TD
    Q["Query + Token Budget, e.g. 90"] --> ROOT["MCTS Root: empty chunk set"]
    ROOT --> SEL["Selection: UCB1 = avg reward + 0.5 x sqrt of ln N / n"]
    SEL --> EXP["Expansion: add an unvisited chunk within budget"]
    EXP --> SIM["Simulation: LLM scores subset utility 0-1"]
    CACHE[("Utility Cache")] <--> SIM
    SIM --> BP["Backpropagation: update visits + rewards on path"]
    BP --> IT{"Iterations left?"}
    IT -->|Yes| SEL
    IT -->|No| BEST["Best Subset: max utility, cost within budget"]
    BEST --> G["Generator LLM"]
    G --> A["Answer"]
```

### Run
```bash
python main.py
```

