# 31] CRAT (Causality-Enhanced Reflective and Retrieval-Augmented Translation)

### Overview
**CRAT** is a multi-agent RAG system for high-precision technical translation:
1. **Unknown Terms Detector**: Flags polysemous or domain-specific terms.
2. **Knowledge Graph Constructor**: Retrieves authoritative domain definitions.
3. **Translator**: Produces an initial translation grounded in the KG.
4. **Causality-Enhanced Judge**: Validates cause-and-effect consistency and refines the final output.

<!-- mermaid-31 -->
### Architecture Diagram

```mermaid
flowchart TD
    SRC["Source Text"] --> A1["Agent 1: Unknown Terms Detector"]
    A1 --> TERMS["Ambiguous Terms"]
    TERMS --> A2["Agent 2: KG Constructor - retrieve definition per term"]
    GL[("Domain Glossary Vectors")] --> A2
    A2 --> KG["Term Knowledge Graph"]
    KG --> A3["Agent 3: Translator"]
    SRC --> A3
    A3 --> DRAFT["Draft Translation"]
    DRAFT --> A4{"Agent 4: Causality Judge"}
    KG --> A4
    SRC --> A4
    A4 -->|"causality valid"| OK["Refined Translation"]
    A4 -->|"mismatch found"| FIX["Critique + correction"]
    FIX --> OK
```

### Run
```bash
python main.py
```

