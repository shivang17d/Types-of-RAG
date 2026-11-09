# 29] Multimodal RAG

### Overview
**Multimodal RAG** indexes and retrieves across **Text**, **Structured Tables**, and **Images/Charts** (via OpenAI Vision captioning + raw image payloads) in a shared semantic vector space, synthesizing answers across all media types simultaneously.

<!-- mermaid-29 -->
### Architecture Diagram

```mermaid
flowchart TD
    subgraph INGEST["Multimodal Ingestion"]
        TXT["Text"]
        TAB["Markdown Table"]
        IMG["Image / Chart, base64"]
        VIS["Vision Captioner gpt-4o-mini"]
        TAG["Modality-tagged summaries"]
        EMB[("Unified Vector Space")]
    end
    TXT --> TAG
    TAB --> TAG
    IMG --> VIS --> TAG
    TAG --> EMB
    Q["Query"] --> S["Cross-modal Cosine Search top-3"]
    EMB --> S
    S --> BLK["Content blocks: text summaries + raw image_url"]
    BLK --> VLM["Vision Generator gpt-4o-mini"]
    VLM --> A["Multimodal Answer"]
```

### Run
```bash
python main.py
```

