"""
31] CRAT (Causality-Enhanced Reflective and Retrieval-Augmented Translation)
============================================================================
Pipeline:
Multi-Agent RAG framework for translating ambiguous, domain-heavy text:
1. Agent 1 (Unknown / Ambiguous Terms Detector):
   Detects polysemous, domain-specific, or ambiguous terms in the source text.
2. Agent 2 (Knowledge Graph Constructor):
   Retrieves internal/external domain definitions and builds a contextual term Knowledge Graph.
3. Agent 3 (Initial Translator):
   Translates the source text into the target language using the Knowledge Graph.
4. Agent 4 (Causality-Enhanced Judge):
   Validates whether the translated terms preserve causal relationships and domain context,
   refining the output if any causal mismatch is found.
"""

import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class CRATTranslationRAG:
    def __init__(self, glossary_docs: List[str], emb_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.glossary_docs = glossary_docs
        self.emb_model = emb_model
        self.chat_model = chat_model

        resp = self.client.embeddings.create(model=self.emb_model, input=glossary_docs)
        self.embeddings = [np.array(d.embedding) for d in resp.data]

    def detect_unknown_terms(self, source_text: str) -> List[str]:
        """Agent 1: Detects domain-ambiguous terms requiring retrieval disambiguation."""
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": 'Identify domain-specific or ambiguous terms in the source text. Return JSON: {"ambiguous_terms": [str]}',
                },
                {"role": "user", "content": source_text},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content).get("ambiguous_terms", [])

    def construct_term_knowledge_graph(self, terms: List[str]) -> List[Dict]:
        """Agent 2: Retrieves definitions for ambiguous terms and builds a contextual KG."""
        retrieved_entries = []
        for term in terms:
            q_emb = np.array(
                self.client.embeddings.create(model=self.emb_model, input=[term]).data[0].embedding
            )
            sims = [
                (float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-10)), i)
                for i, d_emb in enumerate(self.embeddings)
            ]
            sims.sort(key=lambda x: x[0], reverse=True)
            retrieved_entries.append({"term": term, "knowledge": self.glossary_docs[sims[0][1]]})
        return retrieved_entries

    def translate_and_judge_causality(self, source_text: str, target_language: str, kg_entries: List[Dict]) -> Dict:
        """Agents 3 & 4: Translates using the KG and validates causal alignment via the Judge Agent."""
        kg_str = "\n".join([f"- {e['term']}: {e['knowledge']}" for e in kg_entries])

        # Agent 3: Draft Translation
        draft_resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": f"Translate the source text into {target_language} using the Knowledge Graph definitions.",
                },
                {"role": "user", "content": f"Knowledge Graph:\n{kg_str}\n\nSource Text: {source_text}"},
            ],
            temperature=0.0,
        )
        draft_translation = draft_resp.choices[0].message.content.strip()

        # Agent 4: Causality-Enhanced Judge
        judge_resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the CRAT Causality-Enhanced Judge Agent. Verify that the cause-and-effect "
                        "relationships and domain terminology in the draft translation strictly align with "
                        "the source context and Knowledge Graph. Provide the validated/refined translation. "
                        'Return JSON: {"causality_valid": bool, "judge_critique": str, "refined_translation": str}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Source Text: {source_text}\n"
                        f"Knowledge Graph:\n{kg_str}\n"
                        f"Draft Translation ({target_language}): {draft_translation}"
                    ),
                },
            ],
            temperature=0.0,
        )
        return {
            "draft_translation": draft_translation,
            **json.loads(judge_resp.choices[0].message.content),
        }

    def run(self, source_text: str, target_language: str = "Spanish") -> Dict:
        terms = self.detect_unknown_terms(source_text)
        kg = self.construct_term_knowledge_graph(terms)
        result = self.translate_and_judge_causality(source_text, target_language, kg)
        return {"source_text": source_text, "detected_terms": terms, "knowledge_graph": kg, **result}


if __name__ == "__main__":
    domain_glossary = [
        "In semiconductor lithography, 'resist poisoning' (envenenamiento de fotorresistencia) occurs when amine contaminants neutralize acid catalysts in chemically amplified photoresists, causing T-topping defects.",
        "In power grids, 'islanding' (operación en isla) refers to a distributed generator continuing to power a local circuit after the main utility grid disconnects.",
    ]

    crat = CRATTranslationRAG(domain_glossary)
    src = "Trace amine outgassing from the reticle pod triggered resist poisoning, which prevented catalytic deprotection and caused severe T-topping across wafer lot #9."
    res = crat.run(src, target_language="Spanish")

    print("Detected Ambiguous Terms:", res["detected_terms"])
    print("Draft Translation       :", res["draft_translation"])
    print("Judge Critique          :", res["judge_critique"])
    print("Refined Translation     :", res["refined_translation"])

