"""
RAG Benchmark Suite
===================
Runs sample evaluation queries across different RAG architectures and grades
their performance using the RAG Triad Evaluator (Context Relevance, Faithfulness,
Answer Relevance).

Usage:
  python3 benchmark_suite.py --sample
"""

import os
import json
import argparse
from typing import List, Dict
from dotenv import load_dotenv

from rag_evaluator import RAGTriadEvaluator

# Import sample RAG architectures
import importlib.util

def load_module(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BENCHMARK_DATASET = [
    {
        "query": "What encryption does AuroraDB v4.2 use and what is its write latency?",
        "target_rag_ids": [1, 5, 10], # Standard, Hybrid, Self-RAG
        "ground_truth": "AuroraDB v4.2 uses Quantum-Safe Encryption (lattice-based cryptography) and has a 1.4ms write latency."
    },
    {
        "query": "Should we adopt the Helios Solar Array based on ROI and technical specs?",
        "target_rag_ids": [7, 13], # Branched RAG, Agentic RAG
        "ground_truth": "Yes, it has 26.8% conversion efficiency and an estimated payback period of 4.2 years."
    }
]

def run_benchmark():
    evaluator = RAGTriadEvaluator()
    results = []

    print("=" * 70)
    print("STARTING RAG BENCHMARK & TRIAD EVALUATION")
    print("=" * 70)

    for item in BENCHMARK_DATASET:
        query = item["query"]
        print(f"\n[Test Query]: {query}")
        
        # Test Standard RAG (01) as baseline
        try:
            std_mod = load_module(os.path.join(BASE_DIR, "01] Standard_Naive_RAG", "main.py"), "std_rag")
            rag_instance = std_mod.StandardNaiveRAG()
            sample_docs = [
                "AuroraDB v4.2 introduces Quantum-Safe Encryption using lattice-based cryptography. It reduces write latency to 1.4 milliseconds.",
                "Helios Solar Array uses bifacial perovskite cells with 26.8% conversion efficiency and has a 4.2-year ROI payback period."
            ]
            rag_instance.index_documents(sample_docs)
            rag_res = rag_instance.query(query, top_k=1)
            
            # Extract retrieved contexts
            contexts = [c["chunk"] for c in rag_res["retrieved_context"]]
            eval_scores = evaluator.evaluate(query, contexts, rag_res["answer"])
            
            record = {
                "rag_name": "01] Standard_Naive_RAG",
                "query": query,
                "answer": rag_res["answer"],
                "triad_scores": eval_scores
            }
            results.append(record)
            
            print(f"  -> Pipeline: 01] Standard_Naive_RAG")
            print(f"     Overall Triad Score: {eval_scores['overall_triad_score']} | Faithfulness: {eval_scores['faithfulness']}")
            print(f"     Judge Critique: {eval_scores.get('critique', '')}")
            
        except Exception as e:
            print(f"  -> Error running standard benchmark: {e}")

    # Save summary report
    report_path = os.path.join(BASE_DIR, "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark completed! Results saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Run sample benchmark")
    args = parser.parse_args()
    run_benchmark()

