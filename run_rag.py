"""
Unified CLI Launcher & Verifier for All 31 RAG Implementations
==============================================================
Usage:
  python3 run_rag.py --list          # List all 31 RAG folders
  python3 run_rag.py --verify        # Syntax & AST verification of all 31 RAG scripts
  python3 run_rag.py --id 5          # Run a specific RAG implementation (e.g., 05] Hybrid_Fusion_RAG)
"""

import argparse
import ast
import glob
import os
import subprocess
import sys
from typing import List, Dict


def discover_rag_folders(base_dir: str) -> List[Dict]:
    pattern = os.path.join(base_dir, "[0-9][0-9]] *")
    folders = sorted(glob.glob(pattern))
    entries = []
    for folder in folders:
        basename = os.path.basename(folder)
        num_str = basename.split("]")[0].strip()
        main_py = os.path.join(folder, "main.py")
        readme_md = os.path.join(folder, "README.md")
        entries.append(
            {
                "id": int(num_str),
                "code": num_str,
                "name": basename,
                "folder": folder,
                "main_py": main_py,
                "readme_md": readme_md,
            }
        )
    return entries


def verify_all(entries: List[Dict]) -> bool:
    print(f"Verifying {len(entries)} RAG implementations...\n")
    all_ok = True
    for item in entries:
        main_path = item["main_py"]
        if not os.path.exists(main_path):
            print(f"[MISSING] {item['name']}/main.py")
            all_ok = False
            continue
        with open(main_path, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            tree = ast.parse(source, filename=main_path)
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            print(f"  [OK] {item['name']:<40} | Classes: {', '.join(classes)}")
        except SyntaxError as e:
            print(f"  [SYNTAX ERROR] {item['name']}: {e}")
            all_ok = False

    print(f"\nVerification {'PASSED' if all_ok else 'FAILED'} ({len(entries)}/31 folders checked).")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Run or verify any of the 31 RAG architectures.")
    parser.add_argument("--list", action="store_true", help="List all 31 RAG folders")
    parser.add_argument("--verify", action="store_true", help="Verify AST and classes of all 31 RAG scripts")
    parser.add_argument("--id", type=int, help="Run the RAG implementation with this number (1-31)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    entries = discover_rag_folders(base_dir)

    if args.verify:
        sys.exit(0 if verify_all(entries) else 1)

    if args.id is not None:
        match = next((e for e in entries if e["id"] == args.id), None)
        if not match:
            print(f"Error: No RAG folder found with ID {args.id} (valid range: 1-31).")
            sys.exit(1)
        print(f"=== Running {match['name']} ===")
        subprocess.run([sys.executable, match["main_py"]], check=True)
        return

    # Default or --list
    print("=== 31 Types of RAG (OpenAI Implementation Catalog) ===")
    for item in entries:
        print(f"  {item['name']}")
    print("\nTip: Run `python3 run_rag.py --id <1-31>` to execute any RAG pipeline, or `python3 run_rag.py --verify` to inspect all.")


if __name__ == "__main__":
    main()

