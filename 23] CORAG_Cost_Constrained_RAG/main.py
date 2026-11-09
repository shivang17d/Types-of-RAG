"""
23] CORAG (Cost-Constrained RAG with Monte Carlo Tree Search)
=============================================================
Pipeline:
Optimizes chunk combination selection under a strict token/cost budget using MCTS:
- Each candidate chunk i has a token cost c_i and non-monotonic joint utility (adding
  redundant or conflicting chunks can reduce quality or exceed budget).
- Uses Monte Carlo Tree Search (MCTS) with UCB1 exploration to search over chunk subsets
  that satisfy sum(c_i) <= Budget, evaluating subset utility via OpenAI.
"""

import os
import math
import json
from typing import List, Dict, Set, Tuple
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class CORAGMCTS:
    def __init__(self, chunks: List[Dict], chat_model: str = "gpt-4o-mini"):
        """Each chunk is {'id': int, 'text': str, 'cost': int}."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chunks = chunks
        self.chat_model = chat_model
        self.utility_cache: Dict[Tuple[int, ...], float] = {}

    def evaluate_subset_utility(self, query: str, selected_ids: Tuple[int, ...]) -> float:
        """Evaluates the joint utility (0.0 to 1.0) of a chunk combination for answering the query."""
        if not selected_ids:
            return 0.0
        if selected_ids in self.utility_cache:
            return self.utility_cache[selected_ids]

        combined_text = "\n".join([f"[Chunk {i}]: {self.chunks[i]['text']}" for i in selected_ids])
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Evaluate the utility of this chunk combination for answering the query (0.0 to 1.0). "
                        "Penalize missing facts, redundancy, or irrelevant noise. "
                        'Return JSON: {"utility": float}'
                    ),
                },
                {"role": "user", "content": f"Query: {query}\n\nSelected Chunks:\n{combined_text}"},
            ],
            temperature=0.0,
        )
        util = float(json.loads(resp.choices[0].message.content).get("utility", 0.0))
        self.utility_cache[selected_ids] = util
        return util

    def mcts_search(self, query: str, max_cost_budget: int, iterations: int = 6) -> Dict:
        visits: Dict[Tuple[int, ...], int] = {(): 0}
        total_reward: Dict[Tuple[int, ...], float] = {(): 0.0}
        best_state: Tuple[int, ...] = ()
        best_utility = -1.0

        for _ in range(iterations):
            state: Tuple[int, ...] = ()
            current_cost = 0
            path = [state]

            # Selection & Expansion within cost budget
            while True:
                valid_next = [
                    c["id"]
                    for c in self.chunks
                    if c["id"] not in state and current_cost + c["cost"] <= max_cost_budget
                ]
                if not valid_next:
                    break

                # Pick next chunk using UCB1 score
                best_next_state = None
                best_ucb = -1e9
                parent_visits = max(1, visits.get(state, 1))

                for cid in valid_next:
                    child = tuple(sorted(state + (cid,)))
                    if visits.get(child, 0) == 0:
                        best_next_state = child
                        break
                    avg_r = total_reward[child] / visits[child]
                    ucb = avg_r + 0.5 * math.sqrt(math.log(parent_visits) / visits[child])
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_next_state = child

                state = best_next_state
                current_cost = sum(self.chunks[i]["cost"] for i in state)
                path.append(state)
                if visits.get(state, 0) == 0:
                    break

            # Simulation / Evaluation
            reward = self.evaluate_subset_utility(query, state)
            if reward > best_utility:
                best_utility = reward
                best_state = state

            # Backpropagation
            for node in path:
                visits[node] = visits.get(node, 0) + 1
                total_reward[node] = total_reward.get(node, 0.0) + reward

        selected_chunks = [self.chunks[i] for i in best_state]
        total_cost = sum(c["cost"] for c in selected_chunks)
        ctx = "\n".join([c["text"] for c in selected_chunks])

        ans = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Answer using the optimal cost-constrained chunk combination."},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"},
            ],
            temperature=0.0,
        ).choices[0].message.content.strip()

        return {
            "best_chunk_ids": best_state,
            "total_cost": total_cost,
            "budget": max_cost_budget,
            "utility_score": best_utility,
            "answer": ans,
        }


if __name__ == "__main__":
    candidate_chunks = [
        {"id": 0, "text": "NebulaDB read throughput is 450,000 QPS per shard.", "cost": 40},
        {"id": 1, "text": "NebulaDB read throughput is approximately 450k queries per second (redundant log).", "cost": 55},
        {"id": 2, "text": "NebulaDB uses Raft consensus with 3-way quorum replication for zero-data-loss durability.", "cost": 45},
        {"id": 3, "text": "Office parking permits must be renewed at the security desk every January.", "cost": 35},
    ]

    corag = CORAGMCTS(candidate_chunks)
    res = corag.mcts_search(
        query="What is NebulaDB's read throughput and how does it guarantee durability?",
        max_cost_budget=90,
        iterations=6,
    )
    print(f"Optimal Chunk Combination IDs: {res['best_chunk_ids']} (Cost: {res['total_cost']}/{res['budget']} | Utility: {res['utility_score']:.2f})")
    print(f"\nFinal Answer:\n{res['answer']}")

