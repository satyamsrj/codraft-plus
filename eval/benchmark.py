import asyncio
import json
import sys
import os
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from codraft.orchestrator import CoDraftOrchestrator
from codraft.storage import InMemoryStorage
from codraft.engine import OllamaEngine
from codraft.config import settings

BENCHMARK_QUERIES = [
    {
        "id": "q1_factual",
        "category": "Factual QA",
        "query": "What is the physical principle behind lidar sensors and how does time-of-flight calculate distance?",
        "reference": "Time of flight measures d = (c * t) / 2 using laser pulse emission and reflection."
    },
    {
        "id": "q2_reasoning",
        "category": "Reasoning / Math",
        "query": "Explain error propagation in optical flow odometry for a drone experiencing altitude changes and camera tilt.",
        "reference": "Velocity errors scale inversely with ground distance z and linearly with rotational pitch and roll rates."
    },
    {
        "id": "q3_technical",
        "category": "Technical Explainer",
        "query": "Compare the Lucas-Kanade differential optical flow algorithm against Farneback dense optical flow in robotic vision.",
        "reference": "Lucas-Kanade assumes spatial coherence and constant flow in a neighborhood; Farneback approximates neighborhoods with quadratic polynomials."
    },
    {
        "id": "q4_code",
        "category": "Code / Algorithmic",
        "query": "Implement a Python function to compute 1D Kalman filter state updates given measurement z and covariance R.",
        "reference": "Kalman gain K = P / (P + R); x = x + K * (z - x); P = (1 - K) * P."
    }
]

def compute_token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens.intersection(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return round((2.0 * precision * recall) / (precision + recall) * 100.0, 2)

async def run_baseline_evaluation() -> List[Dict[str, Any]]:
    results = []
    
    baselines = [
        {"name": "E1: Single LLM (1-pass)", "rounds": 1, "evidence": False},
        {"name": "E2: Single LLM + Self-Refine (2-pass)", "rounds": 2, "evidence": False},
        {"name": "E3: Sequential Cascade (3-pass)", "rounds": 3, "evidence": False},
        {"name": "E5: Fixed Shared Draft (4-round fixed)", "rounds": 4, "evidence": False},
        {"name": "E6: CoDraft+ (Adaptive + Evidence)", "rounds": None, "evidence": True},
    ]

    for b in baselines:
        total_time = 0.0
        total_in_tokens = 0
        total_out_tokens = 0
        total_rounds = 0
        peak_ram = 0.0
        f1_scores = []

        for q in BENCHMARK_QUERIES:
            storage = InMemoryStorage()
            engine = OllamaEngine(base_url=settings.ollama_base_url)
            orch = CoDraftOrchestrator(storage=storage, engine=engine)

            res = await orch.execute(
                query=q["query"],
                force_rounds=b["rounds"],
                enable_evidence=b["evidence"]
            )

            total_time += res.metrics.duration_sec
            total_in_tokens += res.metrics.input_tokens
            total_out_tokens += res.metrics.output_tokens
            total_rounds += res.rounds_executed
            peak_ram = max(peak_ram, res.metrics.peak_ram_mb)
            f1_scores.append(compute_token_f1(res.final_answer, q["reference"]))

        n = len(BENCHMARK_QUERIES)
        avg_time = round(total_time / n, 3)
        avg_rounds = round(total_rounds / n, 2)
        avg_tokens = round((total_in_tokens + total_out_tokens) / n, 1)
        avg_f1 = round(sum(f1_scores) / n, 2)

        results.append({
            "Baseline Setup": b["name"],
            "Avg Latency (s)": avg_time,
            "Avg Rounds": avg_rounds,
            "Avg Total Tokens": avg_tokens,
            "Peak RAM (MB)": round(peak_ram, 1),
            "F1 Score": avg_f1
        })

    return results

def main():
    try:
        results = asyncio.run(run_baseline_evaluation())
        headers = list(results[0].keys())
        rows = [[r[h] for h in headers] for r in results]
        try:
            from tabulate import tabulate
            print(tabulate(rows, headers=headers, tablefmt="github"))
        except ImportError:
            print(json.dumps(results, indent=2))
    except ConnectionError as e:
        print(f"BENCHMARK ABORTED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
