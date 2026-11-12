import asyncio
import json
import sys
import os
import argparse
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from codraft.orchestrator import CoDraftOrchestrator
from codraft.storage import InMemoryStorage
from codraft.engine import BaseEngine, HuggingFaceEngine, OllamaEngine
from codraft.model_sets import get_model_set, MODEL_SETS
from codraft.config import settings

BENCHMARK_QUERIES = [
    {
        "id": "q1_reasoning",
        "category": "Reasoning",
        "query": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain the logical deduction step-by-step.",
        "reference": "No, we cannot conclude that some roses fade quickly. The subset of flowers that fade quickly may not intersect with the subset of flowers that are roses."
    },
    {
        "id": "q2_factual",
        "category": "Factual QA",
        "query": "What causes the apparent retrograde motion of planets in the night sky from an observer on Earth?",
        "reference": "Apparent retrograde motion is an optical illusion caused by the perspective of an observer on Earth overtaking a slower-moving outer planet in its orbit around the Sun."
    },
    {
        "id": "q3_technical",
        "category": "Technical",
        "query": "Explain how optical flow algorithms compute velocity using temporal and spatial image intensity gradients.",
        "reference": "Optical flow uses the brightness constancy constraint: I_x * u + I_y * v + I_t = 0, combining spatial gradients and temporal frame differences to estimate image velocity."
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

async def run_experiment_protocol(track_name: str, engine_type: str = "huggingface") -> List[Dict[str, Any]]:
    mset = get_model_set(track_name)
    results = []

    print(f"\n================================================================================")
    print(f"CoDraft+ Sub-50M Controlled Experiment: {mset.family_name.upper()} Track ({track_name})")
    print(f"Small A: {mset.small_a.name} ({mset.small_a.params_millions}M) | Small B: {mset.small_b.name} ({mset.small_b.params_millions}M)")
    print(f"Combined Small: {mset.combined_small_params}M | Large Model Target: {mset.large.name} ({mset.large_params}M, {mset.size_ratio}x)")
    print(f"Training Data: {mset.small_a.training_data}")
    print(f"================================================================================\n")

    experiments = [
        {"id": "E1", "name": f"E1: Small A Solo ({mset.small_a.name})", "model_a": mset.small_a.hf_id, "model_b": mset.small_a.hf_id, "comm": False, "rounds": 1},
        {"id": "E2", "name": f"E2: Small B Solo ({mset.small_b.name})", "model_a": mset.small_b.hf_id, "model_b": mset.small_b.hf_id, "comm": False, "rounds": 1},
        {"id": "E3", "name": "E3: Small A + B (Isolated, No Communication)", "model_a": mset.small_a.hf_id, "model_b": mset.small_b.hf_id, "comm": False, "rounds": 2},
        {"id": "E4", "name": "E4: CoDraft+ (A + B Shared-Draft Collaboration)", "model_a": mset.small_a.hf_id, "model_b": mset.small_b.hf_id, "comm": True, "rounds": 3},
        {"id": "E5", "name": f"E5: Large Model Solo ({mset.large.name})", "model_a": mset.large.hf_id, "model_b": mset.large.hf_id, "comm": False, "rounds": 1},
    ]

    engine: BaseEngine
    if engine_type == "huggingface":
        engine = HuggingFaceEngine()
    elif engine_type == "openai":
        from codraft.engine import OpenAIEngine
        engine = OpenAIEngine()
    else:
        engine = OllamaEngine(base_url=settings.ollama_base_url)

    for exp in experiments:
        total_time = 0.0
        total_in_tokens = 0
        total_out_tokens = 0
        total_rounds = 0
        peak_ram = 0.0
        f1_scores = []

        for q in BENCHMARK_QUERIES:
            if exp["id"] == "E3":
                # Isolated Ensemble: Model A and Model B generate independently from scratch
                storage_a = InMemoryStorage()
                orch_a = CoDraftOrchestrator(storage=storage_a, engine=engine, model_a=mset.small_a.hf_id, model_b=mset.small_a.hf_id)
                res_a = await orch_a.execute(query=q["query"], force_rounds=1, enable_evidence=False)

                storage_b = InMemoryStorage()
                orch_b = CoDraftOrchestrator(storage=storage_b, engine=engine, model_a=mset.small_b.hf_id, model_b=mset.small_b.hf_id)
                res_b = await orch_b.execute(query=q["query"], force_rounds=1, enable_evidence=False)

                combined_answer = f"{res_a.final_answer}\n\n{res_b.final_answer}".strip()
                total_time += (res_a.metrics.duration_sec + res_b.metrics.duration_sec)
                total_in_tokens += (res_a.metrics.input_tokens + res_b.metrics.input_tokens)
                total_out_tokens += (res_a.metrics.output_tokens + res_b.metrics.output_tokens)
                total_rounds += 2
                peak_ram = max(peak_ram, res_a.metrics.peak_ram_mb, res_b.metrics.peak_ram_mb)
                f1_scores.append(compute_token_f1(combined_answer, q["reference"]))
            else:
                storage = InMemoryStorage()
                orch = CoDraftOrchestrator(
                    storage=storage,
                    engine=engine,
                    model_a=exp["model_a"],
                    model_b=exp["model_b"]
                )

                res = await orch.execute(
                    query=q["query"],
                    force_rounds=exp["rounds"],
                    enable_evidence=exp["comm"]
                )

                total_time += res.metrics.duration_sec
                total_in_tokens += res.metrics.input_tokens
                total_out_tokens += res.metrics.output_tokens
                total_rounds += res.rounds_executed
                peak_ram = max(peak_ram, res.metrics.peak_ram_mb)
                f1_scores.append(compute_token_f1(res.final_answer, q["reference"]))

        n = len(BENCHMARK_QUERIES)
        results.append({
            "Experiment": exp["id"],
            "Configuration": exp["name"],
            "Avg Latency (s)": round(total_time / n, 3),
            "Avg Rounds": round(total_rounds / n, 1),
            "Avg Total Tokens": round((total_in_tokens + total_out_tokens) / n, 1),
            "Peak RAM (MB)": round(peak_ram, 1),
            "F1 Score (%)": round(sum(f1_scores) / n, 2)
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="CoDraft+ Sub-50M Controlled Benchmark Runner")
    parser.add_argument("--track", type=str, default="babylm", choices=list(MODEL_SETS.keys()), help="Model Track")
    parser.add_argument("--engine", type=str, default="huggingface", choices=["huggingface", "ollama", "openai"], help="Engine")
    args = parser.parse_args()

    try:
        results = asyncio.run(run_experiment_protocol(track_name=args.track, engine_type=args.engine))
        headers = list(results[0].keys())
        rows = [[r[h] for h in headers] for r in results]
        try:
            from tabulate import tabulate
            print(tabulate(rows, headers=headers, tablefmt="github"))
        except ImportError:
            print(json.dumps(results, indent=2))
        
        print("\nPrimary Research Comparison: E4 (CoDraft+ Collaboration) vs E5 (Large 2x Model)")
        print("Information-Exchange Control: E3 (Isolated A+B) vs E4 (Shared Draft Exchange)")
    except ConnectionError as e:
        print(f"\n[BENCHMARK HALTED - FAIL CLOSED]: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
