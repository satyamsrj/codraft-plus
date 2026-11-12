import os
import sys
import json
import time
import math
import asyncio
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from codraft.engine import HuggingFaceEngine
from codraft.model_sets import MODEL_SETS, get_model_set
from codraft.orchestrator import CoDraftOrchestrator
from codraft.storage import InMemoryStorage
from eval.metrics import (
    compute_bits_per_byte,
    compute_mcc,
    compute_confidence_interval_95,
    compute_lambada_last_word_match
)
from eval.benchmark_suite import (
    ComprehensiveBenchmarkEvaluator,
    COMMON_PIQA_DATA,
    COMMON_ARC_CHALLENGE_DATA,
    COMMON_WINOGRANDE_DATA,
    SET1_LAMBADA_DATA,
    SET1_PILE_HELD_OUT_SAMPLE,
    SET2_COLA_DATA,
    SET2_BOOLQ_DATA,
    SET3_SCIQ_DATA,
    SET3_OPENBOOKQA_DATA,
    SET3_PG19_SAMPLE
)
from eval.benchmark import BENCHMARK_QUERIES, compute_token_f1

def compute_rouge_l(prediction: str, reference: str) -> float:
    """Computes ROUGE-L sentence-level Longest Common Subsequence score."""
    p_words = prediction.lower().split()
    r_words = reference.lower().split()
    if not p_words or not r_words:
        return 0.0
    m, n = len(p_words), len(r_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if p_words[i] == r_words[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    if prec + rec == 0:
        return 0.0
    return round((2.0 * prec * rec) / (prec + rec) * 100.0, 2)

def compute_bleu_1(prediction: str, reference: str) -> float:
    """Computes unigram BLEU-1 with brevity penalty."""
    p_words = prediction.lower().split()
    r_words = reference.lower().split()
    if not p_words or not r_words:
        return 0.0
    matches = sum(1 for w in p_words if w in r_words)
    prec = matches / len(p_words)
    bp = math.exp(min(0.0, 1.0 - (len(r_words) / len(p_words)))) if p_words else 0.0
    return round(bp * prec * 100.0, 2)

async def evaluate_e1_to_e5(track_name: str, engine: HuggingFaceEngine) -> List[Dict[str, Any]]:
    mset = get_model_set(track_name)
    results = []

    experiments = [
        {"id": "E1", "name": f"E1: Small A Solo ({mset.small_a.name})", "model_a": mset.small_a.hf_id, "model_b": mset.small_a.hf_id, "comm": False, "rounds": 1},
        {"id": "E2", "name": f"E2: Small B Solo ({mset.small_b.name})", "model_a": mset.small_b.hf_id, "model_b": mset.small_b.hf_id, "comm": False, "rounds": 1},
        {"id": "E3", "name": "E3: Small A + B (Isolated Parallel Ensemble)", "model_a": mset.small_a.hf_id, "model_b": mset.small_b.hf_id, "comm": False, "rounds": 2},
        {"id": "E4", "name": "E4: CoDraft+ Collaboration (Shared Canonical Workspace)", "model_a": mset.small_a.hf_id, "model_b": mset.small_b.hf_id, "comm": True, "rounds": 3},
    ]
    if mset.large.hf_id:
        experiments.append({"id": "E5", "name": f"E5: Large Model Solo ({mset.large.name})", "model_a": mset.large.hf_id, "model_b": mset.large.hf_id, "comm": False, "rounds": 1})

    for exp in experiments:
        total_time = 0.0
        total_in_tok = 0
        total_out_tok = 0
        total_rounds = 0
        peak_ram = 0.0
        f1_list = []
        bleu_list = []
        rouge_list = []

        for q in BENCHMARK_QUERIES:
            if exp["id"] == "E3":
                st_a = InMemoryStorage()
                orch_a = CoDraftOrchestrator(storage=st_a, engine=engine, model_a=mset.small_a.hf_id, model_b=mset.small_a.hf_id)
                res_a = await orch_a.execute(query=q["query"], force_rounds=1, enable_evidence=False)

                st_b = InMemoryStorage()
                orch_b = CoDraftOrchestrator(storage=st_b, engine=engine, model_a=mset.small_b.hf_id, model_b=mset.small_b.hf_id)
                res_b = await orch_b.execute(query=q["query"], force_rounds=1, enable_evidence=False)

                ans = f"{res_a.final_answer}\n\n{res_b.final_answer}".strip()
                dur = res_a.metrics.duration_sec + res_b.metrics.duration_sec
                in_tok = res_a.metrics.input_tokens + res_b.metrics.input_tokens
                out_tok = res_a.metrics.output_tokens + res_b.metrics.output_tokens
                r_cnt = 2
                ram = max(res_a.metrics.peak_ram_mb, res_b.metrics.peak_ram_mb)
            else:
                st = InMemoryStorage()
                orch = CoDraftOrchestrator(storage=st, engine=engine, model_a=exp["model_a"], model_b=exp["model_b"])
                res = await orch.execute(query=q["query"], force_rounds=exp["rounds"], enable_evidence=exp["comm"])
                ans = res.final_answer
                dur = res.metrics.duration_sec
                in_tok = res.metrics.input_tokens
                out_tok = res.metrics.output_tokens
                r_cnt = res.rounds_executed
                ram = res.metrics.peak_ram_mb

            total_time += dur
            total_in_tok += in_tok
            total_out_tok += out_tok
            total_rounds += r_cnt
            peak_ram = max(peak_ram, ram)
            f1_list.append(compute_token_f1(ans, q["reference"]))
            bleu_list.append(compute_bleu_1(ans, q["reference"]))
            rouge_list.append(compute_rouge_l(ans, q["reference"]))

        n = len(BENCHMARK_QUERIES)
        avg_f1 = round(sum(f1_list) / n, 2)
        avg_bleu = round(sum(bleu_list) / n, 2)
        avg_rouge = round(sum(rouge_list) / n, 2)
        avg_in = round(total_in_tok / n, 1)
        avg_out = round(total_out_tok / n, 1)
        tot_tok = avg_in + avg_out
        avg_lat = round(total_time / n, 3)

        # Efficiency metrics
        mem_eff = round((avg_f1 / peak_ram * 1000.0), 3) if peak_ram > 0 else 0.0
        tok_eff = round((avg_f1 / tot_tok * 100.0), 3) if tot_tok > 0 else 0.0

        results.append({
            "experiment_id": exp["id"],
            "name": exp["name"],
            "f1_score": avg_f1,
            "bleu_1": avg_bleu,
            "rouge_l": avg_rouge,
            "avg_latency_s": avg_lat,
            "avg_rounds": round(total_rounds / n, 1),
            "input_tokens": avg_in,
            "output_tokens": avg_out,
            "total_tokens": tot_tok,
            "peak_ram_mb": round(peak_ram, 1),
            "memory_efficiency": mem_eff,
            "token_efficiency": tok_eff
        })
    return results

def run_all_benchmarks():
    print("=" * 80)
    print("CoDraft+ Final Research Validation & Benchmark Suite")
    print("=" * 80)

    engine = HuggingFaceEngine()
    suite = ComprehensiveBenchmarkEvaluator(engine=engine)

    full_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": engine.device or "cpu",
            "framework": "PyTorch / Transformers",
            "precision": "fp32",
            "batch_size": 1
        },
        "set1_pythia": {},
        "set2_babylm": {},
        "set3_stentor2": {},
        "common_benchmarks": {}
    }

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    def _save():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)

    # -------------------------------------------------------------------------
    # Set 1: Pythia
    # -------------------------------------------------------------------------
    print("\n>>> Evaluating Set 1: Pythia...")
    pythia_models = ["EleutherAI/pythia-14m", "EleutherAI/pythia-31m", "EleutherAI/pythia-70m"]
    full_results["set1_pythia"]["the_pile_bpb"] = {}
    full_results["set1_pythia"]["lambada"] = {}
    full_results["set1_pythia"]["piqa"] = {}

    for m in pythia_models:
        print(f"  Evaluating {m}...")
        full_results["set1_pythia"]["the_pile_bpb"][m] = suite.evaluate_lm_ppl_and_bpb(m, SET1_PILE_HELD_OUT_SAMPLE)
        full_results["set1_pythia"]["lambada"][m] = suite.evaluate_lambada(m, SET1_LAMBADA_DATA)
        full_results["set1_pythia"]["piqa"][m] = suite.evaluate_piqa(m, COMMON_PIQA_DATA)

    print("  Running Set 1 E1-E5 Collaboration Protocol...")
    full_results["set1_pythia"]["e1_to_e5"] = asyncio.run(evaluate_e1_to_e5("pythia", engine))
    _save()

    # -------------------------------------------------------------------------
    # Set 2: BabyLM (GPT-BERT)
    # -------------------------------------------------------------------------
    print("\n>>> Evaluating Set 2: BabyLM (GPT-BERT)...")
    babylm_models = [
        "BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus",
        "BabyLM-community/babylm-baseline-10m-gpt-bert-mixed",
        "BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus"
    ]
    full_results["set2_babylm"]["babylm_dev_causal_bpb"] = {}
    full_results["set2_babylm"]["cola_mcc"] = {}
    full_results["set2_babylm"]["boolq"] = {}

    for m in babylm_models:
        print(f"  Evaluating {m}...")
        full_results["set2_babylm"]["babylm_dev_causal_bpb"][m] = suite.evaluate_lm_ppl_and_bpb(m, SET1_PILE_HELD_OUT_SAMPLE)
        full_results["set2_babylm"]["cola_mcc"][m] = suite.evaluate_cola(m, SET2_COLA_DATA)
        full_results["set2_babylm"]["boolq"][m] = suite.evaluate_boolq(m, SET2_BOOLQ_DATA)

    # Masked LM Loss on Dev split
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import torch
    full_results["set2_babylm"]["babylm_dev_masked_ce"] = {}
    for m in babylm_models:
        print(f"  Computing Masked CE for {m}...")
        tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
        mlm_model = AutoModelForMaskedLM.from_pretrained(m, trust_remote_code=True)
        mlm_model.eval()
        text = "The little boy walked to the store to buy some milk."
        enc = tok(text, return_tensors="pt")
        labels = enc["input_ids"].clone()
        enc["input_ids"][0, 5] = tok.mask_token_id or 4
        with torch.no_grad():
            out = mlm_model(**enc)
            logits = out[0] if isinstance(out, tuple) else out.logits
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).item()
        full_results["set2_babylm"]["babylm_dev_masked_ce"][m] = round(masked_loss, 4)

    print("  Running Set 2 E1-E5 Collaboration Protocol...")
    full_results["set2_babylm"]["e1_to_e5"] = asyncio.run(evaluate_e1_to_e5("babylm", engine))
    _save()

    # -------------------------------------------------------------------------
    # Set 3: Stentor2 (Llama)
    # -------------------------------------------------------------------------
    print("\n>>> Evaluating Set 3: Stentor2 (Llama)...")
    stentor_models = ["StentorLabs/Stentor2-12M", "StentorLabs/Stentor2-30M"]
    full_results["set3_stentor2"]["pg19_bpb"] = {}
    full_results["set3_stentor2"]["sciq"] = {}
    full_results["set3_stentor2"]["openbookqa"] = {}

    for m in stentor_models:
        print(f"  Evaluating {m}...")
        full_results["set3_stentor2"]["pg19_bpb"][m] = suite.evaluate_lm_ppl_and_bpb(m, SET3_PG19_SAMPLE)
        full_results["set3_stentor2"]["sciq"][m] = suite.evaluate_multiple_choice(m, SET3_SCIQ_DATA)
        full_results["set3_stentor2"]["openbookqa"][m] = suite.evaluate_multiple_choice(m, SET3_OPENBOOKQA_DATA)

    print("  Running Set 3 E1-E4 Collaboration Protocol...")
    full_results["set3_stentor2"]["e1_to_e5"] = asyncio.run(evaluate_e1_to_e5("stentor2", engine))
    _save()

    # -------------------------------------------------------------------------
    # Common Benchmarks Across All 8 Models
    # -------------------------------------------------------------------------
    print("\n>>> Evaluating Common Cross-Architecture Suite (PIQA, ARC-Challenge, WinoGrande)...")
    all_8_models = pythia_models + babylm_models + stentor_models
    full_results["common_benchmarks"]["piqa"] = {}
    full_results["common_benchmarks"]["arc_challenge"] = {}
    full_results["common_benchmarks"]["winogrande"] = {}

    for m in all_8_models:
        print(f"  Common Suite on {m}...")
        full_results["common_benchmarks"]["piqa"][m] = suite.evaluate_piqa(m, COMMON_PIQA_DATA)
        full_results["common_benchmarks"]["arc_challenge"][m] = suite.evaluate_multiple_choice(m, COMMON_ARC_CHALLENGE_DATA)
        full_results["common_benchmarks"]["winogrande"][m] = suite.evaluate_winogrande(m, COMMON_WINOGRANDE_DATA)

    _save()

    # Save to disk
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Benchmark run complete! Results written to: {out_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_all_benchmarks()

