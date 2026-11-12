import argparse
import asyncio
from codraft.orchestrator import CoDraftOrchestrator
from codraft.analyzer import QueryAnalyzer
from codraft.model_sets import get_model_set, MODEL_SETS
from codraft.config import settings

def main():
    parser = argparse.ArgumentParser(description="CoDraft+ CLI: Multi-LLM Shared Draft System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Collaboratively answer a query")
    ask_parser.add_argument("query", type=str, help="Question or prompt")
    ask_parser.add_argument("--rounds", type=int, default=None, help="Force fixed number of rounds")
    ask_parser.add_argument("--engine", type=str, default="huggingface", choices=["huggingface", "ollama", "openai"], help="Inference engine")
    ask_parser.add_argument("--track", type=str, default="babylm", choices=list(MODEL_SETS.keys()), help="Sub-50M Model Track")
    ask_parser.add_argument("--storage", type=str, default="sqlite", choices=["sqlite", "memory"], help="Storage backend")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze query complexity and recommended routing")
    analyze_parser.add_argument("query", type=str, help="Question to analyze")
    analyze_parser.add_argument("--track", type=str, default="babylm", choices=list(MODEL_SETS.keys()), help="Sub-50M Model Track")

    run_parser = subparsers.add_parser("run", help="Run an individual model directly in terminal (like ollama run)")
    run_parser.add_argument("model", type=str, help="Model name (e.g. pythia-14m, pythia-31m, pythia-70m, or Hugging Face ID)")
    run_parser.add_argument("prompt", type=str, nargs="?", default=None, help="Optional prompt. If omitted, starts interactive chat.")

    args = parser.parse_args()

    if args.command == "analyze":
        mset = get_model_set(args.track)
        analysis = QueryAnalyzer.analyze(args.query, mset.small_a.hf_id, mset.small_b.hf_id)
        print("\n=== CoDraft+ Query Analysis ===")
        print(f"Track: {mset.family_name} ({args.track})")
        print(f"Query: {analysis.query}")
        print(f"Domain: {analysis.domain.upper()}")
        print(f"Complexity Score C: {analysis.complexity_score:.3f}")
        print(f"Flags: {', '.join(analysis.flags) if analysis.flags else 'none'}")
        print(f"Recommended Primary Model: {analysis.recommended_primary_model}")
        print(f"Estimated Rounds: {analysis.estimated_rounds}\n")

    elif args.command == "ask":
        mset = get_model_set(args.track)
        settings.active_track = args.track
        settings.engine_type = args.engine
        settings.storage_type = args.storage
        settings.model_a = mset.small_a.hf_id
        settings.model_b = mset.small_b.hf_id
        settings.model_large = mset.large.hf_id

        orch = CoDraftOrchestrator(model_a=settings.model_a, model_b=settings.model_b)

        print(f"\n[CoDraft+] Initiating collaborative draft for query: '{args.query}'")
        print(f"[CoDraft+] Track: {mset.family_name} | Engine: {args.engine.upper()}")
        print(f"[CoDraft+] Small A: {mset.small_a.name} ({mset.small_a.params_millions}M) | Small B: {mset.small_b.name} ({mset.small_b.params_millions}M)")
        
        async def run():
            return await orch.execute(args.query, force_rounds=args.rounds)

        result = asyncio.run(run())

        print(f"\n=== Final Answer (v{result.final_version}) ===")
        print(result.final_answer)
        print("\n=== Collaboration Metadata ===")
        print(f"Rounds Executed: {result.rounds_executed}")
        print(f"Termination Reason: {result.terminated_reason}")
        print(f"Models Contributed: {', '.join(result.models_used)}")
        print(f"Latency: {result.metrics.duration_sec:.2f}s | Peak RAM: {result.metrics.peak_ram_mb:.1f} MB | CPU: {result.metrics.cpu_utilization_pct}%")
        print(f"Tokens: {result.metrics.input_tokens} In / {result.metrics.output_tokens} Out | Energy: ~{result.metrics.estimated_energy_joules:.1f} J")
        print(f"Claims Tracked: {len(result.claims)}")
        for idx, patch in enumerate(result.patch_history, 1):
            print(f"  Turn {idx} [{patch['editor']}]: {patch['operation'].upper()} -> {patch['section_id']} ({patch.get('reason', '')})")

    elif args.command == "run":
        from codraft.engine import HuggingFaceEngine

        shortcuts = {
            "pythia-14m": "EleutherAI/pythia-14m",
            "pythia-31m": "EleutherAI/pythia-31m",
            "pythia-70m": "EleutherAI/pythia-70m",
            "14m": "EleutherAI/pythia-14m",
            "31m": "EleutherAI/pythia-31m",
            "70m": "EleutherAI/pythia-70m",
        }
        target_model = shortcuts.get(args.model.lower(), args.model)
        engine = HuggingFaceEngine()

        print(f"\n[CoDraft+ CLI] Loading individual model: {target_model}...")
        hf_mod, tok = engine._load_model(target_model)
        param_count = sum(p.numel() for p in hf_mod.parameters())
        print(f"[CoDraft+ CLI] Ready! Model parameters: {param_count:,}")

        async def query_model(text: str) -> str:
            messages = [{"role": "user", "content": text}]
            return await engine.chat(model=target_model, messages=messages, temperature=0.7)

        if args.prompt:
            ans = asyncio.run(query_model(args.prompt))
            print(f"\n>>> {ans}\n")
        else:
            print("\nEntering interactive mode. Type your message and press Enter (type 'exit' or 'quit' to leave):\n")
            while True:
                try:
                    user_input = input(">>> ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ["exit", "quit", "q"]:
                        print("Exiting.")
                        break
                    response = asyncio.run(query_model(user_input))
                    print(f"\n{response}\n")
                except (KeyboardInterrupt, EOFError):
                    print("\nExiting.")
                    break

if __name__ == "__main__":
    main()
