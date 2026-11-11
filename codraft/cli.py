import argparse
import asyncio
from codraft.orchestrator import CoDraftOrchestrator
from codraft.analyzer import QueryAnalyzer
from codraft.config import settings

def main():
    parser = argparse.ArgumentParser(description="CoDraft+ CLI: Multi-LLM Shared Draft System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Collaboratively answer a query")
    ask_parser.add_argument("query", type=str, help="Question or prompt")
    ask_parser.add_argument("--rounds", type=int, default=None, help="Force fixed number of rounds")
    ask_parser.add_argument("--engine", type=str, default="ollama", choices=["ollama", "openai"], help="Inference engine")
    ask_parser.add_argument("--storage", type=str, default="sqlite", choices=["sqlite", "memory"], help="Storage backend")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze query complexity and recommended routing")
    analyze_parser.add_argument("query", type=str, help="Question to analyze")

    args = parser.parse_args()

    if args.command == "analyze":
        analysis = QueryAnalyzer.analyze(args.query, settings.model_a, settings.model_b)
        print("\n=== CoDraft+ Query Analysis ===")
        print(f"Query: {analysis.query}")
        print(f"Domain: {analysis.domain.upper()}")
        print(f"Complexity Score C: {analysis.complexity_score:.3f}")
        print(f"Flags: {', '.join(analysis.flags) if analysis.flags else 'none'}")
        print(f"Recommended Primary Model: {analysis.recommended_primary_model}")
        print(f"Estimated Rounds: {analysis.estimated_rounds}\n")

    elif args.command == "ask":
        settings.engine_type = args.engine
        settings.storage_type = args.storage
        orch = CoDraftOrchestrator()

        print(f"\n[CoDraft+] Initiating collaborative draft for query: '{args.query}'")
        print(f"[CoDraft+] Engine: {args.engine.upper()} | Models: {orch.model_a} + {orch.model_b}")
        
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

if __name__ == "__main__":
    main()
