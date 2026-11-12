import uuid
import math
from typing import Optional, List
from codraft.types import OrchestrationResult, PatchOperation
from codraft.config import settings
from codraft.storage import BaseStorage, SQLiteStorage, InMemoryStorage
from codraft.patcher import ResilientPatcher
from codraft.analyzer import QueryAnalyzer
from codraft.evidence import EvidenceLedger
from codraft.controller import AdaptiveController
from codraft.engine import BaseEngine, OllamaEngine, OpenAIEngine, HuggingFaceEngine
from codraft.profiler import ResourceProfiler

class CoDraftOrchestrator:
    def __init__(
        self,
        storage: Optional[BaseStorage] = None,
        engine: Optional[BaseEngine] = None,
        model_a: Optional[str] = None,
        model_b: Optional[str] = None
    ):
        self.model_a = model_a or settings.model_a
        self.model_b = model_b or settings.model_b

        if storage is not None:
            self.storage = storage
        elif settings.storage_type == "memory":
            self.storage = InMemoryStorage()
        else:
            self.storage = SQLiteStorage(
                db_path=settings.db_path,
                history_file=settings.history_log_path
            )

        if engine is not None:
            self.engine = engine
        elif settings.engine_type == "huggingface":
            self.engine = HuggingFaceEngine()
        elif settings.engine_type == "openai":
            self.engine = OpenAIEngine()
        else:
            self.engine = OllamaEngine(base_url=settings.ollama_base_url)

    async def execute(
        self,
        query: str,
        session_id: Optional[str] = None,
        force_rounds: Optional[int] = None,
        enable_evidence: bool = True
    ) -> OrchestrationResult:
        sid = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        profiler = ResourceProfiler()
        profiler.start()

        analysis = QueryAnalyzer.analyze(query, model_a=self.model_a, model_b=self.model_b)
        state = self.storage.init_session(sid, initial_text="")
        ledger = EvidenceLedger()
        controller = AdaptiveController(
            max_rounds=force_rounds if force_rounds is not None else settings.max_rounds,
            complexity_threshold=settings.complexity_threshold,
            stagnation_diff_threshold=settings.stagnation_diff_threshold,
            stagnation_patience=settings.stagnation_patience,
            w_c=settings.weight_complexity,
            w_u=settings.weight_uncertainty,
            w_d=settings.weight_disagreement,
            w_v=settings.weight_verification
        )

        current_model = analysis.recommended_primary_model
        models_used: List[str] = []
        terminated_reason = "normal_completion"
        last_diff = 1.0
        last_patch: Optional[PatchOperation] = None

        is_fast_path = (force_rounds is None) and controller.should_fast_path_single_model(analysis)
        target_max_rounds = 1 if is_fast_path else (force_rounds or settings.max_rounds)

        system_prompt = (
            "You are an expert AI collaborating on a shared evolving answer. "
            "Instead of rewriting everything, output a structured JSON patch modifying the draft.\n"
            "JSON Schema:\n"
            "{\n"
            '  "section_id": "overview|methodology|limitations|conclusion",\n'
            '  "operation": "update|append|delete|replace_all",\n'
            '  "content": "new or refined content text...",\n'
            '  "reason": "why this change improves accuracy, clarity, or depth",\n'
            '  "confidence": 0.85\n'
            "}\n"
            "Only return valid JSON or markdown enclosed JSON."
        )

        for round_idx in range(1, target_max_rounds + 1):
            if current_model not in models_used:
                models_used.append(current_model)

            current_draft_text = state.full_text if state.full_text.strip() else "(Empty draft - generate initial response)"
            user_prompt = (
                f"Original User Question: {query}\n\n"
                f"Current Draft (v{state.version}):\n{current_draft_text}\n\n"
                f"Critically refine and enhance the draft. Fix any inaccuracies, add missing depth or caveats, and output your JSON patch."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            in_tokens = math.ceil((len(system_prompt) + len(user_prompt)) / 4.0)

            raw_response = await self.engine.chat(
                model=current_model,
                messages=messages,
                keep_alive=settings.model_keep_alive
            )

            out_tokens = math.ceil(len(raw_response) / 4.0)
            profiler.record_tokens(in_tokens, out_tokens)
            profiler.update_peak_memory()

            patch = ResilientPatcher.parse_patch(raw_response, fallback_model=current_model)
            patch.model = current_model
            last_patch = patch

            state, last_diff = ResilientPatcher.apply_patch(state, patch)
            self.storage.save_state(state, patch)

            if enable_evidence:
                ledger.extract_claims(state.full_text, source_model=current_model, version=state.version)
                if patch.confidence and patch.confidence > 0.85:
                    for c in ledger.get_unverified_claims():
                        ledger.verify_claim(c.claim_id, status="verified", confidence=patch.confidence)

            if is_fast_path:
                terminated_reason = f"fast_path_simple_query (C={analysis.complexity_score:.2f} < {settings.complexity_threshold})"
                break

            should_stop, stop_reason, _ = controller.evaluate_stopping(
                current_round=round_idx,
                c_score=analysis.complexity_score,
                last_diff=last_diff,
                state=state,
                ledger=ledger,
                last_patch=last_patch
            )

            if should_stop:
                terminated_reason = stop_reason
                break

            current_model = controller.select_next_model(current_model, self.model_a, self.model_b)

        metrics = profiler.stop()
        history = self.storage.get_history(sid)

        return OrchestrationResult(
            session_id=sid,
            query=query,
            final_answer=state.full_text,
            final_version=state.version,
            rounds_executed=len(history) if history else 1,
            terminated_reason=terminated_reason,
            complexity_score=analysis.complexity_score,
            models_used=models_used,
            claims=ledger.get_all_claims(),
            patch_history=history,
            metrics=metrics
        )
