import json
import asyncio
from typing import List, Dict, Optional
from codraft.types import PatchOperation, DraftState
from codraft.patcher import ResilientPatcher
from codraft.analyzer import QueryAnalyzer
from codraft.storage import SQLiteStorage, InMemoryStorage
from codraft.evidence import EvidenceLedger
from codraft.controller import AdaptiveController
from codraft.orchestrator import CoDraftOrchestrator
from codraft.engine import BaseEngine

class TestProtocolEngine(BaseEngine):
    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        keep_alive: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        return json.dumps({
            "section_id": "overview",
            "operation": "replace_all",
            "content": "Verified system response for integration testing.",
            "reason": "Protocol validation test",
            "confidence": 0.95
        })

def test_query_analyzer():
    q1 = "What is the capital of France?"
    a1 = QueryAnalyzer.analyze(q1)
    assert a1.domain == "factual"
    assert a1.complexity_score < 0.45
    assert a1.recommended_primary_model == "gemma3:1b"

    q2 = "Explain the trade-offs of Kalman filtering vs optical flow odometry in GPS-denied drone navigation and derive the error propagation."
    a2 = QueryAnalyzer.analyze(q2)
    assert a2.domain == "reasoning"
    assert a2.complexity_score >= 0.45
    assert a2.recommended_primary_model == "qwen2.5:3b"

def test_resilient_patcher_parsing():
    raw1 = '{"section_id": "intro", "operation": "update", "content": "Initial text", "reason": "Init", "confidence": 0.9}'
    p1 = ResilientPatcher.parse_patch(raw1)
    assert p1.section_id == "intro"
    assert p1.content == "Initial text"
    assert p1.confidence == 0.9

    raw2 = '```json\n{"section_id": "sec2", "operation": "append", "content": "Appended text", "reason": "Exp"}\n```'
    p2 = ResilientPatcher.parse_patch(raw2)
    assert p2.section_id == "sec2"
    assert p2.operation == "append"
    assert p2.content == "Appended text"

    raw3 = "Direct unformatted string."
    p3 = ResilientPatcher.parse_patch(raw3, fallback_model="test_model")
    assert p3.operation in ["replace_all", "append"]
    assert p3.content == raw3
    assert p3.model == "test_model"

def test_section_patch_application():
    state = DraftState(session_id="test", version=0, full_text="", sections={})
    patch1 = PatchOperation(
        section_id="overview",
        operation="replace_all",
        content="Primary text content.",
        model="model_a"
    )
    state, diff1 = ResilientPatcher.apply_patch(state, patch1)
    assert diff1 > 0.0
    assert "Primary text content." in state.full_text

    patch2 = PatchOperation(
        section_id="details",
        operation="append",
        content="Secondary detail content.",
        model="model_b"
    )
    state, diff2 = ResilientPatcher.apply_patch(state, patch2)
    assert "Secondary detail content." in state.full_text

def test_storage_backends(tmp_path):
    mem_store = InMemoryStorage()
    mem_state = mem_store.init_session("s1", "initial")
    assert mem_state.version == 0
    p = PatchOperation(section_id="sec1", operation="update", content="updated", model="mod1")
    v1 = mem_store.save_state(mem_state, p)
    assert v1 == 1
    assert len(mem_store.get_history("s1")) == 1

    db_file = str(tmp_path / "test_workspace.db")
    sql_store = SQLiteStorage(db_path=db_file)
    sql_state = sql_store.init_session("s2", "initial")
    assert sql_state.version == 0
    v2 = sql_store.save_state(sql_state, p)
    assert v2 == 1
    retrieved = sql_store.get_state("s2")
    assert retrieved is not None
    assert retrieved.version == 1
    assert len(sql_store.get_history("s2")) == 1

def test_evidence_ledger():
    ledger = EvidenceLedger()
    text = "The speed of light is 299792458 meters per second. Photosynthesis converts light into chemical energy."
    claims = ledger.extract_claims(text, source_model="model_a", version=1)
    assert len(claims) >= 1
    assert ledger.compute_verification_pressure() > 0.0

    ledger.verify_claim(claims[0].claim_id, status="verified", confidence=0.99)
    assert claims[0].verification_status == "verified"

def test_adaptive_controller():
    controller = AdaptiveController(
        max_rounds=3,
        complexity_threshold=0.4,
        stagnation_diff_threshold=0.05,
        stagnation_patience=2
    )
    dummy_state = DraftState(session_id="t", version=1, full_text="Same text")
    dummy_ledger = EvidenceLedger()

    stop1, _, _ = controller.evaluate_stopping(1, 0.5, 0.02, dummy_state, dummy_ledger)
    assert not stop1

    stop2, reason2, _ = controller.evaluate_stopping(2, 0.5, 0.01, dummy_state, dummy_ledger)
    assert stop2
    assert "stagnation_detected" in reason2

def test_end_to_end_orchestration():
    async def _run():
        mem_store = InMemoryStorage()
        test_engine = TestProtocolEngine()
        orch = CoDraftOrchestrator(
            storage=mem_store,
            engine=test_engine,
            model_a="qwen2.5:3b",
            model_b="gemma3:1b"
        )

        query = "Explain optical flow in autonomous navigation."
        result = await orch.execute(query=query)

        assert result.session_id.startswith("sess_")
        assert result.rounds_executed >= 1
        assert len(result.final_answer) > 0
        assert result.metrics.input_tokens > 0
        assert result.metrics.output_tokens > 0
        assert len(result.patch_history) >= 1
    
    asyncio.run(_run())
