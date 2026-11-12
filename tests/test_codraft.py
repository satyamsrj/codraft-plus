import os
import sys
import json
import asyncio
from typing import List, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from codraft.types import PatchOperation, DraftState
from codraft.patcher import ResilientPatcher
from codraft.analyzer import QueryAnalyzer
from codraft.storage import SQLiteStorage, InMemoryStorage
from codraft.evidence import EvidenceLedger
from codraft.controller import AdaptiveController
from codraft.orchestrator import CoDraftOrchestrator
from codraft.engine import BaseEngine
from codraft.model_sets import get_model_set, MODEL_SETS

def test_model_sets_registry():
    assert "babylm" in MODEL_SETS
    assert "pythia" in MODEL_SETS
    assert "stentor2" in MODEL_SETS

    babylm = get_model_set("babylm")
    assert babylm.small_a.params_millions == 31.0
    assert babylm.small_b.params_millions == 31.0
    assert babylm.combined_small_params == 62.0
    assert babylm.size_ratio == 2.0

    pythia = get_model_set("pythia")
    assert pythia.small_a.params_millions == 14.07
    assert pythia.small_b.params_millions == 30.49
    assert pythia.size_ratio == 2.0

    stentor = get_model_set("stentor2")
    assert stentor.small_a.params_millions == 12.3
    assert stentor.small_b.params_millions == 30.4
    assert stentor.size_ratio == 2.0


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

def test_metrics_bpb_and_mcc_and_ci():
    from eval.metrics import (
        compute_bits_per_byte,
        compute_mcc,
        compute_confidence_interval_95,
        compute_lambada_last_word_match
    )
    # Test Bits-per-Byte (BPB)
    loss = 2.5
    num_tokens = 100
    utf8_bytes = 400
    bpb = compute_bits_per_byte(loss, num_tokens, utf8_bytes)
    # bpb = (2.5 * 100) / (400 * ln(2)) = 250 / 277.25887 ~ 0.9017
    assert 0.90 < bpb < 0.91

    # Test MCC
    mcc_perfect = compute_mcc(tp=10, tn=10, fp=0, fn=0)
    assert mcc_perfect == 1.0

    mcc_inverse = compute_mcc(tp=0, tn=0, fp=10, fn=10)
    assert mcc_inverse == -1.0

    mcc_balanced = compute_mcc(tp=5, tn=5, fp=5, fn=5)
    assert mcc_balanced == 0.0

    # Test 95% Wald Confidence Interval
    ci_low, ci_high = compute_confidence_interval_95(accuracy=80.0, sample_size=100)
    assert 70.0 < ci_low < 80.0
    assert 80.0 < ci_high < 90.0

    # Test LAMBADA last-word matcher
    assert compute_lambada_last_word_match("She sat down and began to play.", "play")
    assert compute_lambada_last_word_match("Prediction: play", "play")
    assert not compute_lambada_last_word_match("She sat down and cried.", "play")

def test_patcher_four_tiers_and_conflict_resilience():
    # Tier 1: Formatted JSON with section
    p1 = ResilientPatcher.parse_patch('{"section_id": "sec1", "operation": "replace_all", "content": "Alpha"}')
    assert p1.operation == "replace_all"
    assert p1.content == "Alpha"

    # Tier 2: Markdown fenced JSON
    p2 = ResilientPatcher.parse_patch('```json\n{"section_id": "sec2", "operation": "append", "content": "Beta"}\n```')
    assert p2.operation == "append"
    assert p2.content == "Beta"

    # Tier 3: Unified diff patch format
    diff_text = "--- a/doc.txt\n+++ b/doc.txt\n@@ -1,1 +1,1 @@\n-Alpha\n+Alpha Updated"
    p3 = ResilientPatcher.parse_patch(diff_text, fallback_model="reviewer")
    assert p3.operation in ["replace_all", "append"]
    assert len(p3.content) > 0

    # Tier 4: Unformatted plain text fallback
    raw_str = "Completely unstructured response without JSON markers."
    p4 = ResilientPatcher.parse_patch(raw_str, fallback_model="reviewer")
    assert p4.content == raw_str
    assert p4.operation == "append"

    # Test patch application & state lineage
    state0 = DraftState(session_id="test_lineage", version=0, full_text="", sections={})
    state1, diff1 = ResilientPatcher.apply_patch(state0, p1)
    assert diff1 > 0.0
    assert "Alpha" in state1.full_text

    state2, diff2 = ResilientPatcher.apply_patch(state1, p2)
    assert "Alpha" in state2.full_text and "Beta" in state2.full_text

def test_controller_max_rounds_and_entropy_halting():
    controller = AdaptiveController(
        max_rounds=2,
        complexity_threshold=0.4,
        stagnation_diff_threshold=0.05,
        stagnation_patience=1
    )
    state = DraftState(session_id="c_test", version=2, full_text="Stable final answer.")
    ledger = EvidenceLedger()

    # Exceeding max rounds must halt immediately
    stop, reason, _ = controller.evaluate_stopping(
        current_round=2,
        c_score=0.8,
        last_diff=0.5,
        state=state,
        ledger=ledger
    )
    assert stop
    assert "max_rounds_reached" in reason

def test_sqlite_storage_persistence_and_lineage(tmp_path):
    db_file = str(tmp_path / "lineage_test.db")
    storage = SQLiteStorage(db_path=db_file)
    init_state = storage.init_session("sess_persist", "Initial draft")

    # Add 3 revisions
    s = init_state
    for i in range(1, 4):
        p = PatchOperation(section_id="main", operation="append", content=f" Step {i}", model="test_agent")
        s, _ = ResilientPatcher.apply_patch(s, p)
        v = storage.save_state(s, p)
        assert v == i

    # Retrieve history and verify lineage
    history = storage.get_history("sess_persist")
    assert len(history) == 3
    assert history[0]["version"] == 1
    assert history[1]["version"] == 2
    assert history[2]["version"] == 3

    final_state = storage.get_state("sess_persist")
    assert final_state.version == 3
    assert "Step 1" in final_state.full_text
    assert "Step 2" in final_state.full_text
    assert "Step 3" in final_state.full_text

