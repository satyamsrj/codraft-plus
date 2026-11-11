from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from codraft.orchestrator import CoDraftOrchestrator
from codraft.analyzer import QueryAnalyzer
from codraft.types import OrchestrationResult, QueryAnalysis
from codraft.config import settings

app = FastAPI(
    title="CoDraft+ API",
    description="Adaptive Shared-Draft Multi-LLM Orchestrator for Resource-Constrained Local Deployment",
    version="1.0.0"
)

orchestrator = CoDraftOrchestrator()

class AskRequest(BaseModel):
    query: str = Field(..., description="The user query or prompt to collaboratively answer.")
    session_id: Optional[str] = Field(default=None, description="Optional custom session identifier.")
    force_rounds: Optional[int] = Field(default=None, description="Force a fixed number of rounds (e.g. for ablation experiments).")
    enable_evidence: bool = Field(default=True, description="Enable automatic atomic claim extraction and verification ledger.")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": settings.engine_type,
        "storage": settings.storage_type,
        "models": {
            "model_a": settings.model_a,
            "model_b": settings.model_b
        }
    }

@app.post("/analyze", response_model=QueryAnalysis)
async def analyze_query(query: str = Query(..., description="Query to analyze")):
    """Analyze query complexity C, domain, and recommended initial model."""
    return QueryAnalyzer.analyze(query, model_a=settings.model_a, model_b=settings.model_b)

@app.post("/ask", response_model=OrchestrationResult)
async def ask_question(req: AskRequest):
    """Main collaborative inference endpoint."""
    try:
        result = await orchestrator.execute(
            query=req.query,
            session_id=req.session_id,
            force_rounds=req.force_rounds,
            enable_evidence=req.enable_evidence
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/draft/{session_id}")
async def get_draft_state(session_id: str):
    """Retrieve full workspace state and structured sections for an active session."""
    state = orchestrator.storage.get_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state

@app.get("/history/{session_id}")
async def get_patch_history(session_id: str):
    """Retrieve complete provenance audit trail of patch operations for a session."""
    history = orchestrator.storage.get_history(session_id)
    return {"session_id": session_id, "patch_count": len(history), "history": history}

