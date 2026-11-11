from __future__ import annotations
from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field
import datetime

class PatchOperation(BaseModel):
    """Structured patch emitted by an LLM model."""
    section_id: str = Field(..., description="ID or title of the section to target (e.g. 'section_1' or 'overview')")
    operation: Literal["update", "append", "delete", "replace_all"] = Field(
        default="update", 
        description="The editing operation to perform on the target section."
    )
    content: str = Field(..., description="The content to write or append.")
    reason: str = Field(default="", description="Explanation of why this change improves the draft.")
    model: Optional[str] = Field(default=None, description="Model identifier proposing this edit.")
    confidence: Optional[float] = Field(default=0.8, description="Estimated confidence of model in this patch [0.0-1.0].")

class Section(BaseModel):
    id: str
    title: str
    content: str
    last_editor: str
    last_version: int

class DraftState(BaseModel):
    session_id: str
    version: int = 0
    full_text: str = ""
    sections: Dict[str, Section] = Field(default_factory=dict)
    last_editor: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    active_model: Optional[str] = None

class Claim(BaseModel):
    claim_id: str
    statement: str
    section_id: Optional[str] = None
    source_model: str
    verification_status: Literal["unverified", "verified", "refuted", "uncertain"] = "unverified"
    confidence: float = 0.5
    evidence_notes: Optional[str] = None
    created_at_version: int = 1

class QueryAnalysis(BaseModel):
    query: str
    complexity_score: float = Field(..., description="Complexity C in [0.0, 1.0]")
    domain: Literal["factual", "reasoning", "code", "creative", "general"] = "general"
    estimated_rounds: int = 2
    recommended_primary_model: str
    flags: List[str] = Field(default_factory=list)

class ProfilerMetrics(BaseModel):
    duration_sec: float = 0.0
    peak_ram_mb: float = 0.0
    delta_ram_mb: float = 0.0
    cpu_utilization_pct: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_energy_joules: float = 0.0

class OrchestrationResult(BaseModel):
    session_id: str
    query: str
    final_answer: str
    final_version: int
    rounds_executed: int
    terminated_reason: str
    complexity_score: float
    models_used: List[str]
    claims: List[Claim] = Field(default_factory=list)
    patch_history: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: ProfilerMetrics = Field(default_factory=ProfilerMetrics)
