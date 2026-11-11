from pydantic import BaseModel, Field

class Settings(BaseModel):
    model_a: str = Field(default="qwen2.5:3b")
    model_b: str = Field(default="gemma3:1b")
    ollama_base_url: str = Field(default="http://localhost:11434")
    engine_type: str = Field(default="ollama")
    
    storage_type: str = Field(default="sqlite")
    db_path: str = Field(default="codraft_workspace.db")
    redis_url: str = Field(default="redis://localhost:6379/0")
    history_log_path: str = Field(default="draft_history.jsonl")

    max_rounds: int = Field(default=4)
    complexity_threshold: float = Field(default=0.45)
    stagnation_diff_threshold: float = Field(default=0.04)
    stagnation_patience: int = Field(default=2)
    
    weight_complexity: float = 0.35
    weight_uncertainty: float = 0.25
    weight_disagreement: float = 0.25
    weight_verification: float = 0.15

    low_memory_threshold_gb: float = 8.0
    model_keep_alive: str = "5m"

settings = Settings()
