from pydantic import BaseModel, Field
from codraft.model_sets import get_model_set

class Settings(BaseModel):
    active_track: str = Field(default="pythia")
    engine_type: str = Field(default="huggingface")
    
    model_a: str = Field(default="")
    model_b: str = Field(default="")
    model_large: str = Field(default="")
    
    ollama_base_url: str = Field(default="http://localhost:11434")
    
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

    def model_post_init(self, __context):
        if not self.model_a or not self.model_b:
            mset = get_model_set(self.active_track)
            if not self.model_a:
                self.model_a = mset.small_a.hf_id
            if not self.model_b:
                self.model_b = mset.small_b.hf_id
            if not self.model_large:
                self.model_large = mset.large.hf_id

settings = Settings()
