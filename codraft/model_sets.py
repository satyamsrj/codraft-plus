from typing import Dict, Any
from pydantic import BaseModel, Field

class ModelMetadata(BaseModel):
    name: str
    hf_id: str
    params_millions: float
    architecture: str
    training_data: str
    training_tokens: str
    role: str

class ModelSet(BaseModel):
    set_id: str
    family_name: str
    description: str
    small_a: ModelMetadata
    small_b: ModelMetadata
    large: ModelMetadata
    combined_small_params: float
    large_params: float
    size_ratio: float = 2.0

MODEL_SETS: Dict[str, ModelSet] = {
    "pythia": ModelSet(
        set_id="set_1_pythia",
        family_name="GPT-NeoX",
        description="Pythia suite trained on The Pile in identical data ordering.",
        small_a=ModelMetadata(
            name="Pythia-14M",
            hf_id="EleutherAI/pythia-14m",
            params_millions=14.07,
            architecture="GPTNeoXForCausalLM",
            training_data="The Pile (300B tokens, deduplicated)",
            training_tokens="~300B",
            role="Small A (14M base drafting)"
        ),
        small_b=ModelMetadata(
            name="Pythia-31M",
            hf_id="EleutherAI/pythia-31m",
            params_millions=30.49,
            architecture="GPTNeoXForCausalLM",
            training_data="The Pile (300B tokens, deduplicated)",
            training_tokens="~300B",
            role="Small B (31M refinement)"
        ),
        large=ModelMetadata(
            name="Pythia-Custom-89M",
            hf_id="EleutherAI/pythia-70m",  # Public baseline reference; target custom 89.12M
            params_millions=89.12,
            architecture="GPTNeoXForCausalLM",
            training_data="The Pile (300B tokens, deduplicated)",
            training_tokens="~300B",
            role="Large Reference (2x combined size)"
        ),
        combined_small_params=44.56,
        large_params=89.12,
        size_ratio=2.00
    ),
    "babylm": ModelSet(
        set_id="set_2_babylm",
        family_name="GPT-BERT",
        description="BabyLM 10M Strict-Small controlled experiment with divergent causal vs mixed training objectives.",
        small_a=ModelMetadata(
            name="BabyLM-GPT-BERT-Causal",
            hf_id="BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus",
            params_millions=31.0,
            architecture="GPT-BERT (93.75% Causal / 6.25% Masked)",
            training_data="BabyLM 10M Strict-Small",
            training_tokens="10M",
            role="Small A (Causal reasoning)"
        ),
        small_b=ModelMetadata(
            name="BabyLM-GPT-BERT-Mixed",
            hf_id="BabyLM-community/babylm-baseline-10m-gpt-bert-mixed",
            params_millions=31.0,
            architecture="GPT-BERT (50% Causal / 50% Masked)",
            training_data="BabyLM 10M Strict-Small",
            training_tokens="10M",
            role="Small B (Mixed bi-directional context)"
        ),
        large=ModelMetadata(
            name="BabyLM-GPT-BERT-124M",
            hf_id="BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus",  # Reference configuration
            params_millions=124.0,
            architecture="GPT-BERT",
            training_data="BabyLM 10M Strict-Small",
            training_tokens="10M",
            role="Large Target (2x combined size)"
        ),
        combined_small_params=62.0,
        large_params=124.0,
        size_ratio=2.00
    ),
    "stentor2": ModelSet(
        set_id="set_3_stentor2",
        family_name="Llama",
        description="Stentor2 family using LlamaForCausalLM trained on FineWeb-HQ and StenCore.",
        small_a=ModelMetadata(
            name="Stentor2-12M",
            hf_id="StentorLabs/Stentor2-12M",
            params_millions=12.3,
            architecture="LlamaForCausalLM",
            training_data="FineWeb-HQ and StenCore",
            training_tokens="~480M",
            role="Small A (Compact drafting)"
        ),
        small_b=ModelMetadata(
            name="Stentor2-30M",
            hf_id="StentorLabs/Stentor2-30M",
            params_millions=30.4,
            architecture="LlamaForCausalLM",
            training_data="FineWeb-HQ and StenCore",
            training_tokens="~800M",
            role="Small B (Expanded representation)"
        ),
        large=ModelMetadata(
            name="Stentor2-Custom-85M",
            hf_id="StentorLabs/Stentor2-30M",  # Base configuration for scaled architecture
            params_millions=85.4,
            architecture="LlamaForCausalLM",
            training_data="FineWeb-HQ and StenCore",
            training_tokens="Matched budget",
            role="Large Target (2x combined size)"
        ),
        combined_small_params=42.7,
        large_params=85.4,
        size_ratio=2.00
    )
}

def get_model_set(track_name: str) -> ModelSet:
    key = track_name.lower().strip()
    if key not in MODEL_SETS:
        raise ValueError(f"Unknown track '{track_name}'. Choose from: {list(MODEL_SETS.keys())}")
    return MODEL_SETS[key]

