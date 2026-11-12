import os
import sys
import json
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

MODELS_TO_VERIFY = [
    {
        "set": "Set 1 (Pythia / GPT-NeoX)",
        "role": "Small A",
        "id": "EleutherAI/pythia-14m",
        "type": "causal",
        "objective": "100% Autoregressive (The Pile)"
    },
    {
        "set": "Set 1 (Pythia / GPT-NeoX)",
        "role": "Small B",
        "id": "EleutherAI/pythia-31m",
        "type": "causal",
        "objective": "100% Autoregressive (The Pile)"
    },
    {
        "set": "Set 1 (Pythia / GPT-NeoX)",
        "role": "Large Baseline",
        "id": "EleutherAI/pythia-70m",
        "type": "causal",
        "objective": "100% Autoregressive (The Pile)"
    },
    {
        "set": "Set 2 (BabyLM / GPT-BERT)",
        "role": "Small A",
        "id": "BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus",
        "type": "gpt-bert",
        "objective": "93.75% Causal / 6.25% Masked (BabyLM 10M)"
    },
    {
        "set": "Set 2 (BabyLM / GPT-BERT)",
        "role": "Small B",
        "id": "BabyLM-community/babylm-baseline-10m-gpt-bert-mixed",
        "type": "gpt-bert",
        "objective": "50% Causal / 50% Masked (BabyLM 10M)"
    },
    {
        "set": "Set 2 (BabyLM / GPT-BERT)",
        "role": "Large Baseline",
        "id": "BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus",
        "type": "gpt-bert",
        "objective": "93.75% Causal / 6.25% Masked (BabyLM 100M)"
    },
    {
        "set": "Set 3 (Stentor2 / Llama)",
        "role": "Small A",
        "id": "StentorLabs/Stentor2-12M",
        "type": "llama",
        "objective": "100% Autoregressive (FineWeb-HQ / StenCore)"
    },
    {
        "set": "Set 3 (Stentor2 / Llama)",
        "role": "Small B",
        "id": "StentorLabs/Stentor2-30M",
        "type": "llama",
        "objective": "100% Autoregressive (FineWeb-HQ / StenCore)"
    },
]

def audit():
    report = []
    print("=" * 80)
    print("CoDraft+ Comprehensive Offline Checkpoint Verification Audit")
    print("=" * 80)

    for item in MODELS_TO_VERIFY:
        m_id = item["id"]
        print(f"\n[Verifying {item['set']} - {item['role']}: {m_id}]")
        
        entry = {
            "set": item["set"],
            "role": item["role"],
            "model_id": m_id,
            "pretraining_objective": item["objective"],
            "offline_loadable": False,
            "causal_generation": False,
            "masked_inference": False
        }

        try:
            cfg = AutoConfig.from_pretrained(m_id, local_files_only=True, trust_remote_code=True)
            tok = AutoTokenizer.from_pretrained(m_id, local_files_only=True, trust_remote_code=True)
            
            entry["architecture"] = cfg.architectures[0] if cfg.architectures else cfg.model_type
            entry["num_layers"] = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
            entry["hidden_dim"] = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
            entry["num_heads"] = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
            entry["max_context"] = getattr(cfg, "max_position_embeddings", getattr(cfg, "max_sequence_length", getattr(cfg, "n_positions", None)))
            entry["vocab_size"] = len(tok) if hasattr(tok, "__len__") else getattr(cfg, "vocab_size", None)
            entry["offline_loadable"] = True

            # Load Causal LM
            causal_mod = AutoModelForCausalLM.from_pretrained(m_id, local_files_only=True, trust_remote_code=True)
            params = sum(p.numel() for p in causal_mod.parameters())
            trainable_params = sum(p.numel() for p in causal_mod.parameters() if p.requires_grad)
            entry["param_count_exact"] = params
            entry["param_count_millions"] = round(params / 1e6, 2)
            entry["trainable_params"] = trainable_params

            # Causal Generation test
            test_prompt = "The solar system consists of"
            inputs = tok(test_prompt, return_tensors="pt")
            with torch.no_grad():
                gen_out = causal_mod.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=False)
                gen_text = tok.decode(gen_out[0], skip_special_tokens=True).strip()
            entry["causal_generation"] = True
            entry["sample_causal_output"] = gen_text
            print(f"  -> Causal Generation: SUCCESS | \"{gen_text}\"")

            # Masked Inference test (if supported)
            if item["type"] == "gpt-bert":
                try:
                    masked_mod = AutoModelForMaskedLM.from_pretrained(m_id, local_files_only=True, trust_remote_code=True)
                    mask_token = tok.mask_token or "<mask>"
                    masked_prompt = f"The capital of France is {mask_token}."
                    m_inputs = tok(masked_prompt, return_tensors="pt")
                    with torch.no_grad():
                        m_out = masked_mod(**m_inputs)
                        m_logits = m_out[0] if isinstance(m_out, tuple) else getattr(m_out, "logits", m_out[0])
                        mask_id = tok.mask_token_id
                        mask_pos = (m_inputs.input_ids == mask_id).nonzero(as_tuple=True)[1]
                        top_ids = torch.topk(m_logits[0, mask_pos], 3, dim=-1).indices[0]
                        top_tokens = [tok.decode([idx]).strip() for idx in top_ids]
                    entry["masked_inference"] = True
                    entry["sample_masked_predictions"] = top_tokens
                    print(f"  -> Masked Inference: SUCCESS | Top 3: {top_tokens}")
                except Exception as m_err:
                    entry["masked_inference_error"] = str(m_err)
                    print(f"  -> Masked Inference: FAILED ({m_err})")

            print(f"  -> Params: {params:,} ({entry['param_count_millions']}M) | Layers: {entry['num_layers']} | Hidden: {entry['hidden_dim']}")
        except Exception as e:
            entry["error"] = str(e)
            print(f"  -> ERROR: {e}")

        report.append(entry)

    out_json = "eval/checkpoint_verification_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nAudit complete. Full JSON report saved to {out_json}")

if __name__ == "__main__":
    audit()

