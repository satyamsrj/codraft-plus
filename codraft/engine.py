from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
import httpx

class BaseEngine(ABC):
    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        keep_alive: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        pass


class OllamaEngine(BaseEngine):
    def __init__(self, base_url: str = "http://localhost:11434", timeout_sec: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_sec

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        keep_alive: Optional[str] = "5m",
        temperature: float = 0.2
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "")
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"SOURCE_UNAVAILABLE: Unable to connect to Ollama daemon at {self.base_url}. "
                "Ensure Ollama is running and the model is pulled."
            ) from exc


class OpenAIEngine(BaseEngine):
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "local", timeout_sec: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout_sec

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        keep_alive: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"SOURCE_UNAVAILABLE: Unable to connect to OpenAI-compatible endpoint at {self.base_url}."
            ) from exc


class HuggingFaceEngine(BaseEngine):
    def __init__(self, device: Optional[str] = None):
        self.device = device
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}

    @staticmethod
    def _prepare_remote_code_environment():
        """Ensure tokenmonster and tempfile are available in sys.modules for remote-code tokenizers (Stentor2)."""
        import sys
        import tempfile
        try:
            import tokenmonster
            sys.modules["tokenmonster"] = tokenmonster
        except ImportError:
            pass

    def _load_model(self, model_id: str):
        if model_id in self._models:
            return self._models[model_id], self._tokenizers[model_id]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._prepare_remote_code_environment()

        target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            kwargs = {"trust_remote_code": True}
            if target_device != "cpu":
                kwargs["torch_dtype"] = torch.float16

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **kwargs
            ).to(target_device)
            model.eval()

            self._models[model_id] = model
            self._tokenizers[model_id] = tokenizer
            return model, tokenizer
        except Exception as exc:
            raise ConnectionError(
                f"SOURCE_UNAVAILABLE: Failed to load Hugging Face model '{model_id}': {exc}"
            ) from exc

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        keep_alive: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        import torch

        hf_model, tokenizer = self._load_model(model)
        target_device = next(hf_model.parameters()).device

        prompt_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "").strip()
            if role == "system":
                prompt_parts.append(f"Instruction:\n{content}")
            elif role == "user":
                prompt_parts.append(f"Input:\n{content}")
            else:
                prompt_parts.append(f"Response:\n{content}")
        prompt_parts.append("Response:\n")
        prompt = "\n\n".join(prompt_parts)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(target_device)
        with torch.no_grad():
            output_tokens = hf_model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=max(0.01, temperature),
                do_sample=(temperature > 0.05),
                repetition_penalty=1.2,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        gen_tokens = output_tokens[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    def compute_log_likelihood(self, model: str, context: str, continuation: str) -> float:
        import torch
        import torch.nn.functional as F

        hf_model, tokenizer = self._load_model(model)
        target_device = next(hf_model.parameters()).device

        full_text = context + continuation
        ctx_tokens = tokenizer.encode(context, add_special_tokens=False)
        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

        prefix_len = 0
        while prefix_len < len(ctx_tokens) and prefix_len < len(full_tokens) and ctx_tokens[prefix_len] == full_tokens[prefix_len]:
            prefix_len += 1

        if prefix_len >= len(full_tokens):
            return -9999.0

        input_ids = torch.tensor([full_tokens], dtype=torch.long, device=target_device)
        with torch.no_grad():
            outputs = hf_model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

        shift_logits = logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)

        start_idx = max(0, prefix_len - 1)
        end_idx = len(full_tokens) - 1

        cont_log_probs = log_probs[start_idx:end_idx]
        cont_labels = shift_labels[start_idx:end_idx]

        token_log_probs = cont_log_probs.gather(1, cont_labels.unsqueeze(1)).squeeze(1)
        return float(token_log_probs.sum().item())

    def compute_text_loss_and_bpb(self, model: str, text: str, max_ctx: int = 1024) -> Tuple[float, float, float]:
        import math
        import torch

        hf_model, tokenizer = self._load_model(model)
        target_device = next(hf_model.parameters()).device

        tokens = tokenizer.encode(text, add_special_tokens=False)[:max_ctx]
        if len(tokens) < 2:
            return (0.0, 0.0, 0.0)

        num_bytes = len(text.encode("utf-8"))
        input_ids = torch.tensor([tokens], dtype=torch.long, device=target_device)
        with torch.no_grad():
            outputs = hf_model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss_nats = float(loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item())

        ppl = float(math.exp(min(loss_nats, 100.0)))
        bpb = float((loss_nats / math.log(2.0)) * (len(tokens) / max(1, num_bytes)))
        return (round(loss_nats, 4), round(ppl, 2), round(bpb, 4))
