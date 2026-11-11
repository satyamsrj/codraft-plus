from abc import ABC, abstractmethod
from typing import List, Dict, Optional
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
