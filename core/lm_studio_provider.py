import requests
from typing import Generator
from .base_provider import LLMProvider

class LMStudioProvider(LLMProvider):
    def __init__(self, model: str = "local-model", base_url: str = "http://localhost:1234/v1"):
        self.model = model
        # Note: LM Studio's /v1 endpoint is OpenAI compatible
        self.base_url = base_url

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        # Use the OpenAI-compatible completions or chat/completions endpoints.
        # LM Studio usually maps /v1/chat/completions correctly.
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        
        # OpenAI-style streaming response
        for line in response.text.splitlines():
            if line.strip() and line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    yield delta.get("content", "")
                except json.JSONDecodeError:
                    pass

    def generate(self, prompt: str, **kwargs) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

import json
