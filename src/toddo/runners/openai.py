"""OpenAI adapter (closed-source option)."""
from __future__ import annotations

from typing import Optional

from toddo.runners.base import GenConfig


class OpenAIClient:
    def __init__(self, model_id: str = "gpt-4o", *, max_tokens: int = 1024):
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'toddo[closed]' (needs `openai`).") from e
        self.client = openai.OpenAI()
        self.model_id = model_id
        self.max_tokens = max_tokens

    def _msgs(self, prompt: str, system: str) -> list[dict]:
        m = [{"role": "system", "content": system}] if system else []
        return m + [{"role": "user", "content": prompt}]

    def generate(self, prompt: str, *, system: str = "", cfg: Optional[GenConfig] = None) -> str:
        cfg = cfg or GenConfig()
        r = self.client.chat.completions.create(model=self.model_id, messages=self._msgs(prompt, system),
                                                 max_tokens=cfg.max_tokens, temperature=cfg.temperature)
        return r.choices[0].message.content or ""

    def sample(self, prompt: str, n: int, *, system: str = "", cfg: Optional[GenConfig] = None) -> list[str]:
        cfg = cfg or GenConfig()
        r = self.client.chat.completions.create(model=self.model_id, messages=self._msgs(prompt, system),
                                                 max_tokens=cfg.max_tokens, temperature=cfg.temperature, n=n)
        return [c.message.content or "" for c in r.choices]
