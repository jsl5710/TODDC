"""Anthropic (Claude) adapter. Omits sampling params (rejected by current models),
steers with output_config.effort, and handles stop_reason == 'refusal'."""
from __future__ import annotations

from typing import Optional

from toddc.runners.base import GenConfig

DEFAULT_MODEL = "claude-opus-5"


class ClaudeClient:
    def __init__(self, model_id: str = DEFAULT_MODEL, *, effort: str = "medium", max_tokens: int = 1024):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'toddc[closed]' (needs `anthropic`).") from e
        self.client = anthropic.Anthropic()
        self.model_id = model_id
        self.effort = effort
        self.max_tokens = max_tokens

    def _call(self, prompt: str, system: str, cfg: Optional[GenConfig]) -> str:
        kwargs = dict(model=self.model_id, max_tokens=(cfg.max_tokens if cfg else self.max_tokens),
                      messages=[{"role": "user", "content": prompt}],
                      output_config={"effort": self.effort})
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        if getattr(resp, "stop_reason", None) == "refusal":
            return ""
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def generate(self, prompt: str, *, system: str = "", cfg: Optional[GenConfig] = None) -> str:
        return self._call(prompt, system, cfg)

    def sample(self, prompt: str, n: int, *, system: str = "", cfg: Optional[GenConfig] = None) -> list[str]:
        return [self._call(prompt, system, cfg) for _ in range(n)]
