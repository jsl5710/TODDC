"""Provider-agnostic LLM interface (shared shape with TODUQ)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class GenConfig:
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    seed: Optional[int] = None


@runtime_checkable
class LLMClient(Protocol):
    model_id: str

    def generate(self, prompt: str, *, system: str = "", cfg: Optional[GenConfig] = None) -> str: ...
    def sample(self, prompt: str, n: int, *, system: str = "", cfg: Optional[GenConfig] = None) -> list[str]: ...


class EchoClient:
    """Deterministic offline stub — lets the pipeline + tests run with no network."""
    model_id = "echo-stub"

    def generate(self, prompt: str, *, system: str = "", cfg: Optional[GenConfig] = None) -> str:
        return prompt.strip().splitlines()[-1] if prompt.strip() else ""

    def sample(self, prompt: str, n: int, *, system: str = "", cfg: Optional[GenConfig] = None) -> list[str]:
        return [self.generate(prompt, system=system, cfg=cfg) for _ in range(n)]
