"""Build model clients by role from configs/models.yaml (falls back to EchoClient).

Also builds multi-model pools for splitting the generation (and judge) work
evenly across several models on a server run — see docs/generation.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from toddc.runners.base import EchoClient, LLMClient

_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "models" / "models.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_client(spec: dict[str, Any]) -> LLMClient:
    adapter = spec.get("adapter", "")
    model_id = spec.get("model_id", "")
    if adapter.endswith("claude:ClaudeClient"):
        from toddc.runners.claude import ClaudeClient
        return ClaudeClient(model_id or "claude-opus-5")
    if adapter.endswith("openai:OpenAIClient"):
        from toddc.runners.openai import OpenAIClient
        return OpenAIClient(model_id or "gpt-4o")
    if "open_source" in adapter:
        from toddc.runners.open_source import OllamaClient, VLLMClient
        endpoint = spec.get("endpoint", "http://localhost:8000/v1")
        ctor = OllamaClient if "Ollama" in adapter else VLLMClient
        return ctor(model_id, endpoint=endpoint)
    raise ValueError(f"Unknown adapter: {adapter!r}")


def client_for_role(role: str, config_path: Optional[Path] = None) -> LLMClient:
    path = config_path or _CONFIG
    if not path.exists():
        return EchoClient()
    try:
        cfg = _load_yaml(path)
        group = cfg["roles"][role]
        section, _, key = group.partition(".")
        specs = cfg.get(section, {})
        return build_client(specs[key] if key else next(iter(specs.values())))
    except Exception:
        return EchoClient()


def _build_pool(config_path: Optional[Path], key: str, split_section: str):
    from toddc.runners.pool import ModelPool
    path = config_path or _CONFIG
    if not path.exists():
        return None
    cfg = _load_yaml(path)
    specs = cfg.get(key)
    if not specs:
        return None
    clients = [build_client(s) for s in specs]
    s = cfg.get(split_section, {}) or cfg.get("generation", {}) or {}
    return ModelPool(clients, strategy=s.get("split", "round_robin"),
                     weights=s.get("weights"), seed=int(s.get("seed", 0)))


def build_generator_pool(config_path: Optional[Path] = None):
    """Split generation (Pass-3 paraphrase) evenly across the config's `generators:`."""
    return _build_pool(config_path, "generators", "generation")


def build_judge_pool(config_path: Optional[Path] = None):
    """Split the judge (Pass-4 validation) evenly across the config's `judges:`."""
    return _build_pool(config_path, "judges", "judging")
