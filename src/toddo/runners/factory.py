"""Build model clients by role from configs/models.yaml (falls back to EchoClient)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from toddo.runners.base import EchoClient, LLMClient

_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "models" / "models.yaml"


def build_client(spec: dict[str, Any]) -> LLMClient:
    adapter = spec.get("adapter", "")
    model_id = spec.get("model_id", "")
    if adapter.endswith("claude:ClaudeClient"):
        from toddo.runners.claude import ClaudeClient
        return ClaudeClient(model_id or "claude-opus-5")
    if adapter.endswith("openai:OpenAIClient"):
        from toddo.runners.openai import OpenAIClient
        return OpenAIClient(model_id or "gpt-4o")
    raise ValueError(f"Unknown adapter: {adapter!r}")


def client_for_role(role: str, config_path: Optional[Path] = None) -> LLMClient:
    path = config_path or _CONFIG
    if not path.exists():
        return EchoClient()
    try:
        import yaml
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        group = cfg["roles"][role]
        section, _, key = group.partition(".")
        specs = cfg.get(section, {})
        return build_client(specs[key] if key else next(iter(specs.values())))
    except Exception:
        return EchoClient()
