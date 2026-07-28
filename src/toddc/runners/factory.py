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


# --- Config validation (dry-run) --------------------------------------------

def _ping_openai_endpoint(endpoint: str, model_id: str, timeout: float = 5.0) -> tuple[bool, str]:
    """GET {endpoint}/models on an OpenAI-compatible server (vLLM/Ollama/TGI).

    Reachable + serving model_id -> (True, ...); reachable but model absent ->
    (True, warning); no response -> (False, reason). Stdlib-only, no extra deps."""
    import json
    import urllib.error
    import urllib.request
    url = endpoint.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (trusted config URL)
            served = [m.get("id") for m in json.loads(r.read().decode("utf-8")).get("data", [])]
    except Exception as e:  # pragma: no cover - network-dependent
        return False, f"UNREACHABLE at {url} ({e.__class__.__name__}: {e})"
    if model_id in served:
        return True, f"reachable, serving {model_id}"
    return True, f"reachable, but {model_id!r} not in served models {served}"


def _role_spec(cfg: dict[str, Any], role: str) -> Optional[dict[str, Any]]:
    try:
        group = cfg["roles"][role]
        section, _, key = group.partition(".")
        specs = cfg.get(section, {})
        return specs[key] if key else next(iter(specs.values()))
    except Exception:
        return None


def check_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate one client spec: ping open endpoints, check keys for closed APIs."""
    import os
    adapter = spec.get("adapter", "")
    model_id = spec.get("model_id", "")
    if "open_source" in adapter:
        endpoint = spec.get("endpoint", "")
        ok, detail = _ping_openai_endpoint(endpoint, model_id) if endpoint else (False, "no endpoint set")
        return {"model_id": model_id, "kind": "open", "target": endpoint, "ok": ok, "detail": detail}
    env = spec.get("api_key_env", "")
    ok = bool(env and os.environ.get(env))
    detail = f"env {env} is set" if ok else f"env {env or '<none>'} NOT set"
    return {"model_id": model_id, "kind": "closed", "target": env, "ok": ok, "detail": detail}


def check_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Validate every configured generator/judge without generating. Returns
    {"config": path, "generators": [...], "judges": [...]} of check_spec results."""
    path = config_path or _CONFIG
    if not path.exists():
        return {"config": str(path), "exists": False, "generators": [], "judges": []}
    cfg = _load_yaml(path)
    gens = cfg.get("generators") or ([s] if (s := _role_spec(cfg, "generator")) else [])
    juds = cfg.get("judges") or ([s] if (s := _role_spec(cfg, "judge")) else [])
    return {"config": str(path), "exists": True,
            "generators": [check_spec(s) for s in gens],
            "judges": [check_spec(s) for s in juds]}


def planned_split(units: int, key: str, split_section: str,
                  config_path: Optional[Path] = None) -> Optional[dict[str, int]]:
    """Simulate how `units` generation units would divide across the configured
    `key` (generators/judges) — WITHOUT instantiating real clients (no SDK/network
    needed). Returns {model_id: count} or None if that list isn't configured."""
    from toddc.runners.pool import ModelPool
    path = config_path or _CONFIG
    if not path.exists():
        return None
    cfg = _load_yaml(path)
    specs = cfg.get(key)
    if not specs:
        return None
    stubs = [type("_Stub", (), {"model_id": s.get("model_id", f"model-{i}")})()
             for i, s in enumerate(specs)]
    sec = cfg.get(split_section, {}) or cfg.get("generation", {}) or {}
    pool = ModelPool(stubs, strategy=sec.get("split", "round_robin"),
                     weights=sec.get("weights"), seed=int(sec.get("seed", 0)))
    for _ in range(units):
        pool.next()
    return pool.summary()
