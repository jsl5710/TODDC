"""Prompt loading + rendering. Templates live in the top-level prompts/ dir."""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load(name: str) -> str:
    return (_DIR / name).read_text(encoding="utf-8")


def render_paraphrase_prompt(text: str, n: int = 2) -> str:
    return load("apply_paraphrase.txt").replace("{{n}}", str(n)).replace("{{utterance}}", text)


def render_judge_prompt(change_from: str, change_to: str, dimension: str, gold_label: str) -> str:
    return (load("judge_validate.txt")
            .replace("{{change_from}}", change_from).replace("{{change_to}}", change_to)
            .replace("{{dimension}}", str(dimension)).replace("{{gold_label}}", gold_label))
