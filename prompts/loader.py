"""Utility to load agent system prompts from the prompts/ directory."""
import os

_PROMPTS_DIR = os.path.dirname(__file__)


def load_prompt(name: str) -> str:
    """Load a prompt by name (without extension) from the prompts/ directory."""
    path = os.path.join(_PROMPTS_DIR, f"{name}.md")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()
