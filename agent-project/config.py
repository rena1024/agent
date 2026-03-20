"""Project-wide settings."""

import ollama
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Settings:
    model: str = "qwen2.5-coder"
    max_steps: int = 6
    tool_registry: Dict[str, Any] = field(default_factory=dict)
    ollama_host: str = "http://127.0.0.1:11434"


# Runtime singleton; tools register separately
settings = Settings()
