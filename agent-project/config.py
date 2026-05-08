"""Project-wide settings."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Settings:
    model: str = "qwen2.5-coder"
    max_steps: int = 6
    tool_registry: Dict[str, Any] = field(default_factory=dict)
    ollama_host: str = "http://127.0.0.1:11434"
    mode: str = "plan"  # "plan" | "react"
    # If enabled, Agent will use Router to short-circuit into specialized pipelines
    # (e.g. quantitative extract+compute) before falling back to the LLM planner loop.
    enable_router: bool = True
    # Router strategy: "rules" (fast), "llm" (flexible), "hybrid" (rules -> llm).
    router_mode: str = "hybrid"
    # Only accept LLM routing when confidence >= threshold; otherwise fall back to planner loop.
    router_confidence_threshold: float = 0.65

    # Memory
    enable_facts_store: bool = True

    # Retrieval decision policy (Chroma distance; smaller is better).
    retrieval_accept_distance: float = 0.40
    retrieval_soft_distance: float = 0.50
    retrieval_gap_distance: float = 0.02

    # Retrieval query rewrite
    enable_query_rewrite: bool = True
    query_rewrite_max_chars: int = 120


# Runtime singleton; tools register separately
settings = Settings()
