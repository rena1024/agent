"""Parsers for LLM outputs using pydantic validation."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, ValidationError


class PlanModel(BaseModel):
    action: Literal["tool", "final"]
    tool: Optional[str] = None
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[str] = None
    thoughts: Optional[str] = None


def parse_plan(raw: str | dict) -> PlanModel:
    if isinstance(raw, str):
        import json

        raw = json.loads(raw)
    return PlanModel.model_validate(raw)
