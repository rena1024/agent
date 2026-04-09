"""Parsers for LLM outputs using pydantic validation."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, ConfigDict
import json
from colorama import Fore


class PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["tool", "final"]
    tool: Optional[str] = None
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[str] = None
    thoughts: Optional[str] = None


def _from_tool_call(raw: dict) -> PlanModel:
    # map function calling to plan model
    calls = raw.get("tool_calls") or []
    if not calls:
        raise ValueError("No tool_calls found")
    first = calls[0].get("function", {})
    name = first.get("name")
    args_raw = first.get("arguments", "{}")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    except Exception as exc:  # noqa: BLE001
        # Surface as value error so caller can catch
        raise ValueError(f"tool_input parse error: {exc}") from exc
    return PlanModel(
        action="tool", tool=name, tool_input=args, thoughts="via tool_call"
    )


def parse_plan(raw: str | dict) -> PlanModel:
    obj = json.loads(raw) if isinstance(raw, str) else raw
    print(Fore.MAGENTA + "planner.parse: " + str(obj) + Fore.RESET)
    # function calling path
    if isinstance(obj, dict) and obj.get("tool_calls"):
        return _from_tool_call(obj)

    try:
        return PlanModel.model_validate(obj)
    except Exception:
        # treat as final plain text
        return PlanModel(action="final", output=obj)
