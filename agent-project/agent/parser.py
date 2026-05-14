"""Parsers for LLM outputs using pydantic validation."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict
import json
import re


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
    def strip_code_block(text: str) -> str:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        return m.group(1) if m else text

    if isinstance(raw, str):
        raw = strip_code_block(raw).strip()
        try:
            obj = json.loads(raw)
        except Exception:
            return PlanModel(action="final", output=raw)
    else:
        obj = raw

    # unwrap common outer container
    if isinstance(obj, dict) and "message" in obj:
        obj = obj.get("message") or {}

    # unwrap wrapper like {"action": "...", "output": "<json string>"} produced by some clients
    if (
        isinstance(obj, dict)
        and isinstance(obj.get("output"), str)
        and obj.get("tool_calls") is None
        and obj.get("content") is None
    ):
        output_str = strip_code_block(obj["output"]).strip()
        if output_str.startswith("{") and output_str.endswith("}"):
            try:
                obj = json.loads(output_str)
            except Exception:
                pass

    # function calling path
    if isinstance(obj, dict) and obj.get("tool_calls"):
        return _from_tool_call(obj)

    # content might contain a JSON plan string
    if isinstance(obj, dict) and isinstance(obj.get("content"), str):
        content = strip_code_block(obj["content"]).strip()
        try:
            return PlanModel.model_validate(json.loads(content))
        except Exception:
            return PlanModel(action="final", output=content)

    # strict validation for dict-like plan objects
    if isinstance(obj, dict):
        # Normalization: some models return action as tool name (e.g. {"action":"retrieval", ...})
        action = obj.get("action")
        if isinstance(action, str) and action not in ("tool", "final"):
            if action in ("retrieval", "search", "calculator") and "tool" not in obj:
                normalized = dict(obj)
                normalized["tool"] = action
                normalized["action"] = "tool"
                obj = normalized

        return PlanModel.model_validate(obj)

    return PlanModel(action="final", output=str(obj))
