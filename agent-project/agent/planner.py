"""Planner decides the next action for the agent."""

from dataclasses import dataclass
from typing import Any, Dict

from agent.prompt import build_planner_prompt
from llm.client import LLMClient
from config import Settings

from pydantic import ValidationError
from agent.parser import parse_plan
from colorama import Fore


@dataclass
class Plan:
    action: str  # "tool" or "final"
    tool_name: str | None = None
    tool_input: Dict[str, Any] | None = None
    output: str | None = None
    thoughts: str | None = None


class Planner:
    def __init__(self, settings: Settings, logger):
        self.llm = LLMClient(settings=settings, logger=logger)
        self.logger = logger
        self.available_tools = settings.tool_registry.keys()

    def decide(self, memory, trace_id: str) -> Plan:
        facts_block = ""
        try:
            facts_block = memory.facts_block()
        except Exception:
            facts_block = ""
        prompt = build_planner_prompt(memory.messages, self.available_tools, facts_block=facts_block)
        self.logger.info(
            "planner.prompt", extra={"trace_id": trace_id, "prompt": prompt}
        )
        response = self.llm.chat(prompt, trace_id=trace_id)

        return self._parse_response(response)

    def _parse_response(self, response: dict) -> Plan:
        try:
            plan_model = parse_plan(response)
        except ValidationError as e:
            self.logger.error(
                "planner.parse_error", extra={"errors": e.errors(), "raw": response}
            )
            return Plan(action="final", output="Failed to parse plan")

        if plan_model.action == "final":
            return Plan(action="final", output=plan_model.output or "")
        return Plan(
            action="tool",
            tool_name=plan_model.tool,
            tool_input=plan_model.tool_input,
            thoughts=plan_model.thoughts,
        )
