"""ReAct-style runner: think-act-observe loop without separate planner/executor."""

from agent.prompt import build_react_prompt
from agent.parser import parse_plan
from config import Settings


class Reactor:
    def __init__(self, settings: Settings, logger):
        self.settings = settings
        self.logger = logger
        self.max_steps = settings.max_steps
        self.tools = settings.tool_registry

    def run(self, memory, llm, trace_id: str) -> str:
        for step in range(self.max_steps):
            prompt = build_react_prompt(memory.messages, self.tools.keys())
            plan_dict = llm.chat(prompt, trace_id=trace_id)
            if self.logger:
                self.logger.info(
                    "react.plan", extra={"trace_id": trace_id, "step": step, "plan": plan_dict}
                )
            plan = parse_plan(plan_dict)

            if plan.action == "final":
                memory.add_agent_message(plan.output or "", trace_id=trace_id)
                return plan.output or ""

            if plan.action == "tool":
                tool = self.tools.get(plan.tool)
                if not tool:
                    memory.add_agent_message(f"Unknown tool {plan.tool}", trace_id=trace_id)
                    return f"Unknown tool {plan.tool}"
                result = tool.run(plan.tool_input or {}, trace_id=trace_id)
                memory.add_tool_message(result, plan.tool or "unknown", trace_id=trace_id)
                # short-circuit on success
                # if result.get("status") == "ok":
                #     memory.add_agent_message(result.get("output", ""), trace_id=trace_id)
                #     return result.get("output", "")
                continue

        return "Reached max steps without conclusion."
