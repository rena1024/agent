"""High-level agent loop that coordinates planner and executor."""

from agent.executor import Executor
from agent.memory import Memory
from agent.planner import Planner
from agent.reactor import Reactor
from utils.logger import get_logger
from config import Settings
from agent.prompt import build_rag_answer_prompt
from agent.intents import route_intent


class Agent:
    def __init__(self, settings: Settings):
        self.logger = get_logger()
        self.memory = Memory()
        self.planner = Planner(settings=settings, logger=self.logger)
        self.executor = Executor(settings=settings, logger=self.logger)
        self.reactor = Reactor(settings=settings, logger=self.logger)
        self.settings = settings

    def run(self, user_input: str) -> str:
        trace_id = self.logger.new_trace_id()
        self.memory.add_user_message(user_input, trace_id=trace_id)

        if self.settings.mode == "react":
            return self.reactor.run(self.memory, self.planner.llm, trace_id=trace_id)

        for step_idx in range(self.executor.max_steps):
            routed_plan = route_intent(self.memory)
            plan = routed_plan or self.planner.decide(self.memory, trace_id=trace_id)
            self.logger.info(
                "planner.step",
                extra={"trace_id": trace_id, "step": step_idx, "plan": plan},
            )

            if plan.action == "final":
                self.memory.add_agent_message(plan.output, trace_id=trace_id)
                return plan.output

            tool_result = self.executor.execute(plan, trace_id=trace_id)
            # 先记录工具输出
            self.memory.add_tool_message(
                tool_result, plan.tool_name or "unknown", trace_id=trace_id
            )

            # 检索工具成功：用上下文生成最终回答
            if tool_result.get("status") == "ok" and plan.tool_name == "retrieval":
                context = tool_result.get("output", [])
                rag_prompt = build_rag_answer_prompt(self.memory.messages, context)
                llm_resp = self.planner.llm.chat(
                    rag_prompt, use_function_calling=False, trace_id=trace_id
                )
                final_output = (
                    llm_resp.get("output", "")
                    if isinstance(llm_resp, dict)
                    else str(llm_resp)
                )
                self.memory.add_agent_message(final_output, trace_id=trace_id)
                return final_output

            # 其他工具成功：直接返回结果
            if tool_result.get("status") == "ok":
                final_output = tool_result.get("output", "")
                self.memory.add_agent_message(final_output, trace_id=trace_id)
                return final_output

        return "Reached max steps without conclusion."
