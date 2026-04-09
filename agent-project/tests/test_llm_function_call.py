import json
from agent.parser import parse_plan
from agent.planner import Plan
from agent.agent import Agent
from config import settings
from tools import registry
from utils.logger import get_logger


class FakeLLM:
    """模拟支持 function calling 的 LLM 返回值."""

    def __init__(self, call_tool: str, args: dict):
        self.call_tool = call_tool
        self.args = args
        self.logger = get_logger()

    def chat(self, prompt: str, trace_id: str, use_function_calling: bool = True):
        # 仿造 Ollama/OpenAI 返回格式
        return {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": self.call_tool,
                        "arguments": json.dumps(self.args),
                    }
                } 
            ],
        }


def test_function_call_parsed_and_executes(monkeypatch):
    # 准备假 LLM
    fake_llm = FakeLLM("calculator", {"expression": "2+3"})
    # 让 Planner 使用 fake_llm
    from agent.planner import Planner

    planner = Planner(settings=settings, logger=fake_llm.logger)
    planner.llm = fake_llm

    # 覆盖 Agent 的 planner/executor 以复用工具
    agent = Agent(settings=settings)
    agent.planner = planner
    settings.tool_registry = registry  # 确保工具注册

    # 执行一次 run
    result = agent.run("计算 2+3")
    assert result == "5"
