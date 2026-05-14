from agent.reactor import Reactor
from agent.memory import Memory
from config import settings
from tools import registry


class DummyLLM:
    def chat(self, prompt: str, trace_id: str):
        return {
            "action": "tool",
            "tool": "calculator",
            "tool_input": {"expression": "1+1"},
        }


def test_reactor_runs_tool_and_finishes():
    memory = Memory()
    memory.add_user_message("计算 1+1", trace_id="t1")
    settings.tool_registry = registry
    reactor = Reactor(settings=settings, logger=None)
    result = reactor.run(memory, DummyLLM(), trace_id="t1")
    assert result == "2"
