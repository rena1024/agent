from agent.agent import Agent
from config import settings
from tools import registry


def test_low_confidence_llm_route_falls_back_to_search(monkeypatch):
    settings.tool_registry = registry
    settings.router_confidence_threshold = 0.9
    agent = Agent(settings=settings)

    # Force an LLM-sourced low-confidence route; agent should fallback to search pipeline.
    agent.router.classify = lambda *args, **kwargs: {  # type: ignore[assignment]
        "task_type": "Generation",
        "pipeline": "direct_generate",
        "confidence": 0.1,
        "source": "llm",
    }

    def _boom(*args, **kwargs):
        raise AssertionError(
            "direct_generate should not be used on low-confidence route"
        )

    monkeypatch.setattr(agent, "_pipeline_direct_generate", _boom)

    # Mock search pipeline to make it deterministic.
    monkeypatch.setattr(
        agent, "_pipeline_tool_call_answer", lambda *_args, **_kwargs: "ok"
    )
    assert agent.run("随便问点啥") == "ok"
