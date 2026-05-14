from agent.agent import Agent
from config import settings
from tools import registry


def test_router_extract_compute_bypasses_planner(monkeypatch):
    settings.tool_registry = registry
    agent = Agent(settings=settings)

    # Force router to pick the quantitative pipeline.
    agent.router.classify = lambda *args, **kwargs: {
        "task_type": "Quantitative",
        "pipeline": "extract_compute",
    }  # type: ignore[assignment]

    def _boom(*args, **kwargs):
        raise AssertionError("planner.decide should not be called for extract_compute")

    monkeypatch.setattr(agent.planner, "decide", _boom)

    assert agent.run("2+3") == "5"


def test_extract_compute_supports_caret_power():
    settings.tool_registry = registry
    agent = Agent(settings=settings)
    agent.router.classify = lambda *args, **kwargs: {
        "task_type": "Quantitative",
        "pipeline": "extract_compute",
    }  # type: ignore[assignment]
    assert agent.run("2^3") == "8"


def test_quant_word_problem_chicken_rabbit(monkeypatch):
    settings.tool_registry = registry
    agent = Agent(settings=settings)
    agent.router.classify = lambda *args, **kwargs: {
        "task_type": "Quantitative",
        "pipeline": "extract_compute",
    }  # type: ignore[assignment]

    def fake_chat_plain(prompt: str, trace_id: str):
        # Produce explanation + pure expression for calculator.
        return (
            '{"explanation":"1只鸡=2条腿\\n2只兔=2*4条腿\\n总腿数=鸡腿+兔腿",'
            '"expression":"1*2+2*4"}'
        )

    monkeypatch.setattr(agent.planner.llm, "chat_plain", fake_chat_plain)

    out = agent.run("一只鸡两只兔子有多少条腿？")
    assert "10" in out


def test_router_classifies_duck_legs_as_quantitative():
    from agent.router import Router

    r = Router()
    route = r.classify("鸭子有几条腿？")
    assert route["task_type"] == "Quantitative"
    assert route["pipeline"] == "extract_compute"


def test_router_classifies_cat_squirrel_legs_as_quantitative():
    from agent.router import Router

    r = Router()
    route = r.classify("两只猫三只松鼠一共有多少条腿？")
    assert route["task_type"] == "Quantitative"
    assert route["pipeline"] == "extract_compute"


def test_quant_common_sense_duck_legs_does_not_use_retrieval(monkeypatch):
    settings.tool_registry = registry
    agent = Agent(settings=settings)

    # Let router run as-is, but make sure we don't hit planner.decide.
    def _boom(*args, **kwargs):
        raise AssertionError(
            "planner.decide should not be called for quantitative commonsense"
        )

    monkeypatch.setattr(agent.planner, "decide", _boom)

    def fake_chat_plain(prompt: str, trace_id: str):
        return '{"explanation":"鸭子=2条腿","expression":"2"}'

    monkeypatch.setattr(agent.planner.llm, "chat_plain", fake_chat_plain)

    out = agent.run("鸭子有几条腿？")
    assert "2" in out


def test_quant_legs_duck_and_rabbits_llm_and_verify(monkeypatch):
    settings.tool_registry = registry
    agent = Agent(settings=settings)
    agent.router.classify = lambda *args, **kwargs: {
        "task_type": "Quantitative",
        "pipeline": "extract_compute",
    }  # type: ignore[assignment]

    def fake_chat_plain(prompt: str, trace_id: str):
        # First call: derive explanation + expression
        if "文字题转算式" in prompt:
            return (
                '{"explanation":"1只鸭=1*2条腿\\n2只兔=2*4条腿\\n总腿数=1*2+2*4",'
                '"expression":"1*2+2*4"}'
            )
        # Second call: verify ok
        if "审校员" in prompt:
            return '{"ok": true, "expression": "1*2+2*4"}'
        return '{"ok": false, "expression": ""}'

    monkeypatch.setattr(agent.planner.llm, "chat_plain", fake_chat_plain)

    out = agent.run("一只鸭子和两只兔子一共有多少条腿？")
    assert "10" in out
