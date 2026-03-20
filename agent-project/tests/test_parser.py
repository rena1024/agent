from pydantic import ValidationError

from agent.parser import parse_plan


def test_parse_tool_plan():
    raw = {"action": "tool", "tool": "calculator", "tool_input": {"expression": "1+1"}}
    plan = parse_plan(raw)
    assert plan.action == "tool"
    assert plan.tool == "calculator"
    assert plan.tool_input["expression"] == "1+1"


def test_parse_invalid_action():
    raw = {"action": "oops"}
    try:
        parse_plan(raw)
    except ValidationError as e:
        assert e.errors()[0]["loc"] == ("action",)
