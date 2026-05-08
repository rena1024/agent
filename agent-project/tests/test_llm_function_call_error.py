import json
import pytest

from agent.parser import parse_plan
from pydantic import ValidationError


def test_tool_call_arguments_malformed():
    # 模拟 LLM 返回的 tool_calls，但 arguments 不是合法 JSON
    raw = {
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "calculator",
                    "arguments": "{bad json",  # malformed
                }
            }
        ],
    }
    with pytest.raises(ValueError):
        parse_plan(raw)
