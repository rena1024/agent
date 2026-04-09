"""Lightweight intent routing: decide which tool to use before LLM."""

import re
from dataclasses import dataclass


@dataclass
class RoutedPlan:
    action: str
    tool_name: str | None = None
    tool_input: dict | None = None
    thoughts: str | None = None
    output: str | None = None


INTERNAL_KEYWORDS = [
    # 中文
    "内部",
    "文档",
    "项目",
    "接口",
    "配置",
    "公司",
    "团队",
    "架构",
    "方案",
    # 英文
    "internal",
    "doc",
    "docs",
    "documentation",
    "project",
    "api",
    "config",
    "architecture",
    "design",
    "spec",
]

MATH_KEYWORDS = [
    "计算",
    "求值",
    "算",
    "结果",  # 中文
    "calculate",
    "compute",
    "sum",
    "add",
    "minus",
    "multiply",
    "divide",  # 英文
]


def route_intent(memory) -> RoutedPlan | None:
    """Return a RoutedPlan if intent is obvious, else None."""
    if not memory.messages:
        return None
    last_user = next(
        (m for m in reversed(memory.messages) if m["role"] == "user"), None
    )
    if not last_user:
        return None
    text = last_user.get("content", "")

    # arithmetic quick check
    if re.search(r"\d[\d\s+\-*/().]*\d", text) or any(k in text for k in MATH_KEYWORDS):
        return RoutedPlan(
            action="tool",
            tool_name="calculator",
            tool_input={"expression": text},
            thoughts="rule: arithmetic",
        )

    elif any(k in text for k in INTERNAL_KEYWORDS):
        return RoutedPlan(
            action="tool",
            tool_name="retrieval",
            tool_input={"query": text, "top_k": 3},
            thoughts="rule: internal",
        )

    else:
        return RoutedPlan(
            action="tool",
            tool_name="search",
            tool_input={"query": text},
            thoughts="rule: web",
        )
