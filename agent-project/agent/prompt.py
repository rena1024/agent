"""Prompt templates for planner and executor."""

from typing import List


def build_planner_prompt(messages: List[dict], tools) -> str:
    """Build planner prompt that includes conversation history and tool list."""
    history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    tool_list = ", ".join(tools)
    return (
        "You are a helpful suggestion agent. Decide the next action based on the dialogue.\n"
        f"Conversation history:\n{history}\n"
        f"Available tools: [{tool_list}]\n"
        "Rules:\n"
        "- Use conversation history as ground truth. If the answer already exists, answer directly with action='final'.\n"
        "- Only call tools when new external information is required.\n"
        "- For questions about the user's name, reuse the name stated in history; do not search.\n"
        "Return only JSON with fields:\n"
        "action: 'tool' | 'final'\n"
        "tool: tool name when action='tool'\n"
        "tool_input: object for the tool\n"
        "output: final reply to user when action='final'\n"
        "thoughts: brief reasoning\n"
        'Example: {"action":"tool","tool":"calculator","tool_input":{"expression":"1+1"},"thoughts":"先计算"}\n'
        "No extra text outside the JSON."
    )


def build_react_prompt(messages: List[dict], tools) -> str:
    """Prompt for ReAct-style single-step think-act-observe."""
    history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    tool_list = ", ".join(tools)
    return (
        "You are a ReAct agent. Given the dialogue and observations, decide the next action.\n"
        f"Conversation history:\n{history}\n"
        f"Available tools: [{tool_list}]\n"
        "Rules:\n"
        "- If you already have enough info, return final answer with action='final'.\n"
        "- If you need to use a tool, set action='tool' and fill tool/tool_input.\n"
        "- Keep thoughts brief.\n"
        "Output JSON only with fields: action, tool, tool_input, output, thoughts.\n"
        'Example: {"action":"tool","tool":"search","tool_input":{"query":"最新天气"},"thoughts":"先查询天气"}'
    )
