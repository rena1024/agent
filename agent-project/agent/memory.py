"""Simple in-memory message store."""

from typing import List


class Memory:
    def __init__(self):
        self.messages: List[dict] = []

    def add_user_message(self, content: str, trace_id: str):
        self.messages.append({"role": "user", "content": content, "trace_id": trace_id})

    def add_agent_message(self, content: str, trace_id: str):
        self.messages.append({"role": "assistant", "content": content, "trace_id": trace_id})

    def add_tool_message(self, tool_result: dict, tool_name: str, trace_id: str):
        output = tool_result.get("output", "")
        self.messages.append(
            {
                "role": "tool",
                "content": f"tool {tool_name} output: {output}",
                "trace_id": trace_id,
            }
        )
