"""Tool base protocol."""

from typing import Protocol, Any, Dict


class Tool(Protocol):
    name: str
    description: str

    def run(self, tool_input: Dict[str, Any], trace_id: str) -> Dict[str, Any]: ...


class ToolRegistry(dict):
    """Simple registry mapping tool name to instance."""

    def register(self, tool: Tool):
        self[tool.name] = tool
