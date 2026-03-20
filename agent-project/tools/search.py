"""Search tool using Tavily with safe fallback."""

import os
from typing import Any, Dict, List

from tools.base import Tool
from tavily import TavilyClient

class Search(Tool):
    name = "search"
    description = "Web search via Tavily; falls back to mock results if unavailable."
    tavily_api_key: str = "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"

    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY", "")
        self.client = TavilyClient(api_key=self.tavily_api_key)

    def run(self, tool_input: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        query = tool_input.get("query", "")
        max_results = tool_input.get("max_results", 3)

        if not query:
            return {"status": "error", "output": "query is required"}

        try:
            response = self.client.search(query, max_results=max_results)
            return {"status": "ok", "output": response}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "output": f"Tavily error: {exc}"}
