"""Search tool using Tavily."""

import os
from typing import Any, Dict

from tools.base import Tool
from tavily import TavilyClient


class Search(Tool):
    name = "search"
    description = "Web search via Tavily."

    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self.client: TavilyClient | None = None

    def _get_client(self) -> TavilyClient:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        self.client = TavilyClient(api_key=self.api_key)
        return self.client

    def run(self, tool_input: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        query = tool_input.get("query", "")
        max_results = tool_input.get("max_results", 3)

        if not query:
            return {"status": "error", "output": "query is required"}

        try:
            response = self._get_client().search(query, max_results=max_results)
            return {"status": "ok", "output": response}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "output": f"Tavily error: {exc}"}
