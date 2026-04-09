from tools.base import Tool
from rag.index import ChromaIndex
from typing import Dict, Any


class Retrieval(Tool):
    name = "retrieval"
    description = "Retrieves documents from the knowledge base."

    def __init__(self):
        self.index = ChromaIndex()

    def run(self, tool_input: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        query = tool_input.get("query", "")
        if isinstance(query, list):
            query = " ".join(map(str, query))
        query = str(query)
        try:
            top_k = int(tool_input.get("top_k", 3))
        except Exception:
            top_k = 3

        threshold = 0.6

        if not query.strip():
            return {"status": "error", "output": "Query is required"}
        try:
            hits = self.index.query(query, top_k)
            if isinstance(hits, dict) and "error" in hits:
                return {"status": "error", "output": hits["error"]}
            filtered = []
            for h in hits:
                dist = h.get("distance", 1.0)
                if dist <= threshold:
                    filtered.append(h)
            if not filtered:
                return {
                    "status": "error",
                    "output": f"No relevant context found (threshold {threshold})",
                }
            return {"status": "ok", "output": filtered}
        except Exception as e:
            return {"status": "error", "output": str(e)}
