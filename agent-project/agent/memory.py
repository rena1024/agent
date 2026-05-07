"""Simple in-memory message store."""

import re
from typing import List


class Memory:
    def __init__(self):
        self.messages: List[dict] = []
        # Keep a small live window; older turns will be summarized into a system message.
        self.max_turns = 3
        self.max_tool = 2
        self.summary: str | None = None
        self.facts: dict[str, str] = {}

    def facts_block(self) -> str:
        if not self.facts:
            return ""
        lines = [f"- {k}: {v}" for k, v in sorted(self.facts.items())]
        return "\n".join(lines)

    def _extract_facts(self, text: str) -> None:
        """
        Lightweight fact extraction for interview-style chat.
        Keep it deterministic so it doesn't depend on LLM behavior.
        """
        t = (text or "").strip()
        if not t:
            return

        patterns = [
            r"(?:我叫|我的名字是|叫我)\s*([A-Za-z][\w\-]{0,30}|[\u4e00-\u9fff]{1,10})",
            r"(?:my name is|i am|i'm)\s+([A-Za-z][\w\-]{0,30})",
        ]
        for p in patterns:
            m = re.search(p, t, flags=re.IGNORECASE)
            if m:
                name = m.group(1).strip().strip("。.!?，,")
                if name:
                    self.facts["user_name"] = name
                break

    def add_user_message(self, content: str, trace_id: str):
        self._extract_facts(content)
        self.messages.append({"role": "user", "content": content, "trace_id": trace_id})
        self._trim_tool()

    def add_agent_message(self, content: str, trace_id: str):
        self.messages.append(
            {"role": "assistant", "content": content, "trace_id": trace_id}
        )
        self._trim_tool()

    def add_tool_message(self, tool_result: dict, tool_name: str, trace_id: str):
        output = tool_result.get("output", "")
        self.messages.append(
            {
                "role": "tool",
                "content": f"tool {tool_name} output: {output}",
                "trace_id": trace_id,
            }
        )
        self._trim_tool()

    def add_tool_message_compact(
        self, *, content: str, tool_name: str, trace_id: str
    ) -> None:
        """
        Add a compact tool message that does not embed raw tool outputs.
        Useful for retrieval/RAG to avoid stuffing large chunks into history.
        """
        self.messages.append(
            {
                "role": "tool",
                "content": f"tool {tool_name} refs: {content}",
                "trace_id": trace_id,
            }
        )
        self._trim_tool()

    def _trim_tool(self):
        """Keep only the most recent tool outputs to avoid huge prompts."""
        tool_idxs = [i for i, m in enumerate(self.messages) if m.get("role") == "tool"]
        if len(tool_idxs) <= self.max_tool:
            return
        drop = set(tool_idxs[: -self.max_tool])
        self.messages = [m for i, m in enumerate(self.messages) if i not in drop]

    def _turn_spans(self) -> list[tuple[int, int]]:
        """
        Turn = user message plus following assistant/tool messages until next user.
        Returns list of (start_idx, end_idx_exclusive) spans.
        """
        user_idxs = [i for i, m in enumerate(self.messages) if m.get("role") == "user"]
        if not user_idxs:
            return []
        spans: list[tuple[int, int]] = []
        for j, start in enumerate(user_idxs):
            end = user_idxs[j + 1] if j + 1 < len(user_idxs) else len(self.messages)
            spans.append((start, end))
        return spans

    def maybe_summarize(self, llm, trace_id: str):
        """
        Summarize turns older than the live window into a single system message.
        Uses llm.chat_plain when available.
        """
        spans = self._turn_spans()
        if len(spans) <= self.max_turns:
            return

        keep_spans = spans[-self.max_turns :]
        summarize_spans = spans[: -self.max_turns]

        summarize_msgs: list[dict] = []
        for start, end in summarize_spans:
            summarize_msgs.extend(self.messages[start:end])

        def fmt(m: dict) -> str:
            role = m.get("role", "unknown")
            content = str(m.get("content", ""))
            return f"{role}: {content}"

        history_block = "\n".join(fmt(m) for m in summarize_msgs)
        prev_summary = self.summary or ""
        prompt = (
            "Update the conversation summary for an agent.\n"
            "Keep: user facts/preferences, decisions/constraints, unresolved items.\n"
            "Be concise (<= 12 lines). Output summary text only.\n"
            f"\nExisting summary:\n{prev_summary}\n"
            f"\nNew history:\n{history_block}\n"
            "\nUpdated summary:"
        )

        try:
            summary_text = llm.chat_plain(prompt, trace_id=trace_id)
        except Exception:
            summary_text = (prev_summary + "\n" + history_block).strip()[:1200]

        self.summary = str(summary_text).strip()

        kept_msgs: list[dict] = []
        for start, end in keep_spans:
            kept_msgs.extend(self.messages[start:end])

        system_msg = {
            "role": "system",
            "content": f"Conversation summary:\n{self.summary}",
            "trace_id": trace_id,
        }
        self.messages = [system_msg] + kept_msgs
        self._trim_tool()
