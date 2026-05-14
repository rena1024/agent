"""ReAct-style runner: think-act-observe loop without separate planner/executor."""

import json

from agent.prompt import build_react_prompt
from agent.parser import parse_plan
from config import Settings
from rag.rewrite import rewrite_query


class Reactor:
    def __init__(self, settings: Settings, logger):
        self.settings = settings
        self.logger = logger
        self.max_steps = settings.max_steps
        self.tools = settings.tool_registry

    def _retrieval_refs_for_memory(self, *, query: str, tool_result: dict) -> str:
        rerank_backend = tool_result.get("rerank_backend")
        hits = tool_result.get("output") or []
        refs = []
        if isinstance(hits, list):
            for h in hits[:10]:
                meta = (h or {}).get("metadata") or {}
                refs.append(
                    {
                        "source": meta.get("source"),
                        "center_chunk_id": meta.get(
                            "center_chunk_id", meta.get("chunk_id")
                        ),
                        "chunk_ids": meta.get("chunk_ids"),
                    }
                )
        payload = {"query": query, "rerank_backend": rerank_backend, "hits": refs}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _summarize_context(
        self, *, question: str, context_chunks: list, llm, trace_id: str
    ) -> str:
        parts = []
        for i, c in enumerate(context_chunks):
            text = (c.get("text", "") or "")[:300]
            meta = c.get("metadata", {}) or {}
            src = meta.get("source", "unknown")
            center = meta.get("center_chunk_id", meta.get("chunk_id", "?"))
            chunk_ids = meta.get("chunk_ids", None)
            extra = ""
            if isinstance(chunk_ids, list) and chunk_ids:
                extra = f", chunk_ids:{chunk_ids}"
            parts.append(
                f"[{i + 1}] (source:{src}, center_chunk:{center}{extra}) {text}"
            )

        snippets = "\n".join(parts)
        prompt = (
            f"用户问题：{question}\n"
            "请用自己的话总结以下片段，提取与用户问题最相关的要点，限制在180字以内。"
            "禁止逐字复制原文，单句话中不得有超过12个连续原文字符。"
            "保留引用编号（如[1][2]），以便标注来源。\n"
            f"{snippets}\n"
            "输出：直接给出最终回答（含引用编号）。"
        )
        return str(llm.chat_plain(prompt, trace_id=trace_id))

    def _summarize_search_results(
        self, *, question: str, search_output: object, llm, trace_id: str
    ) -> str:
        results = []
        if isinstance(search_output, dict):
            results = search_output.get("results") or []
        if not isinstance(results, list):
            results = []

        parts = []
        src_map = []
        for i, r in enumerate(results[:5]):
            if not isinstance(r, dict):
                continue
            sid = f"S{i + 1}"
            title = str(r.get("title", "") or "")[:160]
            url = str(r.get("url", "") or "")[:300]
            content = str(r.get("content", "") or "")[:500]
            parts.append(f"{sid} {title}\nurl: {url}\nsnippet: {content}")
            src_map.append(f"{sid}: {title} - {url}")

        if not parts:
            return "我没有从搜索结果中找到可用的内容，能否换个关键词或补充细节？"

        prompt = (
            f"用户问题：{question}\n"
            "你会收到若干条搜索结果（标题/URL/片段）。\n"
            "任务：提取要点并综合成简洁回答，避免逐字复制片段。\n"
            "要求：\n"
            "- 用中文回答（除非问题明确要求其他语言）。\n"
            "- 给出 3-6 条要点（可用短句或编号）。\n"
            "- 每条要点末尾标注来源ID，如（来源：S1）或（来源：S1,S3）。\n"
            "- 如果结果是观点/博客，提醒可能存在主观性或时效性。\n"
            "- 末尾输出一个“来源”区块，列出用到的来源ID到URL的映射，格式：S1: 标题 - URL。\n"
            "\n搜索结果：\n"
            + "\n\n".join(parts)
            + "\n\n可用来源映射：\n"
            + "\n".join(src_map)
            + "\n\n输出：最终回答（含来源ID与来源区块）。"
        )
        return str(llm.chat_plain(prompt, trace_id=trace_id))

    def _last_user_text(self, memory) -> str:
        for m in reversed(getattr(memory, "messages", []) or []):
            if m.get("role") == "user":
                return str(m.get("content", ""))
        return ""

    def _facts_block(self, memory) -> str:
        try:
            return memory.facts_block()
        except Exception:
            return ""

    def _rewrite_retrieval_query(
        self, memory, llm, question: str, trace_id: str
    ) -> str:
        if not getattr(self.settings, "enable_query_rewrite", True):
            return (question or "").strip()
        return rewrite_query(
            llm=llm,
            question=question,
            facts_block=self._facts_block(memory),
            max_chars=int(getattr(self.settings, "query_rewrite_max_chars", 120)),
            trace_id=trace_id,
        )

    def _accept_retrieval(self, tool_result: dict) -> bool:
        stats = tool_result.get("retrieval_stats") or {}
        try:
            best = float(stats.get("best_distance", 1.0))
            gap = float(stats.get("gap_distance", 0.0))
        except Exception:
            return True

        accept = float(getattr(self.settings, "retrieval_accept_distance", 0.38))
        soft = float(getattr(self.settings, "retrieval_soft_distance", 0.48))
        gap_th = float(getattr(self.settings, "retrieval_gap_distance", 0.06))

        if best <= accept:
            return True
        if best <= soft and gap >= gap_th:
            return True
        return False

    def run(self, memory, llm, trace_id: str) -> str:
        for step in range(self.max_steps):
            prompt = build_react_prompt(
                memory.messages,
                self.tools.keys(),
                facts_block=self._facts_block(memory),
            )
            plan_dict = llm.chat(prompt, trace_id=trace_id)
            if self.logger:
                self.logger.info(
                    "react.plan",
                    extra={"trace_id": trace_id, "step": step, "plan": plan_dict},
                )
            plan = parse_plan(plan_dict)

            if plan.action == "final":
                memory.add_agent_message(plan.output or "", trace_id=trace_id)
                try:
                    memory.maybe_summarize(llm, trace_id=trace_id)
                except Exception:
                    pass
                return plan.output or ""

            if plan.action == "tool":
                tool = self.tools.get(plan.tool)
                if not tool:
                    memory.add_agent_message(
                        f"Unknown tool {plan.tool}", trace_id=trace_id
                    )
                    return f"Unknown tool {plan.tool}"
                tool_input = plan.tool_input or {}
                if plan.tool == "retrieval":
                    q0 = str(tool_input.get("query", self._last_user_text(memory)))
                    q1 = self._rewrite_retrieval_query(
                        memory, llm, q0, trace_id=trace_id
                    )
                    tool_input = dict(tool_input)
                    tool_input["query"] = q1
                result = tool.run(tool_input, trace_id=trace_id)
                if plan.tool == "retrieval":
                    q = str(tool_input.get("query", ""))
                    try:
                        refs = self._retrieval_refs_for_memory(
                            query=q, tool_result=result
                        )
                        memory.add_tool_message_compact(
                            content=refs, tool_name="retrieval", trace_id=trace_id
                        )
                    except Exception:
                        memory.add_tool_message(
                            result, plan.tool or "unknown", trace_id=trace_id
                        )
                elif plan.tool == "search" and result.get("status") == "ok":
                    # Store compact refs only to avoid bloating history.
                    out = result.get("output") or {}
                    results = out.get("results") if isinstance(out, dict) else []
                    refs = []
                    if isinstance(results, list):
                        for r in results[:5]:
                            if not isinstance(r, dict):
                                continue
                            refs.append(
                                {
                                    "title": r.get("title"),
                                    "url": r.get("url"),
                                    "score": r.get("score"),
                                }
                            )
                    memory.add_tool_message_compact(
                        content=json.dumps(
                            {"query": tool_input.get("query", ""), "hits": refs},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        tool_name="search",
                        trace_id=trace_id,
                    )
                else:
                    memory.add_tool_message(
                        result, plan.tool or "unknown", trace_id=trace_id
                    )
                try:
                    memory.maybe_summarize(llm, trace_id=trace_id)
                except Exception:
                    pass
                # short-circuit on success
                if result.get("status") == "ok":
                    if plan.tool == "retrieval":
                        if not self._accept_retrieval(result):
                            # Let the loop continue so the model can choose another tool / fallback.
                            continue
                        context = result.get("output", [])
                        out = self._summarize_context(
                            question=self._last_user_text(memory),
                            context_chunks=context if isinstance(context, list) else [],
                            llm=llm,
                            trace_id=trace_id,
                        )
                        memory.add_agent_message(out, trace_id=trace_id)
                        try:
                            memory.maybe_summarize(llm, trace_id=trace_id)
                        except Exception:
                            pass
                        return out

                    if plan.tool == "search":
                        out = self._summarize_search_results(
                            question=self._last_user_text(memory),
                            search_output=result.get("output"),
                            llm=llm,
                            trace_id=trace_id,
                        )
                        memory.add_agent_message(out, trace_id=trace_id)
                        try:
                            memory.maybe_summarize(llm, trace_id=trace_id)
                        except Exception:
                            pass
                        return out

                    out = str(result.get("output", ""))
                    memory.add_agent_message(out, trace_id=trace_id)
                    try:
                        memory.maybe_summarize(llm, trace_id=trace_id)
                    except Exception:
                        pass
                    return out
                continue

        return "Reached max steps without conclusion."
