"""High-level agent loop that coordinates planner and executor."""

import json
import re

from agent.executor import Executor
from agent.memory import Memory
from agent.planner import Planner, Plan
from agent.reactor import Reactor
from agent.router import Router
from utils.logger import get_logger
from config import Settings
from rag.rewrite import rewrite_query


class Agent:
    def __init__(self, settings: Settings):
        self.logger = get_logger()
        self.memory = Memory()
        self.planner = Planner(settings=settings, logger=self.logger)
        self.executor = Executor(settings=settings, logger=self.logger)
        self.reactor = Reactor(settings=settings, logger=self.logger)
        self.router = Router(llm=self.planner.llm, settings=settings)
        self.settings = settings

    def _finalize(self, content: str, trace_id: str) -> str:
        self.memory.add_agent_message(content, trace_id=trace_id)
        self.memory.maybe_summarize(self.planner.llm, trace_id=trace_id)
        return content

    _RE_PURE_EXPR = re.compile(r"^\s*[-+()0-9.\s*/^*]+\s*$")
    _RE_FIND_EXPR = re.compile(r"(\d[\d\s+\-*/().^*]*\d|\d)")

    def _extract_math_expression(self, text: str) -> str | None:
        t = (text or "").strip()
        if not t:
            return None
        if self._RE_PURE_EXPR.match(t) and re.search(r"\d", t):
            return t
        m = self._RE_FIND_EXPR.search(t)
        if not m:
            return None
        expr = (m.group(1) or "").strip()
        if not expr or not re.search(r"\d", expr):
            return None
        if not self._RE_PURE_EXPR.match(expr):
            return None
        return expr

    def _normalize_expression(self, expr: str) -> str:
        e = (expr or "").strip()
        # Users often use ^ for exponent; calculator only supports **.
        e = e.replace("^", "**")
        e = re.sub(r"\s+", "", e)
        return e

    def _is_safe_expression(self, expr: str) -> bool:
        """
        Only allow digits/operators/parentheses and **, so the calculator tool input
        stays a pure math expression (no words, no variables).
        """
        e = (expr or "").strip()
        if not e:
            return False
        if not re.search(r"\d", e):
            return False
        if not re.fullmatch(r"[0-9+\-*/().\s*^]+", e):
            return False
        return True

    def _derive_expression_with_llm(
        self, *, question: str, context: str = "", trace_id: str
    ) -> tuple[str, str]:
        """
        Ask the LLM to produce:
        - explanation: short human-readable reasoning steps
        - expression: a pure math expression (digits/operators only; exponent uses **)
        """
        ctx = (context or "").strip()
        ctx_block = f"可用事实/上下文：\n{ctx}\n" if ctx else ""
        legs_hint = (
            "常识提示（如涉及腿数可参考）：\n"
            "- 大多数鸟类（鸡/鸭/鹅等）通常 2 条腿\n"
            "- 大多数常见哺乳动物（兔/猫/狗等）通常 4 条腿\n"
            "- 昆虫通常 6 条腿；蜘蛛等蛛形纲通常 8 条腿\n"
            "如果你无法确定某对象的腿数/单位值，请把 expression 置空（不要猜）。\n"
        )
        prompt = (
            "你是一个“文字题转算式”的助手。\n"
            "任务：先用中文简要分析题意（例如每种对象对应的数量与单个取值），然后把题目改写成一个可计算的数学表达式。\n"
            "严格要求：\n"
            "- 仅输出 JSON，不要任何多余文字。\n"
            '- JSON 结构：{"explanation":"...","expression":"..."}\n'
            "- explanation：不超过4行，每行尽量短。\n"
            "- expression：只能包含数字、小数点、+ - * / ( ) 和 **（幂）；不得出现中文/英文单词/变量名。\n"
            "- 如果用户用了 ^ 表示幂，请改写为 **。\n"
            "- 如果无法确定表达式，expression 置空字符串。\n"
            f"{legs_hint}"
            f"{ctx_block}"
            f"题目：{question}\n"
            "输出："
        )
        raw = ""
        try:
            raw = str(self.planner.llm.chat_plain(prompt, trace_id=trace_id))
            data = json.loads(raw)
        except Exception:
            data = {}
        explanation = str((data or {}).get("explanation", "") or "").strip()
        expression = str((data or {}).get("expression", "") or "").strip()
        expression = self._normalize_expression(expression)
        if not self._is_safe_expression(expression):
            return explanation, ""
        return explanation, expression

    def _verify_expression_with_llm(
        self, *, question: str, explanation: str, expression: str, trace_id: str
    ) -> str:
        """
        Ask the LLM to sanity-check whether an expression matches the question/explanation.
        Return a possibly corrected expression (or empty if unsure).
        """
        if not explanation.strip() or not expression.strip():
            return ""
        prompt = (
            "你是一个审校员，检查“文字题→算式”的正确性。\n"
            "你会得到：题目、分析、算式。你需要判断算式是否与题目一致。\n"
            "严格要求：\n"
            "- 仅输出 JSON，不要多余文字。\n"
            '- JSON 结构：{"ok":true|false,"expression":"<正确算式或空>"}\n'
            "- expression 只能包含数字、小数点、+ - * / ( ) 和 **（幂）。\n"
            "- 若不确定，ok=false 且 expression 置空。\n"
            f"题目：{question}\n"
            f"分析：{explanation}\n"
            f"算式：{expression}\n"
            "输出："
        )
        try:
            raw = str(self.planner.llm.chat_plain(prompt, trace_id=trace_id))
            data = json.loads(raw)
            ok = bool((data or {}).get("ok", False))
            expr = str((data or {}).get("expression", "") or "").strip()
        except Exception:
            return ""
        if not ok:
            return ""
        expr = self._normalize_expression(expr)
        if not self._is_safe_expression(expr):
            return ""
        return expr

    def _pipeline_extract_compute(self, user_input: str, trace_id: str) -> str:
        expr0 = self._extract_math_expression(user_input)
        if expr0:
            expr = self._normalize_expression(expr0)
            tool_result = self.executor.execute(
                Plan(
                    action="tool",
                    tool_name="calculator",
                    tool_input={"expression": expr},
                ),
                trace_id=trace_id,
            )
            self.memory.add_tool_message(tool_result, "calculator", trace_id=trace_id)
            if tool_result.get("status") == "ok":
                return self._finalize(str(tool_result.get("output", "")), trace_id=trace_id)

        # Fallback: word-problem -> derive expression with LLM, then compute.
        explanation, expr1 = self._derive_expression_with_llm(
            question=user_input, trace_id=trace_id
        )
        if expr1:
            fixed = self._verify_expression_with_llm(
                question=user_input,
                explanation=explanation,
                expression=expr1,
                trace_id=trace_id,
            )
            if fixed:
                expr1 = fixed
            tool_result = self.executor.execute(
                Plan(
                    action="tool",
                    tool_name="calculator",
                    tool_input={"expression": expr1},
                ),
                trace_id=trace_id,
            )
            self.memory.add_tool_message(tool_result, "calculator", trace_id=trace_id)
            if tool_result.get("status") == "ok":
                value = str(tool_result.get("output", "") or "").strip()
                if explanation:
                    return self._finalize(f"{explanation}\n答案：{value}", trace_id=trace_id)
                return self._finalize(value, trace_id=trace_id)

        # Last resort: fall back to the legacy planner loop.
        return self._run_planner_loop(user_input, trace_id=trace_id)

    def _pipeline_tool_call_answer(self, user_input: str, trace_id: str) -> str:
        tool_result = self.executor.execute(
            Plan(action="tool", tool_name="search", tool_input={"query": user_input}),
            trace_id=trace_id,
        )
        if tool_result.get("status") == "ok":
            try:
                refs = self._search_refs_for_memory(query=user_input, tool_result=tool_result)
                self.memory.add_tool_message_compact(
                    content=refs, tool_name="search", trace_id=trace_id
                )
            except Exception:
                self.memory.add_tool_message(tool_result, "search", trace_id=trace_id)
            summary = self._summarize_search_results(
                user_input, tool_result.get("output"), trace_id=trace_id
            )
            return self._finalize(summary, trace_id=trace_id)
        self.memory.add_tool_message(tool_result, "search", trace_id=trace_id)
        return self._run_planner_loop(user_input, trace_id=trace_id)

    def _pipeline_tool_then_compute(self, user_input: str, trace_id: str) -> str:
        tool_result = self.executor.execute(
            Plan(action="tool", tool_name="search", tool_input={"query": user_input}),
            trace_id=trace_id,
        )
        if tool_result.get("status") != "ok":
            self.memory.add_tool_message(tool_result, "search", trace_id=trace_id)
            return self._run_planner_loop(user_input, trace_id=trace_id)

        # Keep compact refs in memory; use snippets only for local derivation prompt.
        try:
            refs = self._search_refs_for_memory(query=user_input, tool_result=tool_result)
            self.memory.add_tool_message_compact(
                content=refs, tool_name="search", trace_id=trace_id
            )
        except Exception:
            self.memory.add_tool_message(tool_result, "search", trace_id=trace_id)

        out = tool_result.get("output") or {}
        results = out.get("results") if isinstance(out, dict) else []
        snippets: list[str] = []
        if isinstance(results, list):
            for r in results[:5]:
                if not isinstance(r, dict):
                    continue
                snippets.append(str(r.get("content", "") or "")[:400])
        context = "\n---\n".join([s for s in snippets if s]).strip()[:1600]

        explanation, expr = self._derive_expression_with_llm(
            question=user_input, context=context, trace_id=trace_id
        )
        if not expr:
            return self._run_planner_loop(user_input, trace_id=trace_id)

        calc = self.executor.execute(
            Plan(action="tool", tool_name="calculator", tool_input={"expression": expr}),
            trace_id=trace_id,
        )
        self.memory.add_tool_message(calc, "calculator", trace_id=trace_id)
        if calc.get("status") == "ok":
            value = str(calc.get("output", "") or "").strip()
            if explanation:
                return self._finalize(f"{explanation}\n答案：{value}", trace_id=trace_id)
            return self._finalize(value, trace_id=trace_id)
        return self._run_planner_loop(user_input, trace_id=trace_id)

    def _pipeline_rewrite_retrieval_answer(self, user_input: str, trace_id: str) -> str:
        q = self._rewrite_retrieval_query(user_input, trace_id=trace_id)
        tool_result = self.executor.execute(
            Plan(action="tool", tool_name="retrieval", tool_input={"query": q}),
            trace_id=trace_id,
        )
        try:
            refs = self._retrieval_refs_for_memory(query=q, tool_result=tool_result)
            self.memory.add_tool_message_compact(
                content=refs, tool_name="retrieval", trace_id=trace_id
            )
        except Exception:
            self.memory.add_tool_message(tool_result, "retrieval", trace_id=trace_id)

        if tool_result.get("status") == "ok" and self._accept_retrieval(tool_result):
            context = tool_result.get("output", [])
            summary = self._summarize_context(user_input, context, trace_id=trace_id)
            return self._finalize(summary, trace_id=trace_id)

        # fallback to search
        return self._pipeline_tool_call_answer(user_input, trace_id=trace_id)

    def _pipeline_retrieval_then_compute(self, user_input: str, trace_id: str) -> str:
        q = self._rewrite_retrieval_query(user_input, trace_id=trace_id)
        tool_result = self.executor.execute(
            Plan(action="tool", tool_name="retrieval", tool_input={"query": q}),
            trace_id=trace_id,
        )
        try:
            refs = self._retrieval_refs_for_memory(query=q, tool_result=tool_result)
            self.memory.add_tool_message_compact(
                content=refs, tool_name="retrieval", trace_id=trace_id
            )
        except Exception:
            self.memory.add_tool_message(tool_result, "retrieval", trace_id=trace_id)

        if tool_result.get("status") != "ok" or not self._accept_retrieval(tool_result):
            return self._pipeline_tool_then_compute(user_input, trace_id=trace_id)

        chunks = tool_result.get("output", [])
        ctx_parts: list[str] = []
        if isinstance(chunks, list):
            for c in chunks[:6]:
                if not isinstance(c, dict):
                    continue
                ctx_parts.append(str(c.get("text", "") or "")[:350])
        context = "\n---\n".join([p for p in ctx_parts if p]).strip()[:1600]

        explanation, expr = self._derive_expression_with_llm(
            question=user_input, context=context, trace_id=trace_id
        )
        if not expr:
            # If we can't build an expression, just answer using retrieval summary.
            summary = self._summarize_context(user_input, chunks, trace_id=trace_id)
            return self._finalize(summary, trace_id=trace_id)

        calc = self.executor.execute(
            Plan(action="tool", tool_name="calculator", tool_input={"expression": expr}),
            trace_id=trace_id,
        )
        self.memory.add_tool_message(calc, "calculator", trace_id=trace_id)
        if calc.get("status") == "ok":
            value = str(calc.get("output", "") or "").strip()
            if explanation:
                return self._finalize(f"{explanation}\n答案：{value}", trace_id=trace_id)
            return self._finalize(value, trace_id=trace_id)

        return self._run_planner_loop(user_input, trace_id=trace_id)

    def _pipeline_direct_generate(self, user_input: str, trace_id: str) -> str:
        prompt = (
            "请直接回答用户问题。要求：\n"
            "- 用中文回答（除非用户要求其他语言）。\n"
            "- 简洁、正确。\n"
            f"用户问题：{user_input}\n"
            "回答："
        )
        try:
            out = str(self.planner.llm.chat_plain(prompt, trace_id=trace_id))
        except Exception:
            out = ""
        if out.strip():
            return self._finalize(out.strip(), trace_id=trace_id)
        return self._run_planner_loop(user_input, trace_id=trace_id)

    def _run_planner_loop(self, user_input: str, trace_id: str) -> str:
        for step_idx in range(self.executor.max_steps):
            plan = self.planner.decide(self.memory, trace_id=trace_id)
            self.logger.info(
                "planner.step",
                extra={"trace_id": trace_id, "step": step_idx, "plan": plan},
            )

            if plan.action == "final":
                return self._finalize(plan.output or "", trace_id=trace_id)

            # If the model decided retrieval, rewrite query for better recall.
            if plan.tool_name == "retrieval":
                q0 = str((plan.tool_input or {}).get("query", user_input))
                q1 = self._rewrite_retrieval_query(q0, trace_id=trace_id)
                plan.tool_input = dict(plan.tool_input or {})
                plan.tool_input["query"] = q1
                self.logger.info(
                    "rag.query_rewrite",
                    extra={
                        "trace_id": trace_id,
                        "original_query": q0,
                        "rewritten_query": q1,
                    },
                )

            tool_result = self.executor.execute(plan, trace_id=trace_id)

            # 检索工具成功：用上下文生成最终回答, 失败则回退到搜索
            if plan.tool_name == "retrieval":
                # Only store references in memory to avoid stuffing chunks into the prompt.
                q = str((plan.tool_input or {}).get("query", user_input))
                try:
                    refs = self._retrieval_refs_for_memory(
                        query=q, tool_result=tool_result
                    )
                    self.memory.add_tool_message_compact(
                        content=refs, tool_name="retrieval", trace_id=trace_id
                    )
                except Exception:
                    self.memory.add_tool_message(
                        tool_result, "retrieval", trace_id=trace_id
                    )

                if tool_result.get("status") != "ok" or (
                    tool_result.get("status") == "ok"
                    and not self._accept_retrieval(tool_result)
                ):
                    if tool_result.get("status") == "ok":
                        self.logger.info(
                            "rag.reject",
                            extra={
                                "trace_id": trace_id,
                                "stats": tool_result.get("retrieval_stats"),
                                "reason": "distance_policy",
                            },
                        )
                    search_result = self.executor.execute(
                        Plan(
                            action="tool",
                            tool_name="search",
                            tool_input={"query": user_input},
                        ),
                        trace_id=trace_id,
                    )
                    if search_result.get("status") == "ok":
                        # Store compact refs only (avoid stuffing snippets into history)
                        try:
                            refs = self._search_refs_for_memory(
                                query=user_input, tool_result=search_result
                            )
                            self.memory.add_tool_message_compact(
                                content=refs, tool_name="search", trace_id=trace_id
                            )
                        except Exception:
                            self.memory.add_tool_message(
                                search_result, "search", trace_id=trace_id
                            )

                        summary = self._summarize_search_results(
                            user_input,
                            search_result.get("output"),
                            trace_id=trace_id,
                        )
                        self.logger.info(
                            "search.summary",
                            extra={"trace_id": trace_id, "summary": summary},
                        )
                        return self._finalize(summary, trace_id=trace_id)
                    # keep the error visible but short
                    try:
                        self.memory.add_tool_message_compact(
                            content=json.dumps(
                                {"query": user_input, "status": "error"},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            tool_name="search",
                            trace_id=trace_id,
                        )
                    except Exception:
                        self.memory.add_tool_message(
                            search_result, "search", trace_id=trace_id
                        )
                else:
                    context = tool_result.get("output", [])
                    summary = self._summarize_context(
                        user_input, context, trace_id=trace_id
                    )
                    self.logger.info(
                        "rag.summary",
                        extra={
                            "trace_id": trace_id,
                            "summary": summary,
                            "centers": self._center_chunk_refs(context),
                        },
                    )
                    return self._finalize(summary, trace_id=trace_id)

            # 先记录其他工具输出（允许完整输出，因为通常比较短）
            if plan.tool_name == "search":
                # Avoid stuffing raw snippets into history; store references only.
                q = str((plan.tool_input or {}).get("query", user_input))
                try:
                    refs = self._search_refs_for_memory(query=q, tool_result=tool_result)
                    self.memory.add_tool_message_compact(
                        content=refs, tool_name="search", trace_id=trace_id
                    )
                except Exception:
                    self.memory.add_tool_message(
                        tool_result, "search", trace_id=trace_id
                    )
            else:
                self.memory.add_tool_message(
                    tool_result, plan.tool_name or "unknown", trace_id=trace_id
                )

            # 其他工具成功：直接返回结果
            if tool_result.get("status") == "ok":
                if plan.tool_name == "search":
                    search_output = tool_result.get("output")
                    summary = self._summarize_search_results(
                        user_input, search_output, trace_id=trace_id
                    )
                    self.logger.info(
                        "search.summary",
                        extra={"trace_id": trace_id, "summary": summary},
                    )
                    return self._finalize(summary, trace_id=trace_id)

                final_output = tool_result.get("output", "")
                return self._finalize(str(final_output), trace_id=trace_id)

        return "Reached max steps without conclusion."

    def _summarize_context(
        self, question: str, context_chunks: list, trace_id: str
    ) -> str:
        """Summarize retrieved chunks to reduce length and avoid verbatim copying."""
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
            parts.append(f"[{i+1}] (source:{src}, center_chunk:{center}{extra}) {text}")
        snippets = "\n".join(parts)
        prompt = (
            f"用户问题：{question}\n"
            "请用自己的话总结以下片段，提取与用户问题最相关的要点，限制在180字以内。"
            "禁止逐字复制原文，单句话中不得有超过12个连续原文字符。"
            f"{snippets}\n"
            "输出：直接给出最终回答。"
        )
        resp = self.planner.llm.chat_plain(prompt, trace_id=trace_id)
        return str(resp)

    def _summarize_search_results(self, question: str, search_output: object, trace_id: str) -> str:
        """
        Summarize search results into a final user-facing answer with sources [1][2]...
        Expects Tavily-like output: {"results":[{"title","url","content","score"},...]}.
        """
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
            sid = f"S{i+1}"
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
        return str(self.planner.llm.chat_plain(prompt, trace_id=trace_id))

    def _search_refs_for_memory(self, *, query: str, tool_result: dict) -> str:
        out = tool_result.get("output") or {}
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
        payload = {"query": query, "hits": refs}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _retrieval_refs_for_memory(self, *, query: str, tool_result: dict) -> str:
        rerank_backend = tool_result.get("rerank_backend")
        stats = tool_result.get("retrieval_stats") or {}
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
        payload = {
            "query": query,
            "rerank_backend": rerank_backend,
            "stats": stats,
            "hits": refs,
        }
        # Keep it compact and stable in history.
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _rewrite_retrieval_query(self, question: str, trace_id: str) -> str:
        if not getattr(self.settings, "enable_query_rewrite", True):
            return (question or "").strip()
        facts_block = ""
        try:
            facts_block = self.memory.facts_block()
        except Exception:
            facts_block = ""
        return rewrite_query(
            llm=self.planner.llm,
            question=question,
            facts_block=facts_block,
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

    def _center_chunk_refs(self, context: object) -> list[dict]:
        """
        Extract (source, center_chunk_id, chunk_ids) from retrieval windows/hits for logging.
        """
        if not isinstance(context, list):
            return []
        refs: list[dict] = []
        for h in context[:10]:
            if not isinstance(h, dict):
                continue
            meta = h.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}
            refs.append(
                {
                    "source": meta.get("source"),
                    "center_chunk_id": meta.get(
                        "center_chunk_id", meta.get("chunk_id")
                    ),
                    "chunk_ids": meta.get("chunk_ids"),
                }
            )
        return refs

    def run(self, user_input: str) -> str:
        trace_id = self.logger.new_trace_id()
        self.memory.add_user_message(user_input, trace_id=trace_id)
        self.memory.maybe_summarize(self.planner.llm, trace_id=trace_id)

        if self.settings.mode == "react":
            return self.reactor.run(self.memory, self.planner.llm, trace_id=trace_id)

        if getattr(self.settings, "enable_router", True):
            route = self.router.classify(user_input, trace_id=trace_id)
            self.logger.info("router.route", extra={"trace_id": trace_id, "route": route})
            try:
                conf = float((route or {}).get("confidence", 1.0))
            except Exception:
                conf = 1.0
            th = float(getattr(self.settings, "router_confidence_threshold", 0.65))
            if (route or {}).get("source") == "llm" and conf < th:
                self.logger.info(
                    "router.low_confidence",
                    extra={"trace_id": trace_id, "confidence": conf, "threshold": th, "route": route},
                )
                return self._pipeline_tool_call_answer(user_input, trace_id=trace_id)
            pipeline = (route or {}).get("pipeline")
            if pipeline == "extract_compute":
                return self._pipeline_extract_compute(user_input, trace_id=trace_id)
            if pipeline == "tool_call_answer":
                return self._pipeline_tool_call_answer(user_input, trace_id=trace_id)
            if pipeline == "tool_then_compute":
                return self._pipeline_tool_then_compute(user_input, trace_id=trace_id)
            if pipeline == "rewrite_retrieval_answer":
                return self._pipeline_rewrite_retrieval_answer(user_input, trace_id=trace_id)
            if pipeline == "retrieval_then_compute":
                return self._pipeline_retrieval_then_compute(user_input, trace_id=trace_id)
            if pipeline == "direct_generate":
                return self._pipeline_direct_generate(user_input, trace_id=trace_id)

        return self._run_planner_loop(user_input, trace_id=trace_id)
