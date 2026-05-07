"""
Strong routing: map user requests into one of a small set of pipelines.

Task types (user-defined):
1) Quantitative: counting / computation tasks (often require structured extraction + compute)
2) Retrieval: internal knowledge base (RAG)
3) Generation: content creation / explanation without external tools
4) Tool-based: needs external tools (search, APIs, real-time info)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal


TaskType = Literal["Quantitative", "Retrieval", "Generation", "Tool-based"]


@dataclass(frozen=True)
class Route:
    task_type: TaskType
    pipeline: str
    tool: str | None = None

    def to_dict(self) -> dict:
        d = {"task_type": self.task_type, "pipeline": self.pipeline}
        if self.tool is not None:
            d["tool"] = self.tool
        return d


_RE_INTERNAL = re.compile(r"(根据|基于)?(内部资料|内部文档|知识库|面试题|面试\.txt|cpp\.txt)")
_RE_TOOL_BASED = re.compile(
    r"(今天|明天|后天|最新|现在|实时|股价|汇率|天气|新闻|最近|推荐|价格|金价|原油|比赛|展览|更新|版本)"
)

# Pure arithmetic expression (no words). Calculator supports + - * / ** and parentheses.
_RE_PURE_EXPR = re.compile(r"^\s*[-+()0-9.\s*/**^]+\s*$")

# Word-math/counting indicators.
_RE_QUANT_INDICATOR = re.compile(
    r"(多少|一共|总共|合计|分别|每个|每人|平均|还剩|至少|最多|概率|比例|折扣|单价|总价|用时|速度|距离)"
)

# Contains a number or Chinese numeric character.
_RE_HAS_NUMBER = re.compile(r"(\d|[零一二两三四五六七八九十百千万])")

_RE_QUANT_COMMONSENSE = re.compile(r"(几|多少|一共|总共|合计).*(条腿|腿|个|只|米|公里|分钟|小时)")
_RE_LEGS_WITH_COUNTS = re.compile(r"([零一二两三四五六七八九十百千万\d]+\s*只).*(条腿|腿)")


_ALLOWED: dict[str, set[str]] = {
    "Quantitative": {"extract_compute"},
    "Retrieval": {"rewrite_retrieval_answer", "retrieval_then_compute"},
    "Tool-based": {"tool_call_answer", "tool_then_compute"},
    "Generation": {"direct_generate"},
}


class Router:
    """
    Production-like routing:
    - rules: fast high-precision rules only
    - llm: LLM-only route to pipeline with confidence
    - hybrid: rules for obvious cases, else LLM route; low-confidence falls back upstream
    """

    def __init__(self, *, llm=None, settings=None):
        self.llm = llm
        self.settings = settings

    def _rules(self, text: str) -> dict | None:
        is_quant = bool(
            (_RE_QUANT_INDICATOR.search(text) and _RE_HAS_NUMBER.search(text))
            or _RE_QUANT_COMMONSENSE.search(text)
            or (_RE_LEGS_WITH_COUNTS.search(text) and "腿" in text)
        )

        if _RE_INTERNAL.search(text):
            if is_quant:
                return {
                    "task_type": "Retrieval",
                    "pipeline": "retrieval_then_compute",
                    "confidence": 1.0,
                    "source": "rules",
                }
            return {
                "task_type": "Retrieval",
                "pipeline": "rewrite_retrieval_answer",
                "confidence": 1.0,
                "source": "rules",
            }

        if _RE_TOOL_BASED.search(text):
            if is_quant:
                return {
                    "task_type": "Tool-based",
                    "pipeline": "tool_then_compute",
                    "tool": "search",
                    "confidence": 1.0,
                    "source": "rules",
                }
            return {
                "task_type": "Tool-based",
                "pipeline": "tool_call_answer",
                "tool": "search",
                "confidence": 1.0,
                "source": "rules",
            }

        if (_RE_PURE_EXPR.match(text) and _RE_HAS_NUMBER.search(text)) or is_quant:
            return {
                "task_type": "Quantitative",
                "pipeline": "extract_compute",
                "confidence": 0.95,
                "source": "rules",
            }

        return None

    def _llm_route(self, text: str, *, trace_id: str | None = None) -> dict | None:
        if not self.llm:
            return None

        prompt = (
            "你是一个路由器，把用户问题分配到一个 pipeline。\n"
            "你必须只输出 JSON，不要多余文字。\n"
            "可选 task_type 与 pipeline：\n"
            '- Quantitative: "extract_compute"\n'
            '- Retrieval: "rewrite_retrieval_answer" | "retrieval_then_compute"\n'
            '- Tool-based: "tool_call_answer" | "tool_then_compute"（tool 固定为 search）\n'
            '- Generation: "direct_generate"\n'
            "要求：\n"
            '- 输出格式：{"task_type":"...","pipeline":"...","tool":null或"search","confidence":0到1}\n'
            "- 只有当用户明确提到内部资料/内部文档/知识库/基于文档时，才选 Retrieval。\n"
            "- 只有当用户明确需要实时/最新/今天/价格/新闻/版本等外部信息时，才选 Tool-based。\n"
            "- 常识类且带计数/计算（例如多少条腿/总共多少/合计）优先选 Quantitative。\n"
            "- 不确定就选 Generation 并把 confidence 设低一些。\n"
            f"用户问题：{text}\n"
            "输出："
        )
        try:
            raw = self.llm.chat_plain(prompt, trace_id=trace_id or "router")
            data = json.loads(str(raw))
        except Exception:
            return None

        task_type = str((data or {}).get("task_type", "") or "")
        pipeline = str((data or {}).get("pipeline", "") or "")
        tool = (data or {}).get("tool", None)
        try:
            confidence = float((data or {}).get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        if task_type not in _ALLOWED:
            return None
        if pipeline not in _ALLOWED[task_type]:
            return None
        if task_type == "Tool-based":
            tool = "search"
        else:
            tool = None

        if confidence < 0:
            confidence = 0.0
        if confidence > 1:
            confidence = 1.0

        return {
            "task_type": task_type,
            "pipeline": pipeline,
            "tool": tool,
            "confidence": confidence,
            "source": "llm",
        }

    def classify(self, user_input: str, *, trace_id: str | None = None) -> dict:
        text = (user_input or "").strip()
        if not text:
            return {
                "task_type": "Generation",
                "pipeline": "direct_generate",
                "confidence": 1.0,
                "source": "rules",
            }

        mode = getattr(self.settings, "router_mode", "hybrid") if self.settings else "hybrid"

        if mode in ("rules", "hybrid"):
            r = self._rules(text)
            if r is not None:
                return r

        if mode in ("llm", "hybrid"):
            if not self.llm:
                return {
                    "task_type": "Tool-based",
                    "pipeline": "tool_call_answer",
                    "tool": "search",
                    "confidence": 0.0,
                    "source": "fallback",
                    "reason": "llm_not_configured",
                }
            try:
                r = self._llm_route(text, trace_id=trace_id)
            except Exception:
                r = None
            if r is not None:
                return r
            return {
                "task_type": "Tool-based",
                "pipeline": "tool_call_answer",
                "tool": "search",
                "confidence": 0.0,
                "source": "fallback",
                "reason": "llm_parse_or_validate_error",
            }

        return {
            "task_type": "Tool-based",
            "pipeline": "tool_call_answer",
            "tool": "search",
            "confidence": 0.0,
            "source": "fallback",
            "reason": "rules_no_match",
        }
