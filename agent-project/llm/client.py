"""LLM client using local Ollama with simple tool-call heuristics."""

from opentelemetry.metrics import obj
import json
import re
from typing import List

import ollama
import requests

from llm.schemas import Message
from config import Settings
from colorama import Fore

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate arithmetic expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "math expression"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Web search query",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieval",
            "description": "Query internal knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
]


class LLMClient:
    def __init__(self, settings: Settings, logger):
        self.model = settings.model
        self.logger = logger
        self.host = settings.ollama_host

    def heuristic(self, prompt: str, trace_id: str) -> dict:
        """
        Heuristic flow:
        1) 聚焦最近一条 user 消息做意图判断，避免被历史工具输出短路。
        2) 检测计算/搜索意图 -> 返回工具调用计划。
        3) 否则调用 Ollama 获取自然语言回复，包装成 final。
        """
        # 仅解析“Conversation history”块，避免示例行被误认为工具输出
        hist_match = re.search(
            r"Conversation history:\n(.*?)\nAvailable tools:", prompt, flags=re.S
        )
        history_text = hist_match.group(1) if hist_match else prompt

        lines = re.findall(
            r"^(user|assistant|tool):\s*(.+)$", history_text, flags=re.MULTILINE
        )
        last_role, last_content = lines[-1] if lines else ("user", prompt)

        # 如果最后一条是工具输出，直接把结果作为最终回复
        if last_role == "tool":
            cleaned = re.sub(r"^tool\s+\w+\s+output:\s*", "", last_content).strip()
            return {"action": "final", "output": cleaned or last_content}

        # 取最近一条 user 消息用于意图判定
        user_msgs = [c for r, c in lines if r == "user"]
        latest_user = user_msgs[-1] if user_msgs else prompt

        # 1) 规则判定工具调用
        expr_match = re.search(r"(\d[\d\s+\-*/().]*\d)", latest_user)
        if "计算" in latest_user or expr_match:
            return {
                "action": "tool",
                "tool": "calculator",
                "tool_input": {
                    "expression": expr_match.group(1).strip() if expr_match else "1+1"
                },
                "thoughts": "用计算器求值",
            }
        if re.search(r"搜索|查询|找", latest_user):
            return {
                "action": "tool",
                "tool": "search",
                "tool_input": {"query": latest_user},
                "thoughts": "用搜索工具获取信息",
            }

        # 2) 默认尝试搜索，作为通用外部信息获取
        if latest_user.strip():
            return {
                "action": "tool",
                "tool": "search",
                "tool_input": {"query": latest_user},
                "thoughts": "无法分类，先用搜索获取上下文",
            }

        # 3) 兜底直接模型回复
        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            stream=False,
        )
        return {"action": "final", "output": resp["message"]["content"]}

    def chat(
        self, prompt: str, trace_id: str, use_function_calling: bool = True
    ) -> dict:
        if not use_function_calling:
            return self.heuristic(prompt, trace_id)
        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            stream=False,
            format="json",
            options={"temperature": 0, "seed": 42},
        )

        msg = resp["message"]
        content_text = msg["content"]
        self.logger.info(
            "planner.response", extra={"trace_id": trace_id, "response": content_text}
        )
        obj = json.loads(content_text)
        return {"action": obj.get("action"), "output": content_text}

    def chatWithAPI(self, prompt: str, trace_id: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "")
        return {"action": "final", "output": text}
