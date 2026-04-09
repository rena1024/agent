"""Prompt templates for planner and executor."""

from typing import List


def build_planner_prompt(messages: List[dict], tools) -> str:
    """Build planner prompt that includes conversation history and tool list."""
    history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    tool_list = ", ".join(tools)
    return (
        "You are a helpful agent. Decide the next action based on the dialogue.\n"
        f"Conversation history:\n{history}\n"
        f"Available tools: [{tool_list}]\n"
        "Rules:\n"
        "- If the user asks for facts/internal docs/any nontrivial info, FIRST call the retrieval tool to fetch context, then answer.\n"
        "- Use calculator for arithmetic; use search for web info; use retrieval for internal knowledge.\n"
        "- If the answer already exists in history, answer directly with action='final'.\n"
        "- Only answer directly when the question is simple or already answered; otherwise prefer a tool.\n"
        "- Always return JSON only.\n"
        "Return only JSON with fields:\n"
        "action: only 'tool' or 'final'\n"
        "tool: tool name when action='tool'\n"
        "tool_input: object for the tool\n"
        "output: final reply to user when action='final'\n"
        "thoughts: brief reasoning\n"
        "Examples:\n"
        '{"action":"tool","tool":"retrieval","tool_input":{"query":"项目架构"},"thoughts":"先取内部资料"}\n'
        '{"action":"tool","tool":"calculator","tool_input":{"expression":"1+1"},"thoughts":"先计算"}\n'
        '{"action":"final","output":"你叫 Cara","thoughts":"历史已给出名字"}\n'
        "No extra text outside the JSON. ``` ```surround are forbidden."
    )


def build_react_prompt(messages: List[dict], tools) -> str:
    """Prompt for ReAct-style single-step think-act-observe."""
    history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    tool_list = ", ".join(tools)
    return (
        "You are a ReAct agent. Given the dialogue and observations, decide the next action.\n"
        f"Conversation history:\n{history}\n"
        f"Available tools: [{tool_list}]\n"
        "Rules:\n"
        "- If you already have enough info, return final answer with action='final'.\n"
        "- If you need to use a tool, set action='tool' and fill tool/tool_input.\n"
        "- Keep thoughts brief.\n"
        "Output JSON only with fields: action, tool, tool_input, output, thoughts.\n"
        'Example: {"action":"tool","tool":"search","tool_input":{"query":"最新天气"},"thoughts":"先查询天气"}'
    )


def build_rag_answer_prompt(messages: List[dict], context_chunks: List[dict]) -> str:
    """Construct prompt for answering with retrieved context."""
    history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    context_texts = []
    for i, chunk in enumerate(context_chunks):
        text = chunk.get("text") or ""
        source = chunk.get("metadata", {}).get("source", "unknown")
        context_texts.append(f"[{i+1}] (source: {source}) {text}")
    context_block = "\n".join(context_texts)
    return (
        "You are an assistant that must answer using the provided context.\n"
        "If the context is sufficient, answer concisely in the user's language.\n"
        "If context is missing, say you cannot find the information.\n"
        f"Conversation history:\n{history}\n"
        f"Context:\n{context_block}\n"
        "Answer:"
    )
