"""Prompt templates for planner and executor."""

from typing import List


def build_planner_prompt(messages: List[dict], tools, facts_block: str = "") -> str:
    """Build planner prompt that includes conversation history and tool list."""
    history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    tool_list = ", ".join(tools)
    facts_block = facts_block.strip() or "(none)"
    latest_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            latest_user = str(m.get("content", ""))
            break
    return (
        "You are a helpful agent. Decide the next action based on the dialogue.\n"
        "Known facts (trusted):\n"
        f"{facts_block}\n"
        f"Latest user message (highest priority):\n{latest_user}\n"
        f"Conversation history:\n{history}\n"
        f"Available tools: [{tool_list}]\n"
        "Rules:\n"
        "- You MUST respond to the latest user message; do not ignore it.\n"
        "- If the user adds a new constraint/correction OR expresses dissatisfaction (e.g. '你太笨了/不是这个意思/你错了'), you MUST revise your answer or ask ONE clarifying question.\n"
        "- Only call the retrieval tool when the user explicitly requests internal资料/文档/知识库，或明确要求“根据内部信息/基于文档”来回答。\n"
        "- For common-sense questions (e.g. animal legs) or general knowledge, answer directly; do NOT call retrieval.\n"
        "- If the user asks a quantitative/counting question (e.g. totals, '多少/几/一共/合计', legs count), you MUST call calculator with a pure expression; do NOT do mental arithmetic.\n"
        "- Use calculator for arithmetic; use search for web info; use retrieval for internal knowledge.\n"
        "- If you call calculator: tool_input.expression MUST be a pure math expression using digits and operators only (no words).\n"
        "- For word problems: derive the correct formula first, then call calculator with that formula.\n"
        "  Example (legs): 4 chickens + 7 rabbits => 4*2 + 7*4\n"
        "- You may reuse an answer from history ONLY if the user explicitly asks to repeat/quote/summarize the previous answer. Otherwise, do not copy any prior assistant sentence verbatim; paraphrase and adapt.\n"
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
        '{"action":"final","output":"I am rena.","thoughts":"历史已给出名字"}\n'
        "No extra text outside the JSON. ``` ```surround are forbidden."
    )


def build_react_prompt(messages: List[dict], tools, facts_block: str = "") -> str:
    """Prompt for ReAct-style single-step think-act-observe."""
    history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    tool_list = ", ".join(tools)
    facts_block = facts_block.strip() or "(none)"
    latest_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            latest_user = str(m.get("content", ""))
            break
    return (
        "You are a ReAct agent. Given the dialogue and observations, decide the next action.\n"
        "Known facts (trusted):\n"
        f"{facts_block}\n"
        f"Latest user message (highest priority):\n{latest_user}\n"
        f"Conversation history:\n{history}\n"
        f"Available tools: [{tool_list}]\n"
        "Rules:\n"
        "- You MUST respond to the latest user message; do not ignore it.\n"
        "- If the user adds a new constraint/correction OR expresses dissatisfaction, revise your approach or ask ONE clarifying question.\n"
        "- If you already have enough info, return final answer with action='final'.\n"
        "- If you need to use a tool, set action='tool' and fill tool/tool_input.\n"
        "- If you call calculator: tool_input.expression MUST be digits/operators only. For word problems, derive formula then call calculator.\n"
        "- Do not copy any prior assistant sentence verbatim unless the user explicitly asks to repeat it.\n"
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
        context_texts.append(f"[{i + 1}] (source: {source}) {text}")
    context_block = "\n".join(context_texts)
    return (
        "You are an assistant that must answer using the provided context.\n"
        "If the context is sufficient, answer concisely in the user's language.\n"
        "If context is missing, say you cannot find the information.\n"
        f"Conversation history:\n{history}\n"
        f"Context:\n{context_block}\n"
        "Answer:"
    )
