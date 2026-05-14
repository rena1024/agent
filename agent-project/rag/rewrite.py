import json


def rewrite_query(
    *,
    llm,
    question: str,
    facts_block: str = "",
    max_chars: int = 120,
    trace_id: str,
) -> str:
    """
    Rewrite a user question into a short retrieval query.
    Designed to be robust: if the model returns bad output, we fall back.
    """
    q = (question or "").strip()
    if not q:
        return ""

    facts = (facts_block or "").strip()
    prompt = (
        "You rewrite a user question into a short query for internal document retrieval.\n"
        "Rules:\n"
        f'- Output JSON only: {{"query":"..."}} (no markdown fences)\n'
        f"- Keep it <= {max_chars} characters\n"
        "- Preserve key entities, acronyms, and Chinese/English keywords\n"
        "- Remove polite filler words\n"
        "\n"
        f"Known facts:\n{facts if facts else '(none)'}\n"
        f"\nUser question:\n{q}\n"
        "\nJSON:"
    )

    try:
        raw = llm.chat_plain(prompt, trace_id=trace_id)
        if not isinstance(raw, str):
            raw = str(raw)
        s = raw.strip()
        # try parse whole
        obj = json.loads(s)
        out = obj.get("query", "")
        out = str(out).strip()
        if out:
            return out[:max_chars]
    except Exception:
        pass
    return q[:max_chars]
