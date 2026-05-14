from agent.memory import Memory


def test_add_tool_message_compact_stores_refs_only():
    mem = Memory()
    mem.add_user_message("hi", trace_id="t")
    mem.add_tool_message_compact(
        content='{"query":"q","hits":[{"source":"x","center_chunk_id":1,"chunk_ids":[0,1,2]}]}',
        tool_name="retrieval",
        trace_id="t",
    )

    tool_msgs = [m for m in mem.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "tool retrieval refs:" in tool_msgs[0]["content"]
    assert "center_chunk_id" in tool_msgs[0]["content"]
