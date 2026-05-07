from agent.router import Router
from config import Settings


class DummyLLM:
    def __init__(self, payload: str):
        self.payload = payload

    def chat_plain(self, prompt: str, trace_id: str):
        return self.payload


def test_router_rules_internal_short_circuit():
    r = Router(settings=Settings(router_mode="hybrid"), llm=DummyLLM("{}"))
    out = r.classify("根据内部资料总结一下", trace_id="t1")
    assert out["task_type"] == "Retrieval"
    assert out["pipeline"] == "rewrite_retrieval_answer"
    assert out["source"] == "rules"


def test_router_llm_used_when_rules_no_match():
    llm = DummyLLM(
        '{"task_type":"Generation","pipeline":"direct_generate","tool":null,"confidence":0.9}'
    )
    r = Router(settings=Settings(router_mode="hybrid"), llm=llm)
    out = r.classify("给我写一段介绍Python的文字", trace_id="t1")
    assert out["source"] == "llm"
    assert out["pipeline"] == "direct_generate"


def test_router_llm_invalid_payload_falls_back():
    llm = DummyLLM("not json")
    r = Router(settings=Settings(router_mode="llm"), llm=llm)
    out = r.classify("随便问点啥", trace_id="t1")
    assert out["pipeline"] == "tool_call_answer"
    assert out["source"] == "fallback"
    assert out["reason"] == "llm_parse_or_validate_error"


def test_router_rules_mode_no_match_has_reason():
    r = Router(settings=Settings(router_mode="rules"), llm=None)
    out = r.classify("随便问点啥", trace_id="t1")
    assert out["source"] == "fallback"
    assert out["reason"] == "rules_no_match"
    assert out["pipeline"] == "tool_call_answer"


def test_router_llm_mode_without_llm_has_reason():
    r = Router(settings=Settings(router_mode="llm"), llm=None)
    out = r.classify("随便问点啥", trace_id="t1")
    assert out["source"] == "fallback"
    assert out["reason"] == "llm_not_configured"
    assert out["pipeline"] == "tool_call_answer"
