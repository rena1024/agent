from agent.memory import Memory
from rag.rewrite import rewrite_query


class StubLLM:
    def __init__(self, out: str):
        self.out = out

    def chat_plain(self, prompt: str, trace_id: str) -> str:  # noqa: ARG002
        return self.out


def test_facts_extract_name_cn():
    mem = Memory()
    mem.add_user_message("我叫 Cara。", trace_id="t")
    assert mem.facts.get("user_name") == "Cara"


def test_facts_extract_name_en():
    mem = Memory()
    mem.add_user_message("My name is Alice!", trace_id="t")
    assert mem.facts.get("user_name") == "Alice"


def test_rewrite_query_happy_path():
    llm = StubLLM('{"query":"四次挥手 流程 TIME_WAIT"}')
    q = rewrite_query(
        llm=llm,
        question="根据内部资料，说说四次挥手",
        facts_block="",
        max_chars=30,
        trace_id="t",
    )
    assert "四次挥手" in q


def test_rewrite_query_bad_json_falls_back():
    llm = StubLLM("not-json")
    q = rewrite_query(
        llm=llm, question="hello world", facts_block="", max_chars=10, trace_id="t"
    )
    assert q == "hello worl"
