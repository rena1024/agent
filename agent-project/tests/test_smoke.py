from tools.search import Search


def test_search_reports_missing_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    search = Search()
    result = search.run({"query": "hello"}, trace_id="t1")
    assert result["status"] == "error"
    assert "TAVILY_API_KEY" in result["output"]
