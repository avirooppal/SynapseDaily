"""Offline tests: run with `python tests/test_digest.py` (no network needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import digest
from digest import Item, canon_key, format_item, merge, rank, similar, Gemini, OpenRouter, get_llm


def test_canon_key():
    assert canon_key("https://arxiv.org/pdf/2410.12345v2") == "arxiv:2410.12345"
    assert canon_key("http://arxiv.org/abs/2410.12345v1") == "arxiv:2410.12345"
    assert canon_key("https://huggingface.co/deepseek-ai/DeepSeek-V4") == "hf:deepseek-ai/deepseek-v4"
    assert canon_key("https://huggingface.co/papers/2410.12345") != "hf:papers/2410.12345"
    assert canon_key("https://www.example.com/post/?utm_source=x&id=3") == "example.com/post?id=3"


def test_merge_and_rank():
    hn = Item("DeepSeek V4 released", "https://huggingface.co/deepseek-ai/DeepSeek-V4", "news",
              sources={"HN": "https://news.ycombinator.com/item?id=1"}, score=4)
    hf = Item("deepseek-ai/DeepSeek-V4", "https://huggingface.co/deepseek-ai/DeepSeek-V4", "model",
              sources={"HF Hub": "https://huggingface.co/deepseek-ai/DeepSeek-V4"}, score=2.5)
    paper = Item("A paper", "https://arxiv.org/abs/2410.00001", "paper", "abs",
                 sources={"HF Papers": "x"}, score=1.2)
    noise = Item("Tiny arXiv thing", "https://arxiv.org/abs/2410.00002", "paper", sources={"arXiv": "y"}, score=0.3)
    merged = merge([hn, hf, paper, noise])
    assert len(merged) == 3
    top = rank(merged, {})
    assert top[0].kind == "model" and len(top[0].sources) == 2
    assert all(i.title != "Tiny arXiv thing" for i in top)          # below MIN_SCORE
    # already-delivered items are dropped
    assert rank(merged, {top[0].key: {"d": "2026-10-01", "t": top[0].title}})[0] is not top[0]


def test_similar():
    assert similar("GPT-5 launches today", "GPT 5 launches today!")
    assert not similar("GPT-5", "GPT-5.1")


def test_kind_cap():
    items = [Item(f"Paper number {i}", f"https://arxiv.org/abs/2410.1000{i}", "paper", sources={"HF Papers": "x"}, score=5 - i * 0.1)
             for i in range(8)]
    assert sum(1 for i in rank(items, {}) if i.kind == "paper") == digest.KIND_CAPS["paper"]


def test_format_escapes_html():
    it = Item("t", "https://e.com/a?x=1&y=2", sources={"HN": "https://news.ycombinator.com/item?id=1"})
    msg = format_item(1, it, {"headline": "A <b> & B", "bullets": ["x < y"], "why_it_matters": "ok"})
    assert "A &lt;b&gt; &amp; B" in msg and "x &lt; y" in msg and "&amp;y=2" in msg
    assert len(msg) <= 4096


def test_gemini_parse():
    good = {"candidates": [{"content": {"parts": [{"thought": True, "text": "hmm"},
            {"text": '{"skip": false, "headline": "h", "bullets": ["a","b","c","d","e"], "why_it_matters": "w", "importance": 7}'}]}}]}
    out = Gemini._parse(good)
    assert out["headline"] == "h" and len(out["bullets"]) == 4
    assert Gemini._parse({"candidates": []}) is None
    assert Gemini._parse({"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}) is None


def test_openrouter_parse():
    # Direct JSON
    direct = {
        "choices": [{
            "message": {
                "content": '{"skip": false, "headline": "h", "bullets": ["1", "2", "3", "4", "5"], "why_it_matters": "w", "importance": 8}'
            }
        }]
    }
    out = OpenRouter._parse(direct)
    assert out["headline"] == "h" and len(out["bullets"]) == 4 and out["importance"] == 8

    # Wrapped in markdown code fence
    fenced = {
        "choices": [{
            "message": {
                "content": '```json\n{"skip": true, "headline": "h2", "bullets": ["1"], "why_it_matters": "w", "importance": 3}\n```'
            }
        }]
    }
    out2 = OpenRouter._parse(fenced)
    assert out2["skip"] is True and out2["headline"] == "h2"

    # Invalid / empty
    assert OpenRouter._parse({"choices": [{"message": {"content": ""}}]}) is None
    assert OpenRouter._parse({"choices": []}) is None
    assert OpenRouter._parse({"choices": [{"message": {"content": "not json"}}]}) is None


def test_get_llm(monkeypatch=None):
    import os
    orig_env = os.environ.copy()
    try:
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("LLM_PROVIDER", None)

        assert get_llm() is None

        os.environ["OPENROUTER_API_KEY"] = "or-test"
        assert isinstance(get_llm(), OpenRouter)

        os.environ["GEMINI_API_KEY"] = "gem-test"
        # defaults to OpenRouter when both present unless specified
        assert isinstance(get_llm(), OpenRouter)

        os.environ["LLM_PROVIDER"] = "gemini"
        assert isinstance(get_llm(), Gemini)

        os.environ["LLM_PROVIDER"] = "openrouter"
        assert isinstance(get_llm(), OpenRouter)
    finally:
        os.environ.clear()
        os.environ.update(orig_env)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
