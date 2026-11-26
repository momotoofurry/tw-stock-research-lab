import csv

import pytest

from src import llm_summary as L

SRC = "2026-11-26 虛構公司 9999 收盤 1,234.5，漲 2.5%，成交量 12,000 張。"


def make_cfg(tmp_path, **kw):
    llm = {"enabled": True, "host": "http://127.0.0.1:11434", "model": "m", "timeout_seconds": 5,
           "max_chars": 300, "log_path": str(tmp_path / "llm_log.csv")}
    llm.update(kw)
    return {"llm": llm}


def fake(monkeypatch, text):
    monkeypatch.setattr(L, "_post_chat", lambda c, s: text)


def test_disabled(tmp_path):
    cfg = make_cfg(tmp_path, enabled=False)
    assert L.summarize(cfg, SRC) == (None, "disabled")
    assert not (tmp_path / "llm_log.csv").exists()


@pytest.mark.parametrize("host", ["http://example.com:11434", "http://192.168.1.5:11434", "https://api.openai.com"])
def test_non_local_host_rejected(tmp_path, host):
    called = []
    import requests
    orig = requests.post
    requests.post = lambda *a, **k: called.append(1)
    try:
        s, why = L.summarize(make_cfg(tmp_path, host=host), SRC)
    finally:
        requests.post = orig
    assert s is None and why.startswith("rejected") and not called


@pytest.mark.parametrize("host", ["http://127.0.0.1:11434", "localhost:11434", "http://[::1]:11434"])
def test_local_hosts_allowed(host):
    assert L._check_host(host)


def test_ok(tmp_path, monkeypatch):
    fake(monkeypatch, "9999 收盤 1234.5，漲 2.5，成交量 12000 張。")
    s, why = L.summarize(make_cfg(tmp_path), SRC)
    assert s and why == "ok"


def test_new_number_blocked(tmp_path, monkeypatch):
    fake(monkeypatch, "9999 收盤 1,234.5，漲 3.5%。")
    s, why = L.summarize(make_cfg(tmp_path), SRC)
    assert s is None and why.startswith("numbers_mismatch") and "3.5" in why


def test_number_formats():
    assert L.check_numbers("價格 1,234", "價格 1234")[0]
    assert L.check_numbers("價格 1234", "價格 1,234")[0]
    assert L.check_numbers("漲 12.3%", "漲 12.3")[0]
    assert L.check_numbers("漲 +12.3%", "漲 12.3%")[0]
    assert L.check_numbers("跌 -1.5%", "下跌 1.5%")[0]
    assert L.check_numbers("１２３", "123")[0]          # 全形
    assert L.check_numbers("漲 12.30", "漲 12.3")[0]
    ok, bad = L.check_numbers("漲 12.3%", "漲 12.4%")
    assert not ok and bad == ["12.4"]


def test_date_numbers():
    assert L.check_numbers("2026-11-26 收盤", "11月26日收盤")[0]
    ok, bad = L.check_numbers("2026-11-26 收盤", "11月27日收盤")
    assert not ok and bad == ["27"]


@pytest.mark.parametrize("word", L.BANNED_TERMS)
def test_banned_terms(tmp_path, monkeypatch, word):
    fake(monkeypatch, f"9999 {word}。")
    s, why = L.summarize(make_cfg(tmp_path), SRC + f" {word}")   # 原文有也不行
    assert s is None and why.startswith("banned_terms")


def test_think_stripped(tmp_path, monkeypatch):
    fake(monkeypatch, "<think>\n推論 77 之類\n</think>\n9999 漲 2.5%。")
    s, why = L.summarize(make_cfg(tmp_path), SRC)
    assert s == "9999 漲 2.5%。" and why == "ok"
    assert L.strip_think("想法 99</think>結果") == "結果"


def test_too_long(tmp_path, monkeypatch):
    fake(monkeypatch, "9999" + "字" * 400)
    s, why = L.summarize(make_cfg(tmp_path, max_chars=300), SRC)
    assert s is None and why.startswith("too_long")


def test_empty(tmp_path, monkeypatch):
    fake(monkeypatch, "<think>x</think>")
    assert L.summarize(make_cfg(tmp_path), SRC)[1] == "empty"


@pytest.mark.parametrize("exc", ["conn", "timeout", "bad_json"])
def test_errors_do_not_raise(tmp_path, monkeypatch, exc):
    import requests

    def boom(*a, **k):
        if exc == "conn":
            raise requests.ConnectionError("x")
        if exc == "timeout":
            raise requests.Timeout("x")

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"nope": 1}
        return R()
    monkeypatch.setattr(requests, "post", boom)
    s, why = L.summarize(make_cfg(tmp_path), SRC)
    assert s is None and why.startswith("error")


def test_log_has_no_content(tmp_path, monkeypatch):
    fake(monkeypatch, "9999 漲 2.5%。")
    L.summarize(make_cfg(tmp_path), SRC)
    fake(monkeypatch, "9999 漲 8.8%。")
    L.summarize(make_cfg(tmp_path), SRC)
    text = (tmp_path / "llm_log.csv").read_text(encoding="utf-8")
    assert "虛構公司" not in text and "1,234" not in text and "2.5%" not in text and "8.8" not in text
    rows = list(csv.DictReader(text.splitlines()))
    assert len(rows) == 2 and list(rows[0].keys()) == L.LOG_FIELDS
    assert rows[0]["ok"] == "1" and rows[1]["ok"] == "0" and rows[1]["reason"] == "numbers_mismatch"
