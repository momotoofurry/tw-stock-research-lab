"""決策日誌：合成資料、不連網。"""
import pandas as pd
import pytest

from src import decisions as dc
from src.portfolio import TRADE_COLUMNS


def _book(closes, start="2026-01-01", adj_factor=None):
    d = pd.bdate_range(start, periods=len(closes))
    raw = pd.DataFrame({"date": d, "close": closes})
    adj = raw.copy()
    if adj_factor is not None:  # 向前還原：之後的價格放大
        adj["close"] = adj["close"] * adj_factor
    return raw, None, adj


@pytest.fixture
def cfg(tmp_path):
    return dict(data_dir=str(tmp_path / "data"), reports_dir=str(tmp_path / "reports"))


def _trade(cfg, rows):
    p = dc.trades_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=TRADE_COLUMNS).to_csv(p, index=False)


def _patch(monkeypatch, books):
    monkeypatch.setattr(dc, "_load", lambda sid, start, cfg: books[sid])


def test_empty_fields_write_nothing(cfg):
    assert dc.record_decision(cfg, "2026-01-02", "buy", "1111", 1000, 10.0) is False
    assert not dc.decisions_path(cfg).exists()


def test_append_only(cfg):
    assert dc.record_decision(cfg, "2026-01-02", "buy", "1111", 1000, 10.0, thesis="甲, 乙\n丙")
    assert dc.record_decision(cfg, "2026-01-05", "sell", "1111", 1000, 12.0, horizon_months=6)
    df = pd.read_csv(dc.decisions_path(cfg), dtype={"stock_id": str}, keep_default_na=False)
    assert len(df) == 2 and list(df.columns) == dc.DECISION_COLUMNS
    assert df.loc[0, "thesis"] == "甲, 乙\n丙" and df.loc[0, "stock_id"] == "1111"


def test_buy_paired_with_first_later_sell_and_excess(cfg, monkeypatch):
    # 1111：10 → 12 後賣出（第 3 個交易日），之後續漲到 20；0050：100 → 110
    _patch(monkeypatch, {"1111": _book([10, 11, 12, 15, 20]), "0050": _book([100, 102, 110, 115, 120])})
    dc.record_decision(cfg, "2026-01-01", "buy", "1111", 1000, 10.0, thesis="T", horizon_months=3, exit_if="E")
    _trade(cfg, [["2026-01-01", "1111", "buy", 1000, 10, 20, 0, None, ""],
                 ["2026-01-05", "1111", "sell", 1000, 13, 20, 36, 1900, ""],
                 ["2026-01-06", "1111", "sell", 500, 15, 20, 20, 0, ""]])
    r = dc.review(cfg, "2026-02-01")
    x = r["rows"].iloc[0]
    assert x["closed"] and x["end"] == pd.Timestamp("2026-01-05")
    # 終點用實際成交價 13（收盤 12），不是收盤
    assert x["ret"] == pytest.approx(0.3) and x["bench"] == pytest.approx(0.1)
    assert x["excess"] == pytest.approx(0.2) and x["held_days"] == 4
    text = dc.review_text(r)
    assert "已賣出的買進決策終點用該筆賣出的實際成交價" in text
    assert "T" in text and "E" in text and "預計持有約 3 個月" in text


def test_open_position_runs_to_latest_and_total_return_uses_adjusted(cfg, monkeypatch):
    # 還原價在第 3 天起放大 1.1（除息），實際成交價 10 對應還原座標 10 × 1.0
    adj = [1.0, 1.0, 1.1, 1.1]
    _patch(monkeypatch, {"1111": _book([10, 10, 10, 10], adj_factor=adj), "0050": _book([100] * 4)})
    dc.record_decision(cfg, "2026-01-01", "buy", "1111", 1000, 10.0, thesis="T")
    _trade(cfg, [["2026-01-01", "1111", "buy", 1000, 10, 20, 0, None, ""]])
    x = dc.review(cfg, "2026-02-01")["rows"].iloc[0]
    assert not x["closed"] and x["ret"] == pytest.approx(0.1) and x["bench"] == pytest.approx(0.0)


def test_closed_end_price_converted_to_adjusted_coordinates(cfg, monkeypatch):
    # 還原係數 1.0, 1.0, 1.1：賣出日（第 3 天）收盤 10、成交價 11 → 終點 11 × 1.1 / 10 = 12.1，起點 10
    _patch(monkeypatch, {"1111": _book([10, 10, 10, 10], adj_factor=[1.0, 1.0, 1.1, 1.1]), "0050": _book([100] * 4)})
    dc.record_decision(cfg, "2026-01-01", "buy", "1111", 1000, 10.0, thesis="T")
    _trade(cfg, [["2026-01-05", "1111", "sell", 1000, 11, 20, 36, 0, ""]])
    x = dc.review(cfg, "2026-02-01")["rows"].iloc[0]
    assert x["closed"] and x["ret"] == pytest.approx(0.21)


def test_sell_decision_reports_after_sale(cfg, monkeypatch):
    _patch(monkeypatch, {"1111": _book([10, 12, 18]), "0050": _book([100, 100, 105])})
    dc.record_decision(cfg, "2026-01-02", "sell", "1111", 1000, 12.0, thesis="賣出理由")
    r = dc.review(cfg, "2026-02-01")
    x = r["rows"].iloc[0]
    assert x["ret"] == pytest.approx(0.5) and x["bench"] == pytest.approx(0.05)
    assert "賣出之後發生了什麼" in dc.review_text(r) and r["summary"]["n_buy"] == 0


def test_small_sample_notice_and_no_advice_words(cfg, monkeypatch):
    _patch(monkeypatch, {"1111": _book([10, 12]), "0050": _book([100, 101])})
    dc.record_decision(cfg, "2026-01-01", "buy", "1111", 1000, 10.0, thesis="T")
    path, r = dc.save_review(cfg, "2026-02-02")
    text = path.read_text(encoding="utf-8")
    assert path.name == "decisions_20260202.md"
    assert text.splitlines()[2].startswith("> 只描述你自己的紀錄，不是投資建議")
    assert "樣本太少，不足以下結論" in text
    assert r["summary"]["win_rate"] == 1.0
    body = text.replace("不是投資建議", "")
    for w in ("建議", "應該買", "應該賣", "值得"):
        assert w not in body


def test_missing_data_noted_not_crash(cfg, monkeypatch):
    def boom(sid, start, cfg):
        raise ValueError("x")
    monkeypatch.setattr(dc, "_load", boom)
    dc.record_decision(cfg, "2026-01-01", "buy", "1111", 1000, 10.0, thesis="T")
    r = dc.review(cfg, "2026-02-01")
    assert "資料不足" in r["rows"].iloc[0]["note"] and r["summary"]["n_scored"] == 0


# ---- 到期提醒 ----
def _due_setup(cfg, monkeypatch, horizon=3, thesis="AI 需求, 帶動\n營收", exit_if="毛利率低於 40%"):
    _patch(monkeypatch, {"1111": _book([10, 11, 12, 15, 20]), "0050": _book([100, 102, 110, 115, 120])})
    dc.record_decision(cfg, "2026-01-01", "buy", "1111", 1000, 10.0, thesis=thesis, horizon_months=horizon,
                       exit_if=exit_if)
    _trade(cfg, [["2026-01-01", "1111", "buy", 1000, 10, 20, 0, None, ""]])


def test_due_not_yet(cfg, monkeypatch):
    _due_setup(cfg, monkeypatch)
    assert dc.due_reminders(cfg, "2026-03-31") == ""
    assert not dc.reminders_path(cfg).exists()


def test_due_quotes_verbatim_and_only_once(cfg, monkeypatch):
    _due_setup(cfg, monkeypatch)
    t = dc.due_reminders(cfg, "2026-04-01")
    assert "1111" in t and "2026-01-01" in t and "預計持有 3 個月" in t and "已持有 90 天" in t
    assert "AI 需求, 帶動\n營收" in t and "毛利率低於 40%" in t           # 原文照抄
    assert "+100.0%" in t and "+20.0%" in t                                # 至今含息報酬 vs 0050
    assert t.rstrip().endswith("回頭看看當初的理由和失效條件，是否仍符合現在的情況")
    assert dc.due_reminders(cfg, "2026-04-02") == ""                       # 第二次不再提醒
    assert "1111" in dc.reminders_path(cfg).read_text(encoding="utf-8")


def test_due_print_only_does_not_write_state(cfg, monkeypatch):
    _due_setup(cfg, monkeypatch)
    assert dc.due_reminders(cfg, "2026-04-01", write_state=False)
    assert not dc.reminders_path(cfg).exists()
    assert dc.due_reminders(cfg, "2026-04-01")                              # 仍然會提醒


def test_due_sold_not_reminded(cfg, monkeypatch):
    _due_setup(cfg, monkeypatch)
    _trade(cfg, [["2026-01-01", "1111", "buy", 1000, 10, 20, 0, None, ""],
                 ["2026-01-05", "1111", "sell", 1000, 13, 20, 36, 1900, ""]])
    assert dc.due_reminders(cfg, "2026-06-01") == ""


def test_due_requires_horizon_and_buy(cfg, monkeypatch):
    _patch(monkeypatch, {"1111": _book([10, 11]), "0050": _book([100, 102])})
    dc.record_decision(cfg, "2026-01-01", "buy", "1111", 1000, 10.0, thesis="只有論點")
    dc.record_decision(cfg, "2026-01-01", "sell", "2222", 1000, 10.0, horizon_months=1)
    assert dc.due_reminders(cfg, "2027-01-01") == ""
