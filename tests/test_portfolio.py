"""持股監控：規則、情境試算、風險、讀寫。全部用人工構造的部位，不連網。"""
import json

import numpy as np
import pandas as pd
import pytest

from src import portfolio as pfm


def _pf(**defaults):
    pf = json.loads(json.dumps(pfm.DEFAULT_PORTFOLIO))
    pf["defaults"].update(defaults)
    pf["holdings"] = [dict(stock_id="A", shares=1000, avg_cost=100.0), dict(stock_id="B", shares=2000, avg_cost=50.0)]
    return pf


def _book(close_a=90.0, close_b=60.0, dd_a=-0.2, dd_b=-0.02, cash=0.0):
    pos = pd.DataFrame([
        dict(stock_id="A", name="甲", industry="X", shares=1000, avg_cost=100.0, close=close_a, day_chg=0.0,
             market_value=1000 * close_a, dd_from_high=dd_a, high_basis="近一年", ma20=95.0, ma60=88.0,
             ma120=np.nan, ma240=np.nan),
        dict(stock_id="B", name="乙", industry="Y", shares=2000, avg_cost=50.0, close=close_b, day_chg=0.0,
             market_value=2000 * close_b, dd_from_high=dd_b, high_basis="近一年", ma20=58.0, ma60=61.0,
             ma120=np.nan, ma240=np.nan),
    ])
    total = pos["market_value"].sum() + cash
    pos["weight"] = pos["market_value"] / total
    return pfm.Book(pos, cash, total, pd.DataFrame(), {})


def _alerts(pf, book):
    a = pfm.evaluate_rules(pf, book)
    return {(r.stock_id, r.規則): r.狀態 for r in a.itertuples()}


def test_no_rules_no_alerts():
    assert pfm.evaluate_rules(_pf(), _book()).empty
    assert not pfm._has_rules(_pf())


def test_stop_loss_pct_triggers_and_sign_is_normalized():
    for v in (-0.05, 0.05):  # 使用者輸入 -5% 或 5% 都代表「低於成本 5%」
        a = _alerts(_pf(stop_loss_pct=v), _book(close_a=94.0))
        assert a[("A", "停損價")] == "觸發"           # 94 < 95
        assert ("B", "停損價") not in a               # 60 遠高於 47.5


def test_near_threshold():
    a = _alerts(_pf(stop_loss_pct=-0.05, near_pct=0.03), _book(close_a=97.0))
    assert a[("A", "停損價")] == "接近"               # 97 / 95 - 1 = 2.1% < 3%
    assert ("A", "停損價") not in _alerts(_pf(stop_loss_pct=-0.05, near_pct=0.01), _book(close_a=97.0))


def test_per_holding_price_overrides_default():
    pf = _pf(stop_loss_pct=-0.5)
    pf["holdings"][0]["stop_loss"] = 92.0
    assert _alerts(pf, _book(close_a=91.0))[("A", "停損價")] == "觸發"


def test_take_profit_trailing_ma_and_weight():
    a = _alerts(_pf(take_profit_pct=0.15, trailing_stop_pct=0.15, ma_break=60, max_weight=0.5), _book())
    assert a[("B", "停利價")] == "觸發"               # 60 >= 57.5
    assert a[("A", "移動停損")] == "觸發"             # 回落 20% > 15%
    assert ("B", "移動停損") not in a
    assert a[("B", "跌破 MA60")] == "觸發"            # 60 < 61
    assert a[("A", "跌破 MA60")] == "接近"            # 90 / 88 - 1 = 2.3% < 3%
    assert a[("B", "單檔權重上限")] == "觸發"         # 120k / 210k = 57%


def test_portfolio_drawdown_rule():
    pf = _pf(portfolio_drawdown_pct=0.10)
    a = pfm.evaluate_rules(pf, _book(), risk={"dd_now": -0.12})
    assert list(a["規則"]) == ["組合回撤"] and a["狀態"].iloc[0] == "觸發"


def test_scenario_math():
    book = _book(cash=90_000)                         # 股票 90k + 120k，現金 90k，總 300k
    tbl, chg, pct = pfm.scenario(book, {"A": -0.2})
    assert chg == pytest.approx(-18_000) and pct == pytest.approx(-0.06)
    tbl, chg, pct = pfm.scenario(book, {"A": -0.1, "B": 0.1})
    assert chg == pytest.approx(-9_000 + 12_000)


def test_risk_summary_concentration_and_var():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=600)
    rets = pd.DataFrame({"A": rng.normal(0, 0.02, 600), "B": rng.normal(0, 0.01, 600)}, index=idx)
    book = _book()
    book.returns = rets
    r = pfm.risk_summary(book)
    w = np.array([90_000, 120_000]) / 210_000
    assert r["effective_n"] == pytest.approx(1 / (w ** 2).sum())
    assert r["max_weight_id"] == "B" and r["var95_1d"] > 0
    assert r["worst5"] < 0 and r["worst20"] < 0
    assert set(r["industry"].index) == {"X", "Y"}


def test_save_load_roundtrip(tmp_path):
    pf = _pf(trailing_stop_pct=-0.15, ma_break=60)
    pf["holdings"][0]["buy_date"] = "2025-06-02"
    p = tmp_path / "portfolio.yaml"
    pfm.save_portfolio(pf, p)
    back = pfm.load_portfolio(p)
    assert back["defaults"]["trailing_stop_pct"] == -0.15 and back["defaults"]["ma_break"] == 60
    assert back["holdings"][0]["stock_id"] == "A" and back["holdings"][0]["buy_date"] == "2025-06-02"
    assert pfm.load_portfolio(tmp_path / "missing.yaml")["holdings"] == []


def test_brief_without_holdings_asks_to_fill_in():
    assert "尚未輸入持股" in pfm.daily_brief(_pf() | {"holdings": []}, {})


def test_record_trade_weighted_cost_and_realized(tmp_path):
    cfg = {"data_dir": str(tmp_path), "costs": dict(commission_rate=0.001425, commission_discount=1.0,
                                                    min_commission=20, sell_tax_rate=0.003, price_limit=0.1)}
    pf = json.loads(json.dumps(pfm.DEFAULT_PORTFOLIO))
    pfm.record_trade(pf, cfg, "buy", "2317", 2000, 450, "2026-10-01")
    r = pfm.record_trade(pf, cfg, "buy", "2317", 1000, 480, "2026-10-05")
    h = r["holding"]
    assert h["shares"] == 3000 and h["avg_cost"] == pytest.approx(460) and h["buy_date"] == "2026-10-01"
    r = pfm.record_trade(pf, cfg, "sell", "2317", 1000, 500, "2026-10-20")
    fee, tax = int(500_000 * 0.001425), int(500_000 * 0.003)
    assert r["trade"]["realized_pnl"] == pytest.approx(40 * 1000 - fee - tax)
    assert r["holding"]["shares"] == 2000 and r["holding"]["avg_cost"] == pytest.approx(460)
    with pytest.raises(ValueError):
        pfm.record_trade(pf, cfg, "sell", "2317", 5000, 500)
    r = pfm.record_trade(pf, cfg, "sell", "2317", 2000, 440, "2026-10-21")
    assert r["holding"]["shares"] == 0 and r["holding"]["buy_date"] is None
    log = pd.read_csv(pfm.trades_path(cfg), dtype={"stock_id": str})
    assert list(log["side"]) == ["buy", "buy", "sell", "sell"]
