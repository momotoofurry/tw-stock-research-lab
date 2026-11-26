"""趨勢分析與策略體檢。不連網。"""
import numpy as np
import pandas as pd
import pytest

from src import checkup, oracles, trend
from src.strategies import rules


def _series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_regression_trend_labels():
    up = _series(100 * np.exp(np.linspace(0, 0.3, 60)))
    rng = np.random.default_rng(0)
    flat = _series(100 + rng.normal(0, 1, 60))
    r = trend.regression_trend(up, 60)
    assert r["label"] == "上升" and r["r2"] > 0.99 and r["slope"] == pytest.approx(np.exp(0.3) - 1, rel=1e-6)
    assert trend.regression_trend(flat, 60)["label"] == "盤整"
    assert trend.regression_trend(up.iloc[::-1].reset_index(drop=True).set_axis(up.index), 60)["label"] == "下降"


def test_adx_higher_for_trend_than_noise():
    n = 200
    c_up = _series(100 + np.arange(n) * 0.8)
    rng = np.random.default_rng(1)
    c_noise = _series(100 + rng.normal(0, 1, n).cumsum() * 0.1 + rng.normal(0, 1, n))
    a_up = trend.adx(c_up + 0.5, c_up - 0.5, c_up)["adx"].iloc[-1]
    a_noise = trend.adx(c_noise + 0.5, c_noise - 0.5, c_noise)["adx"].iloc[-1]
    assert a_up > 40 and a_noise < a_up


def test_swing_levels_do_not_use_future():
    px = oracles.synthetic_prices(400, seed=4)
    h, l = px.set_index("date")["high"], px.set_index("date")["low"]
    full = trend.swing_levels(h, l, float(px["close"].iloc[300]), lookback=250)
    # 截斷在第 300 天：用截斷資料算出的轉折點，不能依賴第 300 天之後的價格
    cut = trend.swing_levels(h.iloc[:301], l.iloc[:301], float(px["close"].iloc[300]), lookback=250)
    again = trend.swing_levels(h.iloc[:301].copy(), l.iloc[:301].copy(), float(px["close"].iloc[300]), lookback=250)
    assert cut == again
    assert all(v > px["close"].iloc[300] for v in cut["resistance"])
    assert all(v < px["close"].iloc[300] for v in cut["support"])
    assert isinstance(full, dict)


def test_character_detects_momentum_and_reversion():
    rng = np.random.default_rng(2)
    n = 300
    e = rng.normal(0, 0.02, n)
    mom, rev = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        mom[t] = 0.5 * mom[t - 1] + e[t]
        rev[t] = -0.5 * rev[t - 1] + e[t]
    idx = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    to_daily = lambda r: pd.Series(100 * np.exp(np.cumsum(r)), index=idx)  # noqa: E731
    assert trend.character(to_daily(mom))["label"] == "趨勢延續型"
    assert trend.character(to_daily(rev))["label"] == "均值回歸型"


@pytest.mark.parametrize("fn", [rules.trend_ma, rules.breakout, rules.rsi_reversion])
def test_rule_signals_have_no_lookahead(fn):
    r = oracles.check_feature_no_lookahead(lambda p: fn(p).to_frame())
    assert r.passed, r.detail


def test_breakout_enters_on_new_high_and_exits_on_new_low():
    close = [100.0] * 60 + [110.0] + [111.0] * 5 + [90.0] + [90.0] * 3
    dates = pd.bdate_range("2024-01-01", periods=len(close))
    p = pd.DataFrame({"date": dates, "open": close, "high": close, "low": close, "close": close})
    s = rules.breakout(p, entry=55, exit_=20)
    assert s.iloc[59] == 0 and s.iloc[60] == 1 and s.iloc[65] == 1 and s.iloc[66] == 0


def test_checkup_runs_and_reports(monkeypatch):
    prices = oracles.synthetic_prices(1200, seed=8, start="2018-01-01")
    prices["stock_id"] = "X"
    events = pd.DataFrame(columns=["date", "before_price", "after_price", "kind"])
    monkeypatch.setattr(checkup, "load_stock", lambda sid, s, e, cfg: (prices[prices["date"] <= e].reset_index(drop=True),
                                                                        events, prices[prices["date"] <= e].reset_index(drop=True)))
    from src.config import load_config
    r = checkup.run_checkup("X", load_config(), n_random=20)
    assert list(r["table"]["key"]) == ["buy_hold", "trend_ma", "breakout", "rsi_reversion"]
    assert r["end"] < pd.Timestamp(load_config()["evaluation"]["holdout_start"])   # 不看保留期
    text = "\n".join(checkup.checkup_lines(r))
    assert "不是推薦" in text and "p =" in text
