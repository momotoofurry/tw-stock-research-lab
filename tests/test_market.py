"""大盤概況：指數狀態、連續買賣超、beta、描述文字。不連網。"""
import numpy as np
import pandas as pd
import pytest

from src import market, oracles


def test_index_state_uses_only_past_and_basic_values():
    px = oracles.synthetic_prices(600, seed=11)
    full = market.index_state(px)
    cut = market.index_state(px.iloc[:400])
    again = market.index_state(px.iloc[:400].copy())
    assert cut == again                                          # 只取決於輸入
    c = px["close"]
    assert full["close"] == c.iloc[-1]
    assert full["ret"][20] == pytest.approx(c.iloc[-1] / c.iloc[-21] - 1)
    assert full["dist"][60] == pytest.approx(c.iloc[-1] / c.iloc[-60:].mean() - 1)
    assert full["dd_high"] <= 0 and 0 <= full["vol_pct"] <= 1


def test_streak():
    assert market._streak(pd.Series([1, -2, 3, 4, 5])) == 3
    assert market._streak(pd.Series([5, -1, -2])) == -2
    assert market._streak(pd.Series([1, 0])) == 0


def test_beta_recovers_known_sensitivity():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=300)
    ir = rng.normal(0, 0.01, 300)
    ipx = pd.DataFrame({"date": idx, "close": 100 * np.cumprod(1 + ir)})
    stock_r = pd.Series(1.5 * ir + rng.normal(0, 0.002, 300), index=idx)
    stock_r.iloc[0] = np.nan
    assert market.beta_to_index(stock_r, ipx) == pytest.approx(1.5, abs=0.1)


def test_notes_and_short_lines():
    px = oracles.synthetic_prices(600, seed=12)
    st = market.index_state(px)
    st["flows"] = {"inst": {"外資": {1: 160.0, 5: -1510.0, 20: -8777.0}}, "inst_date": "2026-10-28",
                   "foreign_streak": 1, "margin": {"balance": 6323.0, "chg1": 89.1, "chg20": 0.043}}
    st["breadth"] = {20: 0.62, 60: 0.72, "n": 40}
    st["sectors"] = pd.DataFrame([{"類股": "金融", "20日相對大盤": 0.064}, {"類股": "科技", "20日相對大盤": -0.012},
                                  {"類股": "半導體", "20日相對大盤": -0.029}])
    text = "\n".join(market.notes(st))
    assert "外資 +160 億" in text and "62% 在月線上" in text and "最強 金融 +6.4%" in text and "最弱 半導體 -2.9%" in text
    assert "外資 +160 億" in market.short_lines(st)[1]
