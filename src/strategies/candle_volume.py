"""K 線型態 + 量能確認策略（規則型）。

第 t 天收盤後判斷：
- 進場：多方型態（多頭吞噬 / 錘子 / 紅三兵 / 帶量突破）且收盤 > MA(trend_ma)，且當天量 > 前 5 日均量 × vol_confirm
- 出場：空方型態（空頭吞噬 / 流星 / 黑三兵 / 帶量跌破）、收盤跌破 MA(trend_ma)、或持有滿 max_hold 天
訊號只用第 t 天收盤以前的資料。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..kline import detect_patterns, volume_indicators

BULLISH = ["bull_engulfing", "hammer", "three_white_soldiers", "volume_breakout"]
BEARISH = ["bear_engulfing", "shooting_star", "three_black_crows", "volume_breakdown"]


def generate_signals(prices: pd.DataFrame, trend_ma: int = 20, vol_confirm: float = 1.5,
                     max_hold: int = 20) -> pd.Series:
    c = prices.set_index("date").sort_index()["close"]
    pats = detect_patterns(prices)
    vol = volume_indicators(prices)
    ma = c.rolling(trend_ma).mean()
    uptrend = c > ma
    entry = (pats[BULLISH].any(axis=1) & uptrend & (vol["vol_ratio_5"] > vol_confirm - 1)).to_numpy()
    exit_ = (pats[BEARISH].any(axis=1) | (c < ma)).to_numpy()

    pos = np.zeros(len(c))
    held = 0
    for t in range(len(c)):
        if held > 0:
            held += 1
            if exit_[t] or held > max_hold:
                held = 0
        elif entry[t]:
            held = 1
        pos[t] = 1.0 if held > 0 else 0.0
    return pd.Series(pos, index=c.index, name="signal").where(ma.notna())
