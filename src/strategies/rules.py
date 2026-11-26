"""常見的規則型策略家族（用來做「策略體檢」：看哪一類方法在這檔股票上歷史表現如何）。

訊號都在第 t 天收盤後依 t 以前的資料決定（1 = 持有、0 = 空手），t+1 開盤由回測引擎成交。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import rsi


def _state_machine(enter: np.ndarray, exit_: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(len(enter), np.nan)
    pos = 0
    for t in range(len(enter)):
        if not valid[t]:
            continue
        if pos == 0 and enter[t]:
            pos = 1
        elif pos == 1 and exit_[t]:
            pos = 0
        out[t] = pos
    return out


def trend_ma(prices: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.Series:
    """趨勢追蹤：月線在季線之上持有。"""
    c = prices.set_index("date").sort_index()["close"]
    f, s = c.rolling(fast).mean(), c.rolling(slow).mean()
    return (f > s).astype(float).where(s.notna()).rename("signal")


def breakout(prices: pd.DataFrame, entry: int = 55, exit_: int = 20) -> pd.Series:
    """突破（Donchian）：收盤突破前 entry 日最高價進場、跌破前 exit_ 日最低價出場。"""
    p = prices.set_index("date").sort_index()
    hi = p["high"].rolling(entry).max().shift(1)
    lo = p["low"].rolling(exit_).min().shift(1)
    sig = _state_machine((p["close"] > hi).to_numpy(), (p["close"] < lo).to_numpy(), hi.notna().to_numpy())
    return pd.Series(sig, index=p.index, name="signal")


def rsi_reversion(prices: pd.DataFrame, buy: float = 30, sell: float = 55, n: int = 14) -> pd.Series:
    """均值回歸：RSI 跌破 buy 進場、回到 sell 以上出場。"""
    c = prices.set_index("date").sort_index()["close"]
    r = rsi(c, n)
    sig = _state_machine((r < buy).to_numpy(), (r > sell).to_numpy(), r.notna().to_numpy())
    return pd.Series(sig, index=c.index, name="signal")


FAMILIES = {
    "trend_ma": ("趨勢追蹤（月線 > 季線持有）", trend_ma),
    "breakout": ("突破（55 日新高進、20 日新低出）", breakout),
    "rsi_reversion": ("均值回歸（RSI < 30 進、> 55 出）", rsi_reversion),
}
