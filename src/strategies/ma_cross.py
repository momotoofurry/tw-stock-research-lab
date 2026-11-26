"""均線交叉（基準策略）：第 t 天收盤後，短均線 > 長均線 → 持有，否則空手。"""
from __future__ import annotations

import pandas as pd


def generate_signals(prices: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
    c = prices.set_index("date").sort_index()["close"]
    ma_f, ma_s = c.rolling(fast).mean(), c.rolling(slow).mean()
    sig = (ma_f > ma_s).astype(float)
    return sig.where(ma_s.notna()).rename("signal")
