"""技術指標特徵。

規則：第 t 列的每個特徵只能用「第 t 天收盤以前」的資料（含第 t 天的 OHLCV）。
只允許 rolling / shift(正數) / ewm 等往回看的運算；禁止 shift(負數)、center=True、全樣本標準化。
`check_feature_no_lookahead`（src/oracles.py）用截斷不變性驗證這點。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kline import candle_parts, detect_patterns, volume_indicators


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """輸入日K（date, open, high, low, close, volume），回傳以 date 為 index 的特徵表。"""
    df = prices.set_index("date").sort_index()
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"].astype(float)
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 10, 20):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (5, 20, 60):
        f[f"close_ma{n}"] = c / c.rolling(n).mean() - 1
    f["ma5_ma20"] = c.rolling(5).mean() / c.rolling(20).mean() - 1
    f["rsi_14"] = rsi(c, 14)
    f["vol_chg_1"] = v.pct_change(1).replace([np.inf, -np.inf], np.nan)
    f["vol_ma20"] = v / v.rolling(20).mean() - 1
    f["volatility_20"] = c.pct_change().rolling(20).std()
    f["hl_range"] = (h - l) / c
    f["gap"] = o / c.shift(1) - 1
    # K 線結構、量價指標、K 線型態（0/1）
    parts = candle_parts(prices)
    f = f.join(parts).join(volume_indicators(prices))
    pats = detect_patterns(prices).astype(float)
    return f.join(pats.add_prefix("pat_"))


def make_label(prices: pd.DataFrame, horizon: int = 1, hurdle: float = 0.0) -> pd.Series:
    """標籤：第 t + horizon 天收盤相對第 t 天收盤的報酬是否 > hurdle（1/0）。
    第 t 列的標籤要到 t + horizon 收盤才知道，最後 horizon 列為 NaN。

    只能當作訓練目標，絕不能當特徵。
    """
    c = prices.set_index("date").sort_index()["close"]
    nxt = c.shift(-horizon)
    return (nxt / c - 1 > hurdle).astype(float).where(nxt.notna())
