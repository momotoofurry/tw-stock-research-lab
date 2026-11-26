"""除權息還原（向前還原），只用來產生訊號與特徵；回測成交一律用未還原價格。

向前還原：除權息日（含）之後的價格乘上 before_price / after_price，除權息日之前的價格維持原值。
所以第 t 天以前的還原價只取決於 t 以前發生的除權息，未來的除權息完全不會改動過去的數值
（向後還原會把過去整段等比例縮放，浮點誤差會讓相等的值變得不相等，破壞截斷不變性）。
特徵都是比例型（報酬、均線比、型態），向前或向後還原結果相同。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adjust_for_dividends(prices: pd.DataFrame, dividends: pd.DataFrame | None) -> pd.DataFrame:
    df = prices.sort_values("date").reset_index(drop=True).copy()
    if dividends is None or dividends.empty or df.empty:
        return df
    dates = df["date"].to_numpy()
    factor = np.ones(len(df))
    events = dividends[(dividends["date"] > df["date"].iloc[0]) & (dividends["date"] <= df["date"].iloc[-1])]
    for d, before, after in events[["date", "before_price", "after_price"]].itertuples(index=False):
        factor[dates >= np.datetime64(d)] *= before / after
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * factor
    if "spread" in df:
        df["spread"] = np.nan  # 還原後的 spread 沒有意義
    return df
