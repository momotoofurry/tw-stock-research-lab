"""顯著性檢定：策略 vs「同樣換手率與持股比例的隨機訊號」。

隨機訊號用兩狀態馬可夫鏈產生，進場機率 / 出場機率從策略自己的訊號估計，
所以隨機策略的平均持股比例和平均持有天數跟策略差不多，付的成本也差不多。
策略如果真的有選時能力，績效應該排在隨機分布的前段（例如前 5%）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .metrics import compute_metrics


def transition_probs(signal: pd.Series) -> tuple[float, float, float]:
    """回傳 (p_enter = P(0→1), p_exit = P(1→0), 初始持有機率)。"""
    s = signal.dropna().astype(int).to_numpy()
    prev, nxt = s[:-1], s[1:]
    n0, n1 = (prev == 0).sum(), (prev == 1).sum()
    p_enter = ((prev == 0) & (nxt == 1)).sum() / n0 if n0 else 0.0
    p_exit = ((prev == 1) & (nxt == 0)).sum() / n1 if n1 else 0.0
    return float(p_enter), float(p_exit), float(s.mean()) if len(s) else 0.0


def matched_random_signals(signal: pd.Series, rng: np.random.Generator) -> pd.Series:
    p_enter, p_exit, p0 = transition_probs(signal)
    mask = signal.notna().to_numpy()
    out = np.full(len(signal), np.nan)
    state = int(rng.random() < p0)
    u = rng.random(len(signal))
    for t in np.flatnonzero(mask):
        out[t] = state
        state = int(u[t] < p_enter) if state == 0 else int(u[t] >= p_exit)
    return pd.Series(out, index=signal.index)


def random_benchmark(engine, prices: pd.DataFrame, signal: pd.Series, strategy_cagr: float, n: int = 200,
                     seed: int = 0, dividends=None, start=None, end=None, trading_days: int = 252) -> dict:
    """回傳策略 CAGR 在 n 組隨機訊號中的百分位與單尾 p 值（隨機 >= 策略的比例）。"""
    rng = np.random.default_rng(seed)
    cagrs = np.array([
        compute_metrics(engine.run(prices, matched_random_signals(signal, rng), dividends),
                        start, end, trading_days)["cagr"]
        for _ in range(n)
    ])
    return dict(
        percentile=float((cagrs < strategy_cagr).mean()),
        p_value=float((1 + (cagrs >= strategy_cagr).sum()) / (n + 1)),
        random_median=float(np.median(cagrs)),
        random_p95=float(np.quantile(cagrs, 0.95)),
        n=n,
    )


def sign_test_p(wins: int, n: int) -> float:
    """單尾符號檢定：若勝負各半，n 檔裡至少贏 wins 檔的機率。"""
    if n == 0:
        return float("nan")
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n
