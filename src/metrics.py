"""績效指標。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def _between(s: pd.Series, start, end) -> pd.Series:
    m = pd.Series(True, index=s.index)
    if start is not None:
        m &= s >= pd.Timestamp(start)
    if end is not None:
        m &= s <= pd.Timestamp(end)
    return m


def compute_metrics(res: BacktestResult, start=None, end=None, trading_days: int = 252, rf: float = 0.0) -> dict:
    """[start, end] 期間的績效。期初部位沿用回測當時的狀態（不在期初重新買進）。

    成本 / 股利 / 交易次數只算期間內發生的；來回交易以「出場日」落在期間內為準。
    """
    idx = res.equity.index
    eq = res.equity[_between(idx.to_series(), start, end).to_numpy()]
    total = float(eq.iloc[-1] / eq.iloc[0] - 1)
    n = max(len(eq) - 1, 1)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (trading_days / n) - 1) if eq.iloc[-1] > 0 else -1.0
    daily = eq.pct_change().dropna()
    excess = daily - rf / trading_days
    sharpe = float(excess.mean() / excess.std() * np.sqrt(trading_days)) if excess.std() > 0 else 0.0
    lo, hi = eq.index[0], eq.index[-1]
    tr = res.trades[_between(res.trades["date"], lo, hi)]
    rt = res.round_trips[_between(res.round_trips["exit_date"], lo, hi)]
    sk = res.skipped[_between(res.skipped["date"], lo, hi)]
    dv = res.dividends[_between(res.dividends["date"], lo, hi)]
    win = float((rt["pnl"] > 0).mean()) if len(rt) else float("nan")
    return dict(
        start=lo, end=hi, final_equity=float(eq.iloc[-1]),
        total_return=total, cagr=cagr, max_drawdown=max_drawdown(eq), sharpe=sharpe,
        win_rate=win, round_trips=len(rt), fills=len(tr),
        total_commission=float(tr["commission"].sum()), total_tax=float(tr["tax"].sum()),
        total_dividends=float(dv["amount"].sum()),
        skipped_limit=int(sk["reason"].str.contains("停").sum()),
        skipped_cash=int(sk["reason"].str.startswith("資金不足").sum()),
        skipped_liquidity=int(sk["reason"].str.startswith("流動性不足").sum()),
        exposure=float(_exposure(res, lo, hi)),
    )


def _exposure(res: BacktestResult, lo, hi) -> float:
    """期間內有持股的交易日比例。"""
    eq = res.equity
    state = pd.Series((res.trades["side"] == "buy").astype(float).to_numpy(), index=res.trades["date"])
    state = state.groupby(level=0).last()   # 多檔回測同一天可能有多筆成交（此時只是近似值）
    held = state.reindex(eq.index).ffill().fillna(0.0)
    h = held[(held.index >= lo) & (held.index <= hi)]
    return float(h.mean()) if len(h) else 0.0
