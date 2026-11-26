"""Oracle 檢查：用已知答案驗證回測引擎、特徵與 walk-forward 本身沒有問題。

每個檢查回傳 OracleResult。tests/test_oracles.py 會對正確版本與「故意埋錯」的版本執行，
報告（src/report.py）也會在實際資料上執行並列出結果。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import Backtester, Costs, buy_and_hold_signals
from .metrics import compute_metrics


@dataclass
class OracleResult:
    name: str
    passed: bool
    detail: str


def synthetic_prices(n: int = 1500, drift: float = 0.0006, vol: float = 0.015, seed: int = 0,
                     start: str = "2015-01-05") -> pd.DataFrame:
    """幾何隨機漫步日K，用於離線測試。漲跌幅限制在 ±9% 內以免觸發漲跌停。"""
    rng = np.random.default_rng(seed)
    r = np.clip(rng.normal(drift, vol, n), -0.09, 0.09)
    close = 100 * np.exp(np.cumsum(r))
    prev = np.concatenate([[100.0], close[:-1]])
    open_ = prev * np.exp(np.clip(rng.normal(0, vol / 3, n), -0.03, 0.03))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n), "stock_id": "SYN",
        "open": open_.round(2), "high": high.round(2), "low": low.round(2), "close": close.round(2),
        "volume": rng.integers(1_000_000, 5_000_000, n), "spread": np.nan,
    })


def cheat_signals(prices: pd.DataFrame, costs: Costs) -> pd.Series:
    """作弊策略：偷看「明天」的報酬（第 t+1 天開盤買進、t+2 天開盤賣出的報酬）。正常引擎下應好得離譜。"""
    df = prices.sort_values("date").set_index("date")
    fwd = df["open"].shift(-2) / df["open"].shift(-1) - 1  # 刻意的未來資料
    hurdle = 2 * costs.commission_rate * costs.commission_discount + costs.sell_tax_rate
    return (fwd > hurdle).astype(float).where(fwd.notna())


def random_signals(prices: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(0, 2, len(prices)).astype(float), index=pd.DatetimeIndex(prices["date"]))


# ---------- 1. 作弊模型必須好得離譜 ----------
def check_cheat_detected(engine_cls=Backtester, prices: pd.DataFrame | None = None,
                         costs: Costs | None = None, initial_cash: float = 1_000_000,
                         lot_mode: str = "board", min_cagr: float = 1.0) -> OracleResult:
    prices = synthetic_prices() if prices is None else prices
    costs = costs or Costs()
    res = engine_cls(costs, initial_cash, lot_mode).run(prices, cheat_signals(prices, costs))
    cagr = compute_metrics(res)["cagr"]
    ok = cagr > min_cagr
    return OracleResult("作弊模型（偷看明日報酬）年化 > 100%", ok,
                        f"作弊策略年化報酬 {cagr:.1%}（門檻 {min_cagr:.0%}）")


# ---------- 2a. 成交時點：t 收盤訊號 → t+1 開盤成交 ----------
def check_execution_timing(engine_cls=Backtester) -> OracleResult:
    dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({
        "date": dates, "open": [100, 101, 102, 103, 104, 105], "high": [110] * 6, "low": [95] * 6,
        "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5], "volume": [1000] * 6, "spread": np.nan,
    })
    sig = pd.Series([np.nan, 1, np.nan, 0, np.nan, np.nan], index=dates, dtype=float)
    tr = engine_cls(Costs(), 1_000_000, "board").run(prices, sig).trades
    expected = [(dates[2], "buy", 102.0), (dates[4], "sell", 104.0)]
    got = [(r.date, r.side, float(r.price)) for r in tr.itertuples()]
    ok = got == expected
    return OracleResult("成交時點：第 t 天訊號於第 t+1 天開盤成交", ok,
                        f"預期 {[(str(d.date()), s, p) for d, s, p in expected]}，"
                        f"實際 {[(str(d.date()), s, p) for d, s, p in got]}")


# ---------- 3. 手算成本 ----------
HAND_CASES = [
    # (lot_mode, initial_cash, 預期期末現金, 說明)
    ("board", 200_000, 200_000 - (100_000 + 142) + (110_000 - 156 - 330),
     "整張：100 元買 1 張，手續費 floor(142.5)=142；110 元賣，手續費 floor(156.75)=156、證交稅 330"),
    ("odd", 1_000, 1_000 - (900 + 20) + (990 - 20 - 2),
     "零股：100 元買 9 股，手續費 1.28→最低 20；110 元賣，手續費最低 20、證交稅 floor(2.97)=2"),
]


def check_cost_hand_calc(engine_cls=Backtester) -> OracleResult:
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({
        "date": dates, "open": [100.0, 100.0, 110.0], "high": [101, 106, 111], "low": [99, 99, 109],
        "close": [100.0, 105.0, 110.0], "volume": [1000] * 3, "spread": np.nan,
    })
    sig = pd.Series([1.0, 0.0, np.nan], index=dates)
    details, ok = [], True
    for lot_mode, cash0, expected, desc in HAND_CASES:
        res = engine_cls(Costs(), cash0, lot_mode).run(prices, sig)
        got = float(res.equity.iloc[-1])
        match = abs(got - expected) < 1e-9
        ok &= match
        details.append(f"{desc}：預期 {expected:,}，引擎 {got:,.2f} {'OK' if match else 'MISMATCH'}")
    return OracleResult("手算小交易成本 = 引擎結果", ok, "；".join(details))


# ---------- 4. 隨機訊號扣成本後明顯輸給買進持有 ----------
def check_random_vs_bh(engine_cls=Backtester, prices: pd.DataFrame | None = None, costs: Costs | None = None,
                       initial_cash: float = 1_000_000, lot_mode: str = "board",
                       n_seeds: int = 20, margin: float = 0.05) -> OracleResult:
    prices = synthetic_prices() if prices is None else prices
    costs = costs or Costs()
    eng = engine_cls(costs, initial_cash, lot_mode)
    bh = compute_metrics(eng.run(prices, buy_and_hold_signals(prices)))["cagr"]
    rnd = [compute_metrics(eng.run(prices, random_signals(prices, s)))["cagr"] for s in range(n_seeds)]
    mean_rnd, best_rnd = float(np.mean(rnd)), float(np.max(rnd))
    ok = mean_rnd < bh - margin and best_rnd < bh
    return OracleResult("隨機訊號扣成本後年化明顯低於買進持有", ok,
                        f"買進持有年化 {bh:.1%}；隨機訊號 {n_seeds} 次平均 {mean_rnd:.1%}、最佳 {best_rnd:.1%}"
                        f"（要求平均低於買進持有 {margin:.0%} 以上，且最佳也低於）")


# ---------- 2c. 特徵不可偷看未來：截斷不變性 ----------
def check_feature_no_lookahead(feature_fn, prices: pd.DataFrame | None = None, n_cuts: int = 8) -> OracleResult:
    """第 t 列特徵只能用到 t 收盤以前的資料 ⇒ 把 t 之後的資料砍掉，第 t 列特徵不能變。"""
    prices = (synthetic_prices(400) if prices is None else prices).sort_values("date").reset_index(drop=True)
    full = feature_fn(prices)
    cuts = np.linspace(100, len(prices) - 2, n_cuts).astype(int)
    bad = []
    for c in cuts:
        part = feature_fn(prices.iloc[: c + 1])
        d = prices["date"].iloc[c]
        a, b = part.loc[d], full.loc[d]
        same = np.isclose(a.values.astype(float), b.values.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True)
        if not same.all():
            bad.append(f"{d.date()}: {list(a.index[~same])}")
    ok = not bad
    return OracleResult("特徵截斷不變性（無 look-ahead）", ok,
                        f"檢查 {len(cuts)} 個截斷點，" + ("全部一致" if ok else "不一致：" + "; ".join(bad[:3])))


# ---------- 5. walk-forward：訓練最後日期 < 測試第一天，且中間有 gap ----------
def check_walk_forward(fold_info: pd.DataFrame, trading_dates: pd.DatetimeIndex, gap: int,
                       horizon: int = 1) -> OracleResult:
    bad = [] if gap >= horizon else [f"gap {gap} < 標籤 horizon {horizon}：訓練標籤會跨進測試期"]
    pos = {d: i for i, d in enumerate(pd.DatetimeIndex(trading_dates))}
    for r in fold_info.itertuples():
        if not r.train_end < r.test_start:
            bad.append(f"fold {r.fold}: train_end {r.train_end.date()} >= test_start {r.test_start.date()}")
        elif pos[r.test_start] - pos[r.train_end] - 1 < gap:
            bad.append(f"fold {r.fold}: 間隔 {pos[r.test_start] - pos[r.train_end] - 1} 日 < gap {gap}")
    ok = not bad and len(fold_info) > 0
    return OracleResult("Walk-forward：每折訓練最後日期 < 測試第一天（含 gap）", ok,
                        f"{len(fold_info)} 折，" + ("全部符合" if ok else "; ".join(bad[:3]) or "沒有任何一折"))


def run_all(prices: pd.DataFrame, costs: Costs, initial_cash: float, lot_mode: str, feature_fn,
            fold_info: pd.DataFrame | None = None, gap: int | None = None,
            horizon: int = 1) -> list[OracleResult]:
    """在實際資料上跑全部 oracle（給報告用）。"""
    out = [
        check_cheat_detected(Backtester, prices, costs, initial_cash, lot_mode),
        check_execution_timing(Backtester),
        check_cost_hand_calc(Backtester),
        check_random_vs_bh(Backtester, prices, costs, initial_cash, lot_mode),
        check_feature_no_lookahead(feature_fn, prices),
    ]
    if fold_info is not None:
        out.append(check_walk_forward(fold_info, pd.DatetimeIndex(prices["date"]), gap, horizon))
    return out
