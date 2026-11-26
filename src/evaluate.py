"""單檔 / 多檔共用的評估流程：抓資料 → 還原價產生訊號 → 未還原價回測（含股利）→ 開發期 / 保留期績效。

保留期（holdout）規則：預設把資料截在 holdout_start 前一天，報告裡完全看不到保留期；
加 holdout=True 才會用到保留期，且每次都寫入 reports/holdout_log.md。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from .adjust import adjust_for_dividends
from .backtest import Backtester, BacktestResult, buy_and_hold_signals
from .config import ROOT
from .fetch import fetch_events, fetch_prices
from .metrics import compute_metrics
from .strategies import candle_volume, lgbm, ma_cross

STRATEGIES = ("ma_cross", "candle_volume", "lgbm")


@dataclass
class StockRun:
    stock_id: str
    strategy: str
    prices: pd.DataFrame          # 未還原，回測成交用
    signal_prices: pd.DataFrame   # 還原權息（或未還原，依設定），產生訊號用
    dividends: pd.DataFrame
    signal: pd.Series
    folds: pd.DataFrame | None
    res: BacktestResult
    bh: BacktestResult
    sig_start: pd.Timestamp
    windows: dict                 # {"開發期": (start, end), "保留期": (start, end)}


def generate_signals(strategy: str, prices: pd.DataFrame, cfg: dict):
    p = cfg["strategies"][strategy]
    if strategy == "ma_cross":
        return ma_cross.generate_signals(prices, **p), None
    if strategy == "candle_volume":
        return candle_volume.generate_signals(prices, **p), None
    if strategy == "lgbm":
        sig, folds, _ = lgbm.generate_signals(prices, return_details=True, **p)
        return sig, folds
    raise ValueError(f"未知策略：{strategy}（可用：{', '.join(STRATEGIES)}）")


def holdout_start(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg["evaluation"]["holdout_start"])


def effective_end(cfg: dict, end: str | None, holdout: bool) -> str:
    end = end or date.today().isoformat()
    if holdout:
        return end
    cut = holdout_start(cfg) - pd.Timedelta(days=1)
    return min(pd.Timestamp(end), cut).strftime("%Y-%m-%d")


def load_stock(stock_id: str, start: str, end: str, cfg: dict):
    prices = fetch_prices(stock_id, start, end, cfg)
    divs = fetch_events(stock_id, start, end, cfg, prices=prices)
    sp = adjust_for_dividends(prices, divs) if cfg["evaluation"].get("adjust_signal_prices", True) else prices
    return prices, divs, sp


def run_stock(stock_id: str, strategy: str, start: str, end: str | None, cfg: dict,
              holdout: bool = False, engine: Backtester | None = None) -> StockRun:
    end_eff = effective_end(cfg, end, holdout)
    prices, divs, sp = load_stock(stock_id, start, end_eff, cfg)
    if prices.empty:
        raise RuntimeError(f"{stock_id} 在 {start}~{end_eff} 沒有資料")
    sig, folds = generate_signals(strategy, sp, cfg)
    sig_start = sig.first_valid_index()
    if sig_start is None:
        raise RuntimeError(f"{stock_id} 資料太短，{strategy} 沒有產生任何訊號")
    engine = engine or Backtester.from_config(cfg)
    res = engine.run(prices, sig, divs)
    bh = engine.run(prices, buy_and_hold_signals(prices, sig_start), divs)

    last = prices["date"].iloc[-1]
    hs = holdout_start(cfg)
    windows = {}
    if sig_start < hs:
        windows["開發期"] = (sig_start, min(last, hs - pd.Timedelta(days=1)))
    if holdout and last >= hs:
        windows["保留期"] = (max(hs, sig_start), last)
    if not windows:
        raise RuntimeError(f"{stock_id} 在開發期內沒有訊號（第一個訊號 {sig_start.date()}）")
    return StockRun(stock_id, strategy, prices, sp, divs, sig, folds, res, bh, sig_start, windows)


def window_metrics(run: StockRun, cfg: dict) -> dict:
    tdays = cfg["backtest"]["trading_days_per_year"]
    rf = cfg["backtest"]["risk_free_rate"]
    return {name: (compute_metrics(run.res, s, e, tdays, rf), compute_metrics(run.bh, s, e, tdays, rf))
            for name, (s, e) in run.windows.items()}


def log_holdout_view(cfg: dict, what: str) -> int:
    """每看一次保留期就記一筆；回傳累計次數。看越多次，保留期就越不「乾淨」。"""
    out = ROOT / cfg["reports_dir"] / "holdout_log.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.write_text("# 保留期查看紀錄\n\n每一列代表看過一次保留期（holdout）結果。看過之後再改策略或參數，"
                       "保留期就不再是樣本外。\n\n| 時間 | 內容 | 保留期起日 |\n|---|---|---|\n", encoding="utf-8")
    with open(out, "a", encoding="utf-8") as f:
        f.write(f"| {datetime.now():%Y-%m-%d %H:%M} | {what} | {cfg['evaluation']['holdout_start']} |\n")
    return sum(1 for line in out.read_text(encoding="utf-8").splitlines() if line.startswith("| 20"))
