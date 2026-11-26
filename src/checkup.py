"""策略體檢：把幾種常見的策略家族放到同一檔股票的開發期回測，看哪一類歷史表現較好、是否能和運氣區分。

這是「歷史證據」，不是推薦：只用保留期之前的資料；同時比較多種策略，所以顯著門檻做 Bonferroni 修正；
過去有效不代表未來有效。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Backtester, buy_and_hold_signals
from .evaluate import effective_end, load_stock
from .metrics import compute_metrics
from .significance import random_benchmark
from .strategies.rules import FAMILIES


def run_checkup(stock_id: str, cfg: dict, start: str = "2018-01-01", n_random: int = 100) -> dict:
    end = effective_end(cfg, None, holdout=False)
    prices, events, sp = load_stock(stock_id, start, end, cfg)
    if len(prices) < 300:
        raise ValueError(f"{stock_id} 開發期資料不足（{len(prices)} 天）")
    engine = Backtester.from_config(cfg)
    tdays = cfg["backtest"]["trading_days_per_year"]
    sigs = {k: fn(sp) for k, (_, fn) in FAMILIES.items()}
    first = max(s.first_valid_index() for s in sigs.values())       # 所有策略都有訊號後才開始比
    last = prices["date"].iloc[-1]
    bh = engine.run(prices, buy_and_hold_signals(prices, first), events)
    mb = compute_metrics(bh, first, last, tdays)
    rows = [dict(key="buy_hold", 策略="買進持有（對照）", 年化=mb["cagr"], 最大回撤=mb["max_drawdown"],
                 Sharpe=mb["sharpe"], 持股比例=1.0, 來回=0, 隨機百分位=np.nan, p=np.nan)]
    alpha = 0.05 / len(FAMILIES)
    for k, (label, _) in FAMILIES.items():
        s = sigs[k].copy()
        s[s.index < first] = np.nan
        res = engine.run(prices, s, events)
        m = compute_metrics(res, first, last, tdays)
        rb = random_benchmark(engine, prices, s, m["cagr"], n_random, dividends=events, start=first, end=last,
                              trading_days=tdays)
        rows.append(dict(key=k, 策略=label, 年化=m["cagr"], 最大回撤=m["max_drawdown"], Sharpe=m["sharpe"],
                         持股比例=m["exposure"], 來回=m["round_trips"], 隨機百分位=rb["percentile"], p=rb["p_value"]))
    df = pd.DataFrame(rows)
    df[["隨機百分位", "p"]] = df[["隨機百分位", "p"]].astype(float)
    df["贏買進持有"] = df["年化"] > mb["cagr"]
    df["顯著"] = df["p"] < alpha
    return dict(stock_id=stock_id, start=first, end=last, table=df, alpha=alpha, bh=mb)


def checkup_lines(r: dict) -> list[str]:
    df = r["table"]
    L = [f"策略體檢（開發期 {r['start']:%Y-%m-%d} ~ {r['end']:%Y-%m-%d}，含成本與股利）："]
    for x in df.itertuples():
        if x.key == "buy_hold":
            L.append(f"- {x.策略}：年化 {x.年化:.1%}、最大回撤 {x.最大回撤:.1%}、Sharpe {x.Sharpe:.2f}")
        else:
            verdict = "勝過隨機且顯著" if x.顯著 else "和運氣分不出來"
            L.append(f"- {x.策略}：年化 {x.年化:.1%}（{'贏' if x.贏買進持有 else '輸'}買進持有）、最大回撤 {x.最大回撤:.1%}、"
                     f"Sharpe {x.Sharpe:.2f}、勝過 {x.隨機百分位:.0%} 的同頻率隨機進出（p = {x.p:.3f}，{verdict}）")
    sig = df[(df["key"] != "buy_hold") & df["顯著"]]
    if sig.empty:
        L.append(f"結論：沒有任何策略家族在修正後的門檻（p < {r['alpha']:.3f}）下勝過隨機進出；"
                 "以這檔股票的歷史，擇時方法的表現無法和運氣區分。")
    else:
        L.append("結論：" + "、".join(sig["策略"]) + f" 在修正後門檻（p < {r['alpha']:.3f}）下勝過隨機進出（仍是單檔、歷史結果）。")
    L.append("※ 歷史回測證據，不是推薦；過去有效不代表未來有效。")
    return L
