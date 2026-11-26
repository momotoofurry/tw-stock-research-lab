"""多檔研究：同一個策略跑整個股票池，和買進持有、和隨機訊號比，判斷是不是普遍有效。

結論看三件事：
1. 贏買進持有的檔數（符號檢定 p 值）
2. 顯著勝過隨機訊號（p < 0.05）的檔數——純運氣下約 5% 的檔數會「顯著」
3. 風險面：最大回撤比買進持有小的檔數
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import Backtester  # noqa: E402
from .config import ROOT, load_config  # noqa: E402
from .evaluate import log_holdout_view, run_stock, window_metrics  # noqa: E402
from .fetch import FinMindRateLimitError  # noqa: E402
from .significance import random_benchmark, sign_test_p  # noqa: E402

BAR_COLOR = "#2a78d6"


def _plot_excess(path: Path, title: str, df: pd.DataFrame) -> None:
    d = df.sort_values("excess_cagr")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.24 * len(d) + 1)))
    ax.barh(d["label"], d["excess_cagr"] * 100, color=BAR_COLOR, height=0.7)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_xlabel("Strategy CAGR minus Buy & Hold CAGR (percentage points)")
    ax.set_title(title, loc="left", fontsize=12)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, axis="x", color="#e6e6e6", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_study(strategy: str, start: str = "2018-01-01", end: str | None = None, cfg: dict | None = None,
              holdout: bool = False, stocks: list[str] | None = None, n_random: int | None = None,
              tag: str = "") -> Path:
    cfg = cfg or load_config()
    ev = cfg["evaluation"]
    universe = ev["universe"]
    stocks = stocks or list(universe)
    n_random = ev["n_random"] if n_random is None else n_random
    engine = Backtester.from_config(cfg)
    tdays = cfg["backtest"]["trading_days_per_year"]

    rows, errors = [], []
    for k, sid in enumerate(stocks, 1):
        print(f"[study] ({k}/{len(stocks)}) {sid} {universe.get(sid, '')}")
        try:
            run = run_stock(sid, strategy, start, end, cfg, holdout, engine)
        except FinMindRateLimitError as e:
            errors.append(f"{sid}：{e}（已抓到的資料有快取，稍後重跑會接續）")
            print(f"[study] 限流，停止：{e}")
            break
        except Exception as e:  # 單檔失敗（資料太短等）不影響其他檔，記錄在報告
            errors.append(f"{sid}：{e}")
            continue
        for window, (m, mb) in window_metrics(run, cfg).items():
            s, e = run.windows[window]
            st = random_benchmark(engine, run.prices, run.signal, m["cagr"], n_random,
                                  dividends=run.dividends, start=s, end=e, trading_days=tdays)
            rows.append(dict(window=window, stock_id=sid, name=universe.get(sid, ""), start=m["start"],
                             end=m["end"], cagr=m["cagr"], bh_cagr=mb["cagr"], excess_cagr=m["cagr"] - mb["cagr"],
                             mdd=m["max_drawdown"], bh_mdd=mb["max_drawdown"], sharpe=m["sharpe"],
                             bh_sharpe=mb["sharpe"], exposure=m["exposure"], round_trips=m["round_trips"],
                             costs=m["total_commission"] + m["total_tax"], dividends=m["total_dividends"],
                             percentile=st["percentile"], p_value=st["p_value"]))
    if not rows:
        raise RuntimeError("沒有任何一檔跑成功：" + "；".join(errors))
    df = pd.DataFrame(rows)
    df["label"] = df["stock_id"] + " " + df["name"]
    views = log_holdout_view(cfg, f"study {strategy} ({len(stocks)} 檔)") if holdout else None

    out_dir = ROOT / cfg["reports_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"study_{strategy}{'_' + tag if tag else ''}_{date.today():%Y%m%d}" + ("_holdout" if holdout else "")
    p = cfg["strategies"][strategy]
    lines = [
        f"# 多檔研究：{strategy}" + ("（含保留期）" if holdout else ""),
        "",
        "> 僅供研究與模擬，不構成投資建議；本工具不含任何下單功能。",
        "",
        f"- 股票池：{len(stocks)} 檔（config.yaml `evaluation.universe`），成功 {df['stock_id'].nunique()} 檔",
        f"- 策略參數：`{p}`",
        f"- 保留期：{ev['holdout_start']} 起。" + (
            f"**本報告包含保留期，這是第 {views} 次查看保留期。**" if holdout else "本報告看不到保留期。"),
        "- 買進持有與策略都含股利；訊號用還原價、成交用未還原價；成本、漲跌停、流動性規則同單檔回測",
        f"- 顯著性：每檔和 {n_random} 組同持股比例 / 換手率的隨機訊號比較",
        "- **存活者偏差**：股票池都是現在仍上市的公司，已下市的股票不在裡面，所有結果都偏樂觀",
        "",
    ]
    for window, g in df.groupby("window", sort=False):
        n = len(g)
        wins = int((g["excess_cagr"] > 0).sum())
        sig = int((g["p_value"] < 0.05).sum())
        lower_mdd = int((g["mdd"] > g["bh_mdd"]).sum())
        better_sharpe = int((g["sharpe"] > g["bh_sharpe"]).sum())
        png = out_dir / f"{stem}_{'dev' if window == '開發期' else 'holdout'}.png"
        _plot_excess(png, f"{strategy}: excess CAGR vs Buy & Hold ({'dev' if window == '開發期' else 'holdout'})", g)
        lines += [
            f"## {window}",
            "",
            "| 項目 | 結果 |",
            "|---|---|",
            f"| 年化報酬贏買進持有 | {wins} / {n} 檔（符號檢定 p = {sign_test_p(wins, n):.3f}） |",
            f"| 年化超額報酬中位數 | {g['excess_cagr'].median():+.1%}（平均 {g['excess_cagr'].mean():+.1%}） |",
            f"| 顯著勝過隨機訊號（p < 0.05） | {sig} / {n} 檔（純運氣預期約 {0.05 * n:.1f} 檔） |",
            f"| 勝過隨機訊號的平均百分位 | {g['percentile'].mean():.0%}（無選時能力時約 50%） |",
            f"| 最大回撤比買進持有小 | {lower_mdd} / {n} 檔 |",
            f"| Sharpe 比買進持有高 | {better_sharpe} / {n} 檔 |",
            f"| 平均持股比例 / 平均來回次數 | {g['exposure'].mean():.0%} / {g['round_trips'].mean():.0f} |",
            "",
            f"![excess]({png.name})",
            "",
            "| 股票 | 期間 | 策略年化 | 買進持有年化 | 超額 | 策略 MDD | 買進持有 MDD | Sharpe（策略 / 買持） "
            "| 持股比例 | 來回 | 成本 | 隨機百分位 | p 值 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in g.sort_values("excess_cagr", ascending=False).itertuples():
            lines.append(
                f"| {r.label} | {r.start.date()} ~ {r.end.date()} | {r.cagr:.1%} | {r.bh_cagr:.1%} | "
                f"{r.excess_cagr:+.1%} | {r.mdd:.1%} | {r.bh_mdd:.1%} | {r.sharpe:.2f} / {r.bh_sharpe:.2f} | "
                f"{r.exposure:.0%} | {r.round_trips} | {r.costs:,.0f} | {r.percentile:.0%} | {r.p_value:.3f} |")
        lines.append("")
    if errors:
        lines += ["## 未完成的股票", ""] + [f"- {e}" for e in errors] + [""]
    md = out_dir / f"{stem}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    df.drop(columns="label").to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    print(f"[study] {md}")
    return md

