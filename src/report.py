"""產生績效報告（markdown + 圖），並和「買進持有」比較、列出跑過的 oracle 檢查。"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import oracles  # noqa: E402
from .backtest import Backtester  # noqa: E402
from .config import ROOT, load_config  # noqa: E402
from .features import build_features  # noqa: E402
from .adjust import adjust_for_dividends  # noqa: E402
from .evaluate import holdout_start, log_holdout_view, run_stock, window_metrics  # noqa: E402
from .fetch import fetch_prices  # noqa: E402
from .significance import random_benchmark  # noqa: E402
from .kline import PATTERNS, detect_patterns, pattern_stats  # noqa: E402

STRATEGY_COLOR, BH_COLOR = "#2a78d6", "#eb6834"
UP_COLOR, DOWN_COLOR = "#d9363e", "#1f9d55"   # 台股慣例：紅漲綠跌
MA_COLORS = {5: "#2a78d6", 20: "#eb6834"}


def _plot(path: Path, title: str, eq: pd.Series, bh: pd.Series, holdout_at=None) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    for s, label, color in ((eq, "Strategy", STRATEGY_COLOR), (bh, "Buy & Hold", BH_COLOR)):
        norm = s / s.iloc[0]
        ax1.plot(norm.index, norm.values, color=color, lw=2, label=label)
        ax1.annotate(f"{label} {norm.iloc[-1]:.2f}x", (norm.index[-1], norm.iloc[-1]), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=9, color="#333333")
        dd = s / s.cummax() - 1
        ax2.plot(dd.index, dd.values * 100, color=color, lw=1.5, label=label)
    if holdout_at is not None:
        for ax in (ax1, ax2):
            ax.axvline(holdout_at, color="#888888", lw=1, ls="--")
        ax1.annotate("holdout", (holdout_at, 1), xytext=(4, 4), textcoords="offset points", fontsize=9,
                     color="#555555", xycoords=("data", "axes fraction"), va="top")
    ax1.set_title(title, loc="left", fontsize=12)
    ax1.set_ylabel("Equity (start = 1.0)")
    ax1.legend(loc="upper left", frameon=False)
    ax2.set_ylabel("Drawdown (%)")
    for ax in (ax1, ax2):
        ax.grid(True, color="#e6e6e6", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_kline(path: Path, title: str, prices: pd.DataFrame, trades: pd.DataFrame | None = None,
                days: int = 120) -> None:
    """近 days 個交易日的 K 線 + 均線 + 成交量（下方獨立面板），標出模擬買賣點。"""
    df = prices.sort_values("date").reset_index(drop=True)
    ma = {n: df["close"].rolling(n).mean() for n in MA_COLORS}
    df = df.iloc[-days:].reset_index(drop=True)
    x = range(len(df))
    up = df["close"] >= df["open"]
    colors = [UP_COLOR if u else DOWN_COLOR for u in up]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.vlines(x, df["low"], df["high"], colors=colors, lw=1)
    bottom = df[["open", "close"]].min(axis=1)
    height = (df["close"] - df["open"]).abs().clip(lower=df["close"] * 0.0005)
    ax1.bar(x, height, bottom=bottom, color=colors, width=0.7, edgecolor=colors, linewidth=0.5)
    for n, color in MA_COLORS.items():
        ax1.plot(x, ma[n].iloc[-days:].values, color=color, lw=1.5, label=f"MA{n}")
    if trades is not None and len(trades):
        pos = {d: i for i, d in enumerate(df["date"])}
        span = df["high"].max() - df["low"].min()
        for t in trades.itertuples():
            if t.date in pos:
                i = pos[t.date]
                if t.side == "buy":
                    ax1.plot(i, df["low"][i] - span * 0.03, marker="^", color="#222222", ms=8, ls="")
                else:
                    ax1.plot(i, df["high"][i] + span * 0.03, marker="v", color="#222222", ms=8, ls="")
        ax1.plot([], [], marker="^", color="#222222", ls="", label="Buy (sim)")
        ax1.plot([], [], marker="v", color="#222222", ls="", label="Sell (sim)")
    ax2.bar(x, df["volume"] / 1000, color=colors, width=0.7)
    ax2.plot(x, (df["volume"].rolling(20, min_periods=1).mean() / 1000).values, color="#555555", lw=1.2,
             label="Vol MA20")
    ax1.set_title(title, loc="left", fontsize=12)
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", frameon=False, ncol=4, fontsize=9)
    ax2.set_ylabel("Volume (lots)")
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    step = max(len(df) // 8, 1)
    ax2.set_xticks(list(x)[::step])
    ax2.set_xticklabels([d.strftime("%Y-%m-%d") for d in df["date"][::step]], fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(True, color="#e6e6e6", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _kline_section(prices: pd.DataFrame, horizon: int = 5, recent: int = 10, stats_before=None) -> list[str]:
    stats_prices = prices if stats_before is None else prices[prices["date"] < pd.Timestamp(stats_before)]
    stats = pattern_stats(stats_prices, horizon)
    pats = detect_patterns(prices)
    vol = build_features(prices)[["vol_ratio_5", "vol_ma5_ma20", "obv_slope_20", "close_vwap20", "mfi_14"]]
    lines = [
        "## K 線型態與量價分析",
        "",
        f"型態出現後「隔日開盤買、{horizon} 個交易日後開盤賣」的歷史統計（未扣成本；事後統計，僅供研究，"
        f"樣本少的型態不具參考性；統計期間 {stats_prices['date'].iloc[0].date()} ~ {stats_prices['date'].iloc[-1].date()}）。注意：統計期間涵蓋整段回測期，若依這張表回頭挑選策略型態或參數，"
        "就是資料窺探，回測結果會過度樂觀；要調參請改用回測期之前的資料或另留 holdout 期間：",
        "",
        "| 型態 | 說明 | 次數 | 平均報酬 | 上漲比例 | 相對全部交易日 |",
        "|---|---|---|---|---|---|",
    ]
    for r in stats.itertuples():
        if r.次數 == 0:
            lines.append(f"| {r.pattern} | {r.說明} | 0 | — | — | — |")
        else:
            lines.append(f"| {r.pattern} | {r.說明} | {r.次數} | {r.平均報酬:.2%} | {r.上漲比例:.1%} | {r.超額報酬:+.2%} |")
    last = pats.iloc[-recent:]
    lines += ["", f"### 最近 {recent} 個交易日", "",
              "| 日期 | 型態 | 量 / 前5日均量 | 5日均量 / 20日均量 | OBV 斜率(20) | 收盤 / VWAP20 | MFI(14) |",
              "|---|---|---|---|---|---|---|"]
    for d, row in last.iterrows():
        names = "、".join(PATTERNS[k].split("（")[0] for k, v in row.items() if v) or "—"
        v = vol.loc[d]
        lines.append(f"| {d.date()} | {names} | {v.vol_ratio_5 + 1:.2f}x | {v.vol_ma5_ma20 + 1:.2f}x | "
                     f"{v.obv_slope_20:+.2f} | {v.close_vwap20:+.2%} | {v.mfi_14:.0f} |")
    lines += ["", "量價指標說明：OBV 斜率 > 0 代表近 20 日上漲日的量較多；收盤 / VWAP20 > 0 代表收盤高於近 20 日量加權均價；"
              "MFI > 80 偏過熱、< 20 偏超賣。", ""]
    return lines


def run_kline_report(stock_id: str, start: str, end: str | None = None, cfg: dict | None = None,
                     days: int = 120) -> Path:
    """只做 K 線與量價分析（不回測）。"""
    cfg = cfg or load_config()
    end = end or date.today().isoformat()
    prices = fetch_prices(stock_id, start, end, cfg)
    if prices.empty:
        raise RuntimeError(f"{stock_id} 在 {start}~{end} 沒有資料")
    out_dir = Path(cfg["reports_dir"])
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{stock_id}_kline_{prices['date'].iloc[0]:%Y%m%d}_{prices['date'].iloc[-1]:%Y%m%d}"
    png = out_dir / f"{stem}.png"
    _plot_kline(png, f"{stock_id} K-line, last {days} trading days", prices, days=days)
    lines = [f"# {stock_id} K 線與量價分析", "", "> 僅供研究，不構成投資建議。", "",
             f"- 資料期間：{prices['date'].iloc[0].date()} ~ {prices['date'].iloc[-1].date()}（{len(prices)} 個交易日）",
             "", f"![kline]({png.name})", ""] + _kline_section(prices, stats_before=cfg["evaluation"]["holdout_start"])
    md = out_dir / f"{stem}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {md}")
    print(f"[report] {png}")
    return md


def _participation_text(engine: Backtester) -> str:
    if engine.max_volume_participation is None:
        return "不限（未設定 max_volume_participation）"
    return f"前 {engine.volume_window} 日均量 × {engine.max_volume_participation:.0%}（不含成交當天）；賣出一次全賣"


def _fmt_row(name: str, m: dict) -> str:
    wr = "—" if pd.isna(m["win_rate"]) else f"{m['win_rate']:.1%}"
    return (f"| {name} | {m['total_return']:.1%} | {m['cagr']:.1%} | {m['max_drawdown']:.1%} | {m['sharpe']:.2f} "
            f"| {wr} | {m['exposure']:.0%} | {m['round_trips']} ({m['fills']} 筆成交) | {m['total_commission']:,.0f} "
            f"| {m['total_tax']:,.0f} | {m['total_dividends']:,.0f} | {m['skipped_limit']} | {m['skipped_cash']} "
            f"| {m['final_equity']:,.0f} |")


PERF_HEADER = [
    "| 策略 | 總報酬 | 年化報酬 | 最大回撤 | Sharpe | 勝率 | 持股比例 | 交易次數（來回） | 手續費 | 證交稅 | 股利入帳 "
    "| 漲跌停跳過 | 資金不足跳過 | 期末資產 |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]


def run_report(stock_id: str, strategy: str, start: str, end: str | None = None, cfg: dict | None = None,
               lot_mode: str | None = None, holdout: bool = False) -> Path:
    cfg = cfg or load_config()
    if lot_mode:
        cfg["backtest"]["lot_mode"] = lot_mode
    run = run_stock(stock_id, strategy, start, end, cfg, holdout)
    prices, res, bh_res, engine = run.prices, run.res, run.bh, Backtester.from_config(cfg)
    wm = window_metrics(run, cfg)
    ev = cfg["evaluation"]
    sig_tests = {}
    for name, (s, e) in run.windows.items():
        sig_tests[name] = random_benchmark(engine, prices, run.signal, wm[name][0]["cagr"], ev["n_random"],
                                           dividends=run.dividends, start=s, end=e,
                                           trading_days=cfg["backtest"]["trading_days_per_year"])
    views = log_holdout_view(cfg, f"report {stock_id} {strategy}") if holdout else None

    gap = cfg["strategies"]["lgbm"]["gap"] if strategy == "lgbm" else None
    divs = run.dividends
    checks = oracles.run_all(prices, engine.costs, engine.initial_cash, engine.lot_mode,
                             lambda p: build_features(adjust_for_dividends(p, divs)), run.folds, gap,
                             cfg["strategies"]["lgbm"].get("horizon", 1))

    out_dir = Path(cfg["reports_dir"])
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_holdout" if holdout else ""
    stem = f"{stock_id}_{strategy}_{prices['date'].iloc[0]:%Y%m%d}_{prices['date'].iloc[-1]:%Y%m%d}{suffix}"
    png = out_dir / f"{stem}.png"
    eq = res.equity[res.equity.index >= run.sig_start]
    bh = bh_res.equity[bh_res.equity.index >= run.sig_start]
    _plot(png, f"{stock_id} {strategy} vs Buy & Hold (incl. dividends)", eq, bh,
          holdout_start(cfg) if "保留期" in run.windows else None)
    kline_png = out_dir / f"{stem}_kline.png"
    _plot_kline(kline_png, f"{stock_id} {strategy}: K-line and simulated trades, last 120 trading days",
                prices, res.trades)

    c = engine.costs
    lines = [
        f"# {stock_id} 回測報告：{strategy}" + ("（含保留期）" if holdout else ""),
        "",
        "> 僅供研究與模擬，不構成投資建議；本工具不含任何下單功能。",
        "",
        f"- 資料期間：{prices['date'].iloc[0].date()} ~ {prices['date'].iloc[-1].date()}（{len(prices)} 個交易日）",
        f"- 保留期：{ev['holdout_start']} 起。" + (
            f"**本報告包含保留期，這是第 {views} 次查看保留期（紀錄在 reports/holdout_log.md）。**" if holdout
            else "本報告資料截在保留期前一天，看不到保留期。"),
        f"- 訊號用{'除權息還原價' if ev.get('adjust_signal_prices', True) else '未還原價'}產生；成交用未還原價；"
        f"持有中遇到除權息，股利（含配股折價值）於除權息日入帳，稅率 {c.dividend_tax_rate:.2%}，不再投入",
        f"- 策略參數：`{cfg['strategies'][strategy]}`",
        f"- 初始資金 {engine.initial_cash:,.0f} 元，單位：{'整張（1000 股）' if engine.lot == 1000 else '零股（1 股）'}",
        f"- 成本：手續費 {c.commission_rate:.4%} × 折扣 {c.commission_discount}（最低 {c.min_commission:g} 元，無條件捨去），賣出證交稅 {c.sell_tax_rate:.1%}",
        "- 成交規則：第 t 天收盤後產生訊號，第 t+1 天開盤價成交；開盤漲停不買、開盤跌停不賣（記錄於下方）",
        f"- 流動性：每次買進最多 {_participation_text(engine)}",
        "",
    ]
    for name, (m, mb) in wm.items():
        st = sig_tests[name]
        lines += [
            f"## 績效：{name}（{m['start'].date()} ~ {m['end'].date()}）",
            "",
            *PERF_HEADER,
            _fmt_row(strategy, m),
            _fmt_row("買進持有", mb),
            "",
            f"**顯著性**：和 {st['n']} 組「同樣持股比例與換手率」的隨機訊號比，策略年化 {m['cagr']:.1%} "
            f"勝過 {st['percentile']:.0%} 的隨機組（隨機中位數 {st['random_median']:.1%}、前 5% 門檻 "
            f"{st['random_p95']:.1%}），單尾 p = {st['p_value']:.3f}。"
            + ("p < 0.05：選時可能有效（仍需多檔驗證）。" if st["p_value"] < 0.05
               else "p ≥ 0.05：無法和運氣區分。"),
            "",
        ]
    lines += [
        f"![equity]({png.name})",
        "",
        f"![kline]({kline_png.name})",
        "",
    ] + _kline_section(run.signal_prices, stats_before=ev["holdout_start"])
    if run.folds is not None:
        folds = run.folds
        lines += [
            "## Walk-forward",
            "",
            f"共 {len(folds)} 折，平均測試準確率 {folds['accuracy'].mean():.1%}（基準：測試期上漲日比例約 50%）。",
            "",
            "| 折 | 訓練期 | 測試期 | 訓練筆數 | 測試準確率 |",
            "|---|---|---|---|---|",
        ]
        for r in folds.itertuples():
            lines.append(f"| {r.fold} | {r.train_start.date()} ~ {r.train_end.date()} | "
                         f"{r.test_start.date()} ~ {r.test_end.date()} | {r.n_train} | {r.accuracy:.1%} |")
        lines.append("")
    lines += ["## 跳過紀錄", ""]
    limit_skips = res.skipped[res.skipped["reason"].str.contains("停")]
    cash_skips = res.skipped[res.skipped["reason"].str.startswith("資金不足")]
    liq_skips = res.skipped[res.skipped["reason"].str.startswith("流動性不足")]
    if limit_skips.empty:
        lines.append("漲跌停跳過：無。")
    else:
        lines += ["| 日期 | 方向 | 開盤價 | 原因 |", "|---|---|---|---|"]
        lines += [f"| {r.date.date()} | {r.side} | {r.price} | {r.reason} |" for r in limit_skips.itertuples()]
    if not cash_skips.empty:
        lines += ["", f"資金不足一個交易單位而無法買進：{len(cash_skips)} 天"
                  f"（{cash_skips['date'].min().date()} ~ {cash_skips['date'].max().date()}）。"
                  "整張模式下股價 × 1000 超過可用資金時會發生，可提高 initial_cash 或改用 --lot-mode odd。"]
    if not liq_skips.empty:
        lines += ["", f"流動性不足（均量上限不到一個交易單位）而無法買進：{len(liq_skips)} 天。"]
    buys = res.trades[res.trades["side"] == "buy"]
    if engine.max_volume_participation is not None and len(buys):
        lines += ["", f"買進股數受均量上限限制的次數：{int(buys['capped'].sum())} / {len(buys)}。"]
    lines += [
        "",
        "## Oracle 檢查（在本次資料上執行）",
        "",
        "| 檢查 | 結果 | 說明 |",
        "|---|---|---|",
    ]
    lines += [f"| {o.name} | {'PASS' if o.passed else '**FAIL**'} | {o.detail} |" for o in checks]
    lines += [
        "",
        "pytest 另外驗證：三個故意埋錯的版本（當天收盤成交、忘記扣證交稅、特徵往未來偏移一天）都會被上述檢查擋下（tests/test_oracles.py）。",
        "",
    ]
    md = out_dir / f"{stem}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    if not all(o.passed for o in checks):
        print("[report] 警告：有 oracle 檢查未通過，回測結果不可信！")
    print(f"[report] {md}")
    print(f"[report] {png}")
    return md
