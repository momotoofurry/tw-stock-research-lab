"""選股排名策略研究：對照組、三項穩健性檢查（鄰近參數、前後兩段、交易成本）、隨機選股顯著性。

預設只用開發期（保留期之前）；--holdout 才看保留期並記錄。
"""
from __future__ import annotations

from datetime import date
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import Costs  # noqa: E402
from .config import ROOT  # noqa: E402
from .evaluate import effective_end, holdout_start, log_holdout_view  # noqa: E402
from .metrics import compute_metrics  # noqa: E402
from .rank_strategy import (eligible_pools, factor_table, load_universe,  # noqa: E402
                            matched_random_targets, rebalance_days, run_portfolio, select, strategy_targets, turnover)

COLORS = {"策略": "#2a78d6", "0050 含息": "#eb6834", "股票池等權": "#1baf7a", "純動能（無基本面）": "#e87ba4"}


def _metrics(res, start, end, tdays):
    m = compute_metrics(res, start, end, tdays)
    return dict(cagr=m["cagr"], mdd=m["max_drawdown"], sharpe=m["sharpe"], total=m["total_return"],
                costs=m["total_commission"] + m["total_tax"], dividends=m["total_dividends"],
                turnover=turnover(res))


def _scaled_costs(c: Costs, k: float) -> Costs:
    return Costs(commission_rate=c.commission_rate * k, commission_discount=c.commission_discount,
                 min_commission=c.min_commission * k, sell_tax_rate=c.sell_tax_rate * k, price_limit=c.price_limit,
                 dividend_tax_rate=c.dividend_tax_rate)


def study_setup(cfg: dict, universe: str = "default") -> tuple:
    """研究用的股票池與參數：回傳 (rc, ids, ends, data_start, names, pool_desc)。rank 與 lab 共用。"""
    rc = dict(cfg["rank"])
    if universe == "full":
        cand = pd.read_csv(ROOT / cfg["data_dir"] / "full_candidates.csv", dtype=str)
        ids, ends, data_start = list(cand["stock_id"]), dict(zip(cand["stock_id"], cand["end"])), "2016-01-01"
        rc.update(start=rc["full_start"], min_turnover=rc["full_min_turnover"], top_n=rc["full_top_n"])
        from .fetch import fetch_stock_info
        info = fetch_stock_info(cfg)
        names = dict(zip(info["stock_id"], info["stock_name"]))
        pool_desc = (f"全市場上市櫃普通股中，曾在某個調整日通過流動性門檻的 {len(ids)} 檔（含已下市；"
                     "下市股以最後收盤出清，對破產下市偏樂觀）")
    else:
        ids, ends, data_start = list(cfg["evaluation"]["universe"]), None, "2018-01-01"
        names = cfg["evaluation"]["universe"]
        pool_desc = f"{len(ids)} 檔（config `evaluation.universe`，皆為現存上市公司，有存活者偏差）"
    return rc, ids, ends, data_start, names, pool_desc


def run_rank_study(cfg: dict, holdout: bool = False, n_random: int = 100, universe: str = "default",
                   end: str | None = None) -> Path:
    """universe：default = config 的 40 檔；full = 全市場流動性候選（先跑 python -m src.cli bulk-fetch）。"""
    tdays = cfg["backtest"]["trading_days_per_year"]
    end = effective_end(cfg, end, holdout)
    rc, ids, ends, data_start, names, pool_desc = study_setup(cfg, universe)
    U = load_universe(ids, data_start, end, cfg, ends=ends)
    B = load_universe([rc["benchmark"]], data_start, end, cfg, fundamentals=False)
    costs = Costs.from_config(cfg)
    cash0 = cfg["backtest"]["initial_cash"]
    start = rc["start"]
    hs = holdout_start(cfg)
    dev_end = min(pd.Timestamp(end), hs - pd.Timedelta(days=1))
    windows = {"開發期": (pd.Timestamp(start), dev_end)}
    if holdout and pd.Timestamp(end) >= hs:
        windows["保留期"] = (hs, pd.Timestamp(end))
    kw = dict(min_price=rc["min_price"], min_turnover=rc["min_turnover"])
    bt = cfg["backtest"]

    # 所有組合（策略、對照組、隨機、穩健性）套用同一個成交量上限
    run = partial(run_portfolio, max_participation=bt.get("max_volume_participation"),
                  volume_window=bt.get("volume_window", 20))

    def strat(lookback=rc["lookback"], top_n=rc["top_n"], every=rc["rebalance_days"], fund=True, c=costs,
              s=start, e=end):
        tg = strategy_targets(U, s, e, lookback, top_n, every, use_fundamentals=fund, **kw)
        return run(U, tg, c, cash0, s, e), tg

    main, targets = strat()
    momo, _ = strat(fund=False)
    first = U.calendar[U.calendar >= pd.Timestamp(start)][0]
    bench = run(B, {first: [rc["benchmark"]]}, costs, cash0, start, end)
    # 對照組的股票池：每個調整日當天通過價格、流動性、歷史長度門檻的股票（和策略同一池，不用未來資訊）
    pools = eligible_pools(U, rebalance_days(U, start, end, rc["rebalance_days"]), rc["lookback"], **kw)
    ew = run(U, pools, costs, cash0, start, end)
    curves = {"策略": main, "純動能（無基本面）": momo, "股票池等權": ew, "0050 含息": bench}
    # 壓力測試：下市時仍持有的部位全部歸零（破產下市的最壞情況；資料分不出合併或破產）
    stress = run(U, targets, costs, cash0, start, end, delist_recovery=0.0)
    n_delist = int(main.skipped["reason"].str.contains("下市").sum())

    out_dir = ROOT / cfg["reports_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rank_study_{'full_' if universe == 'full' else ''}{date.today():%Y%m%d}" + ("_holdout" if holdout else "")
    views = log_holdout_view(cfg, "rank study") if holdout else None

    from .provenance import data_manifest, run_info
    L = ["# 選股排名策略研究" + ("（含保留期）" if holdout else ""), "",
         *([f"> ⚠️ 此規則已淘汰（{rc.get('retired_note', '')}）。保留期已消耗，本報告只作紀錄與對照組方法的檢查，"
            "不是樣本外證據，也不能再用來調參。", ""] if rc.get("retired") else []),
         "> 僅供研究與模擬，不構成投資建議；本工具不含任何下單功能。", "",
         f"- 股票池：{pool_desc}；實際載入 {len(U.stocks)} 檔",
         f"- 規則：每 {rc['rebalance_days']} 個交易日，篩選 近 63 日漲幅 > 0、近 {rc['lookback']} 日漲幅 > 0、"
         f"近 3 個月營收年增 > 0、近四季 EPS > 0、股價 > {rc['min_price']}、20 日均成交金額 > {rc['min_turnover'] / 1e6:.0f} 百萬，"
         f"依近 {rc['lookback']} 日漲幅取前 {rc['top_n']} 檔等權；t 收盤決定、t+1 開盤成交；零股；含成本、股利、公司行動",
         "- 基本面只用當時已公布的資料：月營收次月 10 日；季報一般公司季末 + 45 天、金融業 + 2 個月、年報 3/31；再加 1 天",
         f"- 保留期 {cfg['evaluation']['holdout_start']} 起" + (f"，**本報告包含保留期（第 {views} 次查看）**" if holdout
                                                               else "，本報告看不到保留期"),
         f"- 成交量上限：每檔每天買進 ≤ 前 {bt.get('volume_window', 20)} 日均量 × {bt.get('max_volume_participation')}；"
         f"對照組股票池為各調整日當天通過門檻的股票（平均 {np.mean([len(v) for v in pools.values()]):.0f} 檔）",
         "- 重現資訊：" + run_info(cfg, U.calendar.max(), data_manifest(cfg, ids + [rc["benchmark"]],
                                                                      out_dir / f"{stem}.data.json"))
         + f"（逐檔雜湊：{stem}.data.json）", ""]

    for wname, (ws, we) in windows.items():
        L += [f"## {wname}（{ws.date()} ~ {we.date()}）", "",
              "| 組合 | 年化 | 最大回撤 | Sharpe | 年週轉率 | 成本 | 股利 |", "|---|---|---|---|---|---|---|"]
        mm = {k: _metrics(v, ws, we, tdays) for k, v in curves.items()}
        for k, m in mm.items():
            L.append(f"| {k} | {m['cagr']:.1%} | {m['mdd']:.1%} | {m['sharpe']:.2f} | {m['turnover']:.1f}x | "
                     f"{m['costs']:,.0f} | {m['dividends']:,.0f} |")
        m = _metrics(stress, ws, we, tdays)
        L += [f"| 策略（壓力：下市持股歸零） | {m['cagr']:.1%} | {m['mdd']:.1%} | {m['sharpe']:.2f} | "
              f"{m['turnover']:.1f}x | {m['costs']:,.0f} | {m['dividends']:,.0f} |", "",
              f"策略全期間遇到持股下市 / 資料結束 {n_delist} 次；上表「策略」以最後收盤出清，壓力列以 0 計，真實結果介於兩者之間。", ""]
        # 隨機選股：同一個當天股票池、同樣持股檔數、同樣續抱檔數（換手率相同）、同樣成本與成交量上限
        rng = np.random.default_rng(0)
        rm = [_metrics(run(U, matched_random_targets(targets, pools, rng), costs, cash0, start, end),
                       ws, we, tdays) for _ in range(n_random)]
        L += [f"**顯著性**：和 {n_random} 組配對的隨機選股比（每次從當天通過價格 / 流動性門檻的股票抽，"
              "持股檔數與續抱檔數和策略相同）", "", "| 指標 | 策略 | 隨機中位數 | 隨機前 5% 門檻 | 策略勝過 | 單尾 p |",
              "|---|---|---|---|---|---|"]
        for key, label, fmt in (("cagr", "年化", "{:.1%}"), ("sharpe", "Sharpe", "{:.2f}")):
            rnd, sv = np.array([m[key] for m in rm]), mm["策略"][key]
            p = (1 + (rnd >= sv).sum()) / (n_random + 1)
            L.append(f"| {label} | {fmt.format(sv)} | {fmt.format(np.median(rnd))} | "
                     f"{fmt.format(np.quantile(rnd, 0.95))} | {(rnd < sv).mean():.0%} | {p:.3f} |")
        L.append("")

    # 穩健性（只在開發期做）
    ws, we = windows["開發期"]
    bm = _metrics(bench, ws, we, tdays)
    L += ["## 穩健性檢查（開發期）", "", f"比較基準：0050 含息 Sharpe {bm['sharpe']:.2f}、年化 {bm['cagr']:.1%}。", "",
          "**1. 鄰近參數**（Sharpe，括號為年化）", "",
          "| 持股檔數 ＼ 動能回看天數 | " + " | ".join(str(x) for x in rc["grid_lookback"]) + " |",
          "|---" * (len(rc["grid_lookback"]) + 1) + "|"]
    wins = 0
    for n in rc["grid_top_n"]:
        cells = []
        for lb in rc["grid_lookback"]:
            m = _metrics(strat(lookback=lb, top_n=n)[0], ws, we, tdays)
            wins += m["sharpe"] > bm["sharpe"]
            cells.append(f"{m['sharpe']:.2f}（{m['cagr']:.1%}）")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    total = len(rc["grid_top_n"]) * len(rc["grid_lookback"])
    L += ["", f"Sharpe 勝過 0050：{wins} / {total} 組（標準：至少 {int(np.ceil(total * 0.75))} 組）", ""]
    mid = ws + (we - ws) / 2
    L += ["**2. 前後兩段**", "", "| 期間 | 策略 Sharpe | 0050 Sharpe | 策略年化 | 0050 年化 |", "|---|---|---|---|---|"]
    halves_ok = True
    for a, b in ((ws, mid), (mid + pd.Timedelta(days=1), we)):
        m, mb_ = _metrics(main, a, b, tdays), _metrics(bench, a, b, tdays)
        halves_ok &= m["sharpe"] > mb_["sharpe"]
        L.append(f"| {a.date()} ~ {b.date()} | {m['sharpe']:.2f} | {mb_['sharpe']:.2f} | {m['cagr']:.1%} | {mb_['cagr']:.1%} |")
    L += ["", "**3. 交易成本**", "", "| 成本 | 年化 | Sharpe | 最大回撤 |", "|---|---|---|---|"]
    cost_ok = True
    for k in (0.0, 1.0, 2.0):
        m = _metrics(strat(c=_scaled_costs(costs, k))[0], ws, we, tdays)
        if k == 2.0:
            cost_ok = m["sharpe"] > bm["sharpe"]
        L.append(f"| ×{k:g} | {m['cagr']:.1%} | {m['sharpe']:.2f} | {m['mdd']:.1%} |")
    L += ["", f"結論：鄰近參數 {'過' if wins >= np.ceil(total * 0.75) else '不過'}、前後兩段 {'過' if halves_ok else '不過'}、"
              f"兩倍成本 {'過' if cost_ok else '不過'}。", ""]

    # 最近一次選股（開發期最後一次調整）
    last_reb = [d for d in targets if d <= we][-1]
    f = factor_table(U, last_reb, rc["lookback"])
    picks = select(f, rc["top_n"], **kw)
    L += [f"## 開發期最後一次選股（{last_reb.date()}）", "", "| 股票 | 近 126 日 | 近 63 日 | 營收年增（3 月） | 近四季 EPS |",
          "|---|---|---|---|---|"]
    for r in f[f["stock_id"].isin(picks)].sort_values("mom", ascending=False).itertuples():
        L.append(f"| {r.stock_id} {names.get(r.stock_id, '')} | {r.mom:+.1%} | {r.mom63:+.1%} | {r.yoy3:+.1%} | {r.eps_ttm:.2f} |")

    png = out_dir / f"{stem}.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, v in curves.items():
        e = v.equity[(v.equity.index >= windows["開發期"][0])]
        if not holdout:
            e = e[e.index <= we]
        ax.plot(e.index, e / e.iloc[0], label={"策略": "Strategy", "0050 含息": "0050 (incl. dividends)",
                                                "股票池等權": "Universe equal-weight",
                                                "純動能（無基本面）": "Momentum only"}[k], color=COLORS[k], lw=2 if k == "策略" else 1.5)
    if holdout and "保留期" in windows:
        ax.axvline(hs, color="#888888", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_ylabel("Equity (start = 1, log scale)")
    ax.legend(frameon=False)
    ax.grid(True, color="#e6e6e6", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)
    L[L.index("## 穩健性檢查（開發期）")] = f"![equity]({png.name})\n\n## 穩健性檢查（開發期）"
    md = out_dir / f"{stem}.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[rank] {md}")
    return md


def headline(md_path) -> str:
    """從研究報告抓出給 LINE 的重點：開發期對照表、顯著性、穩健性結論。"""
    lines = Path(md_path).read_text(encoding="utf-8").splitlines()
    keep, in_dev, in_sig = [], False, False
    for line in lines:
        if (line.startswith("## 開發期") or line.startswith("## 保留期")) and "最後一次選股" not in line:
            in_dev, in_sig = True, False
            keep.append(line.lstrip("# "))
            continue
        if in_dev and line.startswith("## "):
            in_dev = False
        if in_dev and line.startswith("**顯著性**"):
            in_sig = True
            keep.append("隨機選股對照（同股票池、同檔數與換手率）：")
            continue
        if in_dev and line.startswith("| ") and not line.startswith(("| 組合", "| 指標", "|---")):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if in_sig:
                keep.append(f"  {cells[0]}：策略 {cells[1]}、隨機中位數 {cells[2]}、勝過 {cells[4]}、p = {cells[5]}")
            else:
                keep.append(f"{cells[0]}：年化 {cells[1]}、回撤 {cells[2]}、Sharpe {cells[3]}")
        if line.startswith("結論："):
            keep.append("穩健性" + line)
    keep.append(f"完整報告：{Path(md_path).name}")
    keep.append("※ 研究結果，不是投資建議。")
    return "\n".join(keep)
