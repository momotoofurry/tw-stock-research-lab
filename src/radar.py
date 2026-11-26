"""選股雷達：全市場排名規則在最新資料上「目前選出」的股票。

這是規則的機械輸出，不是推薦：規則本身在開發期報酬高但回撤深、風險調整後不穩定（見 rank_study_full 報告），
推播時一定附上規則的歷史成績與限制。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .config import ROOT
from .fetch import fetch_stock_info
from .rank_strategy import factor_table, load_universe, rebalance_days, select


def current_picks(cfg: dict) -> dict:
    rc = cfg["rank"]
    cand = pd.read_csv(ROOT / cfg["data_dir"] / "full_candidates.csv", dtype=str)
    alive = cand[cand["end"] >= (pd.Timestamp(date.today()) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")]
    U = load_universe(list(alive["stock_id"]), "2016-01-01", date.today().isoformat(), cfg)
    last = U.calendar[-1]
    kw = dict(min_price=rc["min_price"], min_turnover=rc["full_min_turnover"])
    # 規則現在持有的：最近一次調整日（從 full_start 起每 rebalance_days 個交易日）選出的股票
    reb = [d for d in rebalance_days(U, rc["full_start"], last.strftime("%Y-%m-%d"), rc["rebalance_days"])]
    last_reb = reb[-1]
    held = select(factor_table(U, last_reb, rc["lookback"]), rc["full_top_n"], **kw)
    # 如果今天調整會選出的
    f_now = factor_table(U, last, rc["lookback"])
    now = select(f_now, rc["full_top_n"], **kw)
    next_reb = U.calendar[U.calendar.get_loc(last_reb) + rc["rebalance_days"]] \
        if U.calendar.get_loc(last_reb) + rc["rebalance_days"] < len(U.calendar) else None
    info = fetch_stock_info(cfg)
    names = dict(zip(info["stock_id"], info["stock_name"]))
    inds = dict(zip(info["stock_id"], info["industry_category"]))
    table = f_now[f_now["stock_id"].isin(now)].copy()
    table["名稱"] = table["stock_id"].map(names)
    table["產業"] = table["stock_id"].map(inds)
    table = table.set_index("stock_id").loc[now].reset_index()
    return dict(date=last, last_reb=last_reb, held=held, now=now, table=table, names=names, n=len(U.stocks),
                next_reb=next_reb, retired=bool(rc.get("retired")))


def rule_record() -> list[str]:
    """從最新的全市場研究報告抓規則成績（開發期，若有則含保留期）。"""
    files = sorted((ROOT / "reports").glob("rank_study_full_*.md"), key=lambda p: p.stat().st_mtime)
    if not files:
        return ["（尚無全市場研究報告）"]
    out = []
    for line in files[-1].read_text(encoding="utf-8").splitlines():
        if line.startswith("## 開發期") or line.startswith("## 保留期"):
            if "最後一次選股" in line:
                continue
            out.append(line.lstrip("# "))
        if line.startswith("| 策略 |") or line.startswith("| 0050 含息 |"):
            c = [x.strip() for x in line.strip("|").split("|")]
            out.append(f"  {c[0]}：年化 {c[1]}、回撤 {c[2]}、Sharpe {c[3]}")
    return out


def radar_message(r: dict) -> str:
    L = (["⚠️ 此規則已在保留期驗證失敗而淘汰，以下只是機械輸出的紀錄。", ""] if r.get("retired") else []) + [
         f"【選股雷達】{r['date']:%Y-%m-%d}（全市場 {r['n']} 檔流動性候選）",
         "規則：近 63 / 126 日上漲、近 3 月營收年增 > 0、近四季 EPS > 0、成交金額 > 5000 萬，依 126 日漲幅取前 10",
         "", "今天若調整會選出："]
    for i, x in enumerate(r["table"].itertuples(), 1):
        L.append(f"{i}. {x.stock_id} {x.名稱}（{x.產業}）126 日 {x.mom:+.0%}、營收年增 {x.yoy3:+.0%}、EPS {x.eps_ttm:.1f}")
    changed_in = [s for s in r["now"] if s not in r["held"]]
    changed_out = [s for s in r["held"] if s not in r["now"]]
    L += ["", f"規則目前持有（{r['last_reb']:%Y-%m-%d} 調整）：" + "、".join(f"{s} {r['names'].get(s, '')}" for s in r["held"])]
    if changed_in or changed_out:
        L.append("和目前持有相比：新進 " + ("、".join(changed_in) or "無") + "；移出 " + ("、".join(changed_out) or "無"))
    if r["next_reb"] is not None:
        L.append(f"規則下次調整：{r['next_reb']:%Y-%m-%d}")
    L += ["", "規則歷史成績："] + rule_record()
    L += ["", "※ 這是選股規則的機械輸出，不是推薦。規則報酬高但回撤深（曾腰斬），風險調整後沒有穩定勝過 0050。"]
    return "\n".join(L)


def save_radar(r: dict, cfg: dict) -> Path:
    out = ROOT / cfg["reports_dir"] / f"radar_{r['date']:%Y%m%d}.md"
    out.write_text(radar_message(r) + "\n", encoding="utf-8")
    return out
