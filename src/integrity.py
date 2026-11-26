"""資料完整性盤點（只讀本機快取，不連網）：

1. 股票池：現存上市櫃、官方下市清單、候選名單是否涵蓋所有「曾經達到流動性門檻」的股票
2. 資料提早結束的候選股（下市、合併、長期停牌）：回測以最後收盤出清，列出最後 60 日走勢，看清算假設偏多少
3. 停牌：交易日比加權指數少、連續缺 5 天以上的區段
4. 基本面版本：FinMind 是否保存每次公告的原始版本（目前只有最新值）

輸出 reports/integrity_YYYYMMDD.md。只描述資料，不改任何快取。
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .config import ROOT

GAP_MIN_DAYS = 5


def _read_prices(sid: str, cfg: dict) -> pd.DataFrame:
    f = ROOT / cfg["data_dir"] / f"TaiwanStockPrice_{sid}.csv"
    if not f.exists():
        return pd.DataFrame(columns=["date", "close", "volume"])
    px = pd.read_csv(f, parse_dates=["date"], dtype={"stock_id": str})
    return px[(px["open"] > 0) & (px["close"] > 0)].sort_values("date").reset_index(drop=True)


def suspension_gaps(dates: pd.Series, calendar: pd.DatetimeIndex, min_days: int = GAP_MIN_DAYS) -> list[tuple]:
    """在個股第一天到最後一天之間，連續缺 min_days 個以上交易日的區段：[(開始, 結束, 天數)]。"""
    if len(dates) == 0:
        return []
    cal = calendar[(calendar >= dates.min()) & (calendar <= dates.max())]
    missing = ~cal.isin(pd.DatetimeIndex(dates))
    out, run = [], []
    for d, m in zip(cal, missing):
        if m:
            run.append(d)
        else:
            if len(run) >= min_days:
                out.append((run[0], run[-1], len(run)))
            run = []
    return out


def build_integrity(cfg: dict) -> dict:
    from .bulk import full_universe, liquid_ever
    cfg = {**cfg, "finmind": {**cfg.get("finmind", {}), "offline": True}}
    rc = cfg["rank"]
    uni = full_universe(cfg)
    cand = pd.read_csv(ROOT / cfg["data_dir"] / "full_candidates.csv", dtype=str)
    idx = _read_prices("TAIEX", cfg)
    calendar = pd.DatetimeIndex(idx["date"])
    last_day = calendar.max()
    names = dict(zip(uni["stock_id"], uni["name"]))
    official = set(uni.loc[uni["delisted"].notna(), "stock_id"])

    missed, early, gaps = [], [], []
    for sid in uni["stock_id"]:
        px = _read_prices(sid, cfg)
        if px.empty:
            continue
        in_cand = sid in set(cand["stock_id"])
        if not in_cand and liquid_ever(px, rc["full_start"], rc["min_price"], rc["full_min_turnover"]):
            missed.append(sid)
        if not in_cand:
            continue
        end = px["date"].max()
        if end < last_day - pd.Timedelta(days=10):
            tail = px.tail(60)
            early.append(dict(stock_id=sid, name=names.get(sid, ""), last=end.date(), close=float(px["close"].iloc[-1]),
                              ret60=float(tail["close"].iloc[-1] / tail["close"].iloc[0] - 1),
                              official=sid in official))
        for a, b, n in suspension_gaps(px["date"], calendar):
            gaps.append(dict(stock_id=sid, name=names.get(sid, ""), start=a.date(), end=b.date(), days=n))

    # 基本面版本：看月營收快取有沒有公告時間欄位
    revs = list((ROOT / cfg["data_dir"]).glob("TaiwanStockMonthRevenue_*.json"))
    has_time = 0
    for f in revs[:200]:
        rows = pd.read_json(f)
        has_time += bool(len(rows)) and "create_time" in rows and rows["create_time"].astype(str).str.len().gt(0).any()
    return dict(day=last_day, n_universe=len(uni), n_official=len(official),
                official_by_year=uni.loc[uni["delisted"].notna(), "delisted"].dt.year.value_counts().sort_index(),
                n_cand=len(cand), missed=missed, early=pd.DataFrame(early), gaps=pd.DataFrame(gaps),
                n_rev_checked=min(len(revs), 200), n_rev_with_time=int(has_time))


def integrity_text(r: dict) -> str:
    L = [f"# 資料完整性盤點（快取截至 {r['day']:%Y-%m-%d}）", "", "> 只描述資料；不改快取。", "",
         "## 1. 股票池", "",
         f"- 上市櫃四位數普通股（含官方下市清單）：{r['n_universe']} 檔；官方下市清單 {r['n_official']} 檔",
         "- 官方下市清單各年：" + "、".join(f"{y} 年 {n}" for y, n in r["official_by_year"].items())
         + "。2021 年後幾乎空白，資料源不完整；下市股主要靠「價格資料提早結束」辨識",
         f"- 候選名單 {r['n_cand']} 檔；另有 **{len(r['missed'])} 檔**曾在某天達到流動性門檻卻不在名單"
         + ("（舊名單只看每 63 個交易日的抽樣日；已改規則，需重跑 bulk-fetch 補下載）" if r["missed"] else ""), ""]
    e = r["early"]
    L += ["## 2. 資料提早結束的候選股（回測以最後收盤出清）", ""]
    if len(e):
        L += [f"共 {len(e)} 檔，其中在官方下市清單 {int(e['official'].sum())} 檔。最後 60 個交易日下跌超過 30% 的有 "
              f"{int((e['ret60'] < -0.3).sum())} 檔：這些最可能是經營問題下市，以最後收盤出清最偏樂觀。", "",
              "| 股票 | 最後交易日 | 最後收盤 | 最後 60 日 | 官方清單 |", "|---|---|---|---|---|"]
        for x in e.sort_values("ret60").itertuples():
            L.append(f"| {x.stock_id} {x.name} | {x.last} | {x.close:,.2f} | {x.ret60:+.0%} | {'是' if x.official else '否'} |")
    else:
        L.append("無。")
    g = r["gaps"]
    L += ["", f"## 3. 停牌（連續缺 {GAP_MIN_DAYS} 個交易日以上）", ""]
    if len(g):
        L += [f"{g['stock_id'].nunique()} 檔、{len(g)} 段。停牌期間引擎無法成交，持股以停牌前收盤估值；"
              "復牌第一天若跳空，回測照實反映。最長 20 段：", "", "| 股票 | 開始 | 結束 | 交易日數 |", "|---|---|---|---|"]
        for x in g.sort_values("days", ascending=False).head(20).itertuples():
            L.append(f"| {x.stock_id} {x.name} | {x.start} | {x.end} | {x.days} |")
    else:
        L.append("無。")
    L += ["", "## 4. 基本面版本", "",
          f"- 抽查 {r['n_rev_checked']} 檔月營收快取，有公告時間的 {r['n_rev_with_time']} 檔。",
          "- FinMind 月營收與季財報每期只有一個數值（最新版本），沒有每次公告的原始版本。",
          "- 程式用法定公告期限決定「何時可用」，時間點是對的；但如果公司事後更正營收或重編財報，回測在舊時點用到的是更正後的數字。",
          "- 影響範圍：排名規則只用正負號（營收年增 > 0、近四季 EPS > 0），對小幅更正不敏感；但不能排除。"
          "開發期「加基本面」優於「純動能」的差距，有一部分可能來自這個偏差。", ""]
    return "\n".join(L)


def save_integrity(cfg: dict):
    out = ROOT / cfg["reports_dir"] / f"integrity_{date.today():%Y%m%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(integrity_text(build_integrity(cfg)), encoding="utf-8")
    return out
