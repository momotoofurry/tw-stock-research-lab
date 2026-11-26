"""全市場資料下載（可中斷、可續傳）。

第一階段：所有上市櫃普通股（含 2015 年後下市的）的日K。
第二階段：只對「曾經在某個調整日通過流動性門檻」的股票抓除權息事件、月營收、季財報。
流動性門檻只用當時的價量資料判斷，所以這樣挑候選股不會偷看未來。
遇到限流就等一段時間再繼續；已經有快取的部分不會重抓。進度記在 data/bulk.log。
"""
from __future__ import annotations

import time
from datetime import date, datetime

import pandas as pd

from .config import ROOT
from .fetch import FinMindRateLimitError, _cached_rows, fetch_events, fetch_prices, fetch_stock_info
from .fundamentals import eps_pit, revenue_pit

EXCLUDE_INDUSTRIES = {"ETF", "ETN", "存託憑證", "受益證券", "指數投資證券", "Index", "大盤"}


def _log(cfg, msg: str) -> None:
    p = ROOT / cfg["data_dir"] / "bulk.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def full_universe(cfg: dict, since: str = "2015-01-01") -> pd.DataFrame:
    """上市（twse）與上櫃（tpex）的四位數普通股，加上 since 之後下市的四位數股票。
    回傳 stock_id、name、industry、delisted（下市日或 NaT）。"""
    info = fetch_stock_info(cfg)
    cur = info[info["type"].isin(["twse", "tpex"]) & info["stock_id"].str.fullmatch(r"\d{4}")
               & ~info["industry_category"].isin(EXCLUDE_INDUSTRIES)]
    out = pd.DataFrame({"stock_id": cur["stock_id"], "name": cur["stock_name"], "industry": cur["industry_category"],
                        "delisted": pd.NaT})
    rows = _cached_rows("TaiwanStockDelisting", "all", {"start_date": since}, cfg, ttl_hours=24 * 7)
    dl = pd.DataFrame(rows)
    if len(dl):
        dl = dl[dl["stock_id"].astype(str).str.fullmatch(r"\d{4}") & ~dl["stock_id"].astype(str).isin(out["stock_id"])]
        out = pd.concat([out, pd.DataFrame({"stock_id": dl["stock_id"].astype(str), "name": dl["stock_name"],
                                            "industry": "已下市", "delisted": pd.to_datetime(dl["date"])})],
                        ignore_index=True)
    return out.drop_duplicates("stock_id").reset_index(drop=True)


def _with_retry(cfg, fn, what: str, wait_min: int = 15, max_waits: int = 12):
    for k in range(max_waits + 1):
        try:
            return fn()
        except FinMindRateLimitError:
            if k == max_waits:
                raise
            _log(cfg, f"{what}：限流，等 {wait_min} 分鐘後繼續（{k + 1}/{max_waits}）")
            time.sleep(wait_min * 60)


def liquid_ever(px: pd.DataFrame, start: str, min_price: float = 10, min_turnover: float = 5e7) -> bool:
    """start 之後是否有任何一天 20 日均成交金額與股價達標。
    只決定「要不要下載」：名單必須是每個調整日股票池的超集，真正能不能投資由研究時當天的門檻判斷。
    （舊版只看個股自己每 63 個交易日抽樣的日子，和研究的調整日錯開，會漏掉約 140 檔。）"""
    if px.empty:
        return False
    p = px.set_index("date").sort_index()
    turnover = (p["close"] * p["volume"]).rolling(20).mean()
    ok = (turnover > min_turnover) & (p["close"] > min_price)
    ok = ok[ok.index >= pd.Timestamp(start)]
    return bool(ok.any())


def run_bulk(cfg: dict, start: str = "2016-01-01", stage: str = "all", limit: int | None = None) -> dict:
    """stage：prices / fundamentals / all。limit：只跑前 N 檔（測試用）。"""
    uni = full_universe(cfg)
    if limit:
        uni = uni.head(limit)
    end = date.today().isoformat()
    rc = cfg["rank"]
    first_reb = rc.get("full_start", rc["start"])
    stats = {"universe": len(uni), "prices": 0, "candidates": 0, "fundamentals": 0}
    _log(cfg, f"開始：{len(uni)} 檔（含已下市 {uni['delisted'].notna().sum()} 檔），stage={stage}")
    candidates = []
    for k, r in enumerate(uni.itertuples(), 1):
        stop = r.delisted.strftime("%Y-%m-%d") if not pd.isna(r.delisted) else end
        px = _with_retry(cfg, lambda: fetch_prices(r.stock_id, start, stop, cfg), f"{r.stock_id} 日K")
        stats["prices"] += 1
        if liquid_ever(px, first_reb, rc["min_price"], rc.get("full_min_turnover", 5e7)):
            candidates.append((r.stock_id, stop))
        if k % 100 == 0:
            _log(cfg, f"日K 進度 {k}/{len(uni)}，候選 {len(candidates)}")
    stats["candidates"] = len(candidates)
    pd.DataFrame(candidates, columns=["stock_id", "end"]).to_csv(ROOT / cfg["data_dir"] / "full_candidates.csv",
                                                                 index=False)
    _log(cfg, f"日K 完成，流動性候選 {len(candidates)} 檔")
    if stage in ("fundamentals", "all"):
        for k, (sid, stop) in enumerate(candidates, 1):
            _with_retry(cfg, lambda: fetch_events(sid, start, stop, cfg), f"{sid} 除權息 / 公司行動")
            _with_retry(cfg, lambda: revenue_pit(sid, cfg), f"{sid} 月營收")
            _with_retry(cfg, lambda: eps_pit(sid, cfg), f"{sid} 財報")
            stats["fundamentals"] += 1
            if k % 50 == 0:
                _log(cfg, f"基本面進度 {k}/{len(candidates)}")
    _log(cfg, f"完成：{stats}")
    return stats
