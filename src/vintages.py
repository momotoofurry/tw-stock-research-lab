"""基本面版本快照：FinMind 只給最新值，這裡每次抓到就把「新出現或數值變動」的列追加存檔。

只追加不改寫：每檔股票一個檔 `data/vintages/{dataset}/{stock_id}.csv`，欄位 = stock_id、first_seen + 原始欄位。
日後累積出真正的「當時看到的版本」，才能做不含後見之明的基本面研究。

注意：第一次為某檔建立快照時，所有既有歷史的 first_seen 都是建檔當天，不是真正的公告時點；
只有之後新增或被更正的列，first_seen 才有時點意義。
"""
from __future__ import annotations

import csv
import json
import time
from datetime import date

import pandas as pd

from .config import ROOT
from .fetch import FinMindError

KEYS = {
    "TaiwanStockMonthRevenue": (["revenue_year", "revenue_month"], "revenue"),
    "TaiwanStockFinancialStatements": (["date", "type"], "value"),
}
START = "2016-01-01"   # 和 fundamentals.py 相同，讓快取檔共用


def _dir(cfg: dict):
    return ROOT / cfg["data_dir"] / "vintages"   # data_dir 為絕對路徑時 ROOT / 會直接採用它


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _same(a, b) -> bool:
    x, y = _num(a), _num(b)
    return x == y   # NaN != NaN：視為不同


def record(dataset: str, stock_id: str, rows, cfg: dict, today=None) -> int:
    """追加該股票快照中沒有、或數值和該 key 最後一筆不同的列；回傳追加筆數。"""
    if dataset not in KEYS or not rows:
        return 0
    keys, val = KEYS[dataset]
    stock_id = str(stock_id)
    new = []
    for r in rows:
        if not isinstance(r, dict) or any(k not in r for k in keys) or val not in r:
            continue
        new.append(r)
    if not new:
        return 0
    path = _dir(cfg) / dataset / f"{stock_id}.csv"
    last: dict[tuple, str] = {}
    cols = None
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        cols = list(old.columns)
        for r in old.to_dict("records"):
            last[tuple(str(r[k]) for k in keys)] = r[val]
    stamp = str(today or date.today())
    out = []
    for r in new:
        k = tuple(str(r[c]) for c in keys)
        if k in last and _same(last[k], r[val]):
            continue
        last[k] = r[val]   # 同一批內同一個 key：同值只記一次，值不同仍會再追加
        out.append(r)
    if not out:
        return 0
    if cols is None:
        cols = ["stock_id", "first_seen"] + list(out[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        if f.tell() == 0:
            w.writeheader()
        for r in out:
            w.writerow({"stock_id": stock_id, "first_seen": stamp, **r})
    return len(out)


def _count(cfg: dict, stocks) -> int:
    n = 0
    for ds in KEYS:
        for sid in stocks:
            p = _dir(cfg) / ds / f"{sid}.csv"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    n += max(sum(1 for _ in f) - 1, 0)
    return n


def refresh_batch(cfg: dict, n: int, today=None) -> dict:
    """更新持股 + watchlist，再依游標從全市場候選名單輪流取 n 檔，重抓月營收與財報（觸發快照）。"""
    from . import portfolio as pfm, users
    from .daily import tracked_stocks
    from .research import fetch_dataset
    owner = users.owner_profile(cfg)
    stocks = tracked_stocks(pfm.load_portfolio(owner.portfolio_path), cfg)
    cand_path = ROOT / cfg["data_dir"] / "full_candidates.csv"
    cands = list(dict.fromkeys(pd.read_csv(cand_path, dtype=str)["stock_id"])) if cand_path.exists() else []
    cur_path = _dir(cfg) / "cursor.json"
    cur = 0
    if cur_path.exists():
        try:
            cur = int(json.loads(cur_path.read_text(encoding="utf-8")).get("cursor", 0))
        except (ValueError, TypeError):
            cur = 0
    batch = []
    if cands and n > 0:
        cur %= len(cands)
        batch = [cands[(cur + i) % len(cands)] for i in range(min(n, len(cands)))]
    todo = list(dict.fromkeys(stocks + batch))
    before, errors, stopped = _count(cfg, todo), 0, None
    deadline = time.monotonic() + cfg.get("vintages", {}).get("max_minutes", 20) * 60
    done = set()
    for sid in todo:
        if time.monotonic() > deadline:
            stopped = "time"
            break
        try:
            for ds in KEYS:
                fetch_dataset(ds, sid, START, cfg, ttl_hours=20)
        except FinMindError as e:   # 限流會讓每個請求重試很久：第一次失敗就停，不再抓剩下的
            errors += 1
            stopped = str(e) or type(e).__name__
            break
        done.add(sid)
    adv = 0   # 游標只前進到實際處理完的連續候選股
    for sid in batch:
        if sid not in done:
            break
        adv += 1
    if cands:
        cur = (cur + adv) % len(cands)
    cur_path.parent.mkdir(parents=True, exist_ok=True)
    cur_path.write_text(json.dumps({"cursor": cur}), encoding="utf-8")
    out = {"stocks": len(todo), "added": _count(cfg, todo) - before, "errors": errors}
    if stopped:
        out["stopped"] = stopped
    return out
