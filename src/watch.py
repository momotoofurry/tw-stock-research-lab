"""自選股每週變化提醒：只列出「和上次檢查相比有變化」的項目。

只描述資料變化（新公布的營收 / EPS、創高創低、趨勢階段、支撐壓力、外資買賣超），
不給任何買賣建議。狀態存在 data/watch_state.json（只記錄上次看到的狀態）。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

FOOTER = "※ 只描述資料變化，不是投資建議。"
FAIL = "資料取得失敗"


def _dir(cfg: dict, key: str) -> Path:
    d = Path(cfg[key])
    return d if d.is_absolute() else ROOT / d


def state_path(cfg: dict) -> Path:
    return _dir(cfg, "data_dir") / "watch_state.json"


def _wcfg(cfg: dict) -> dict:
    w = cfg.get("watch", {})
    return dict(high_low_window=int(w.get("high_low_window", 250)), chips_days=int(w.get("chips_days", 5)),
                recent_days=int(w.get("recent_days", 5)))


def gather(stock_id: str, cfg: dict) -> dict:
    """抓一檔的原始資料。測試會整個替換掉這個函式（不連網）。"""
    from .dashboard_data import load
    from .research import chips, quarterly_table, revenue_table
    from .trend import analyze_trend
    start = (pd.Timestamp(date.today()) - pd.Timedelta(days=900)).strftime("%Y-%m-%d")
    prices, _, adj = load(stock_id, start, cfg)
    return dict(revenue=revenue_table(stock_id, cfg), quarterly=quarterly_table(stock_id, cfg),
                adj=adj, trend=analyze_trend(adj, prices), chips=chips(stock_id, cfg))


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def snapshot(data: dict, wc: dict) -> dict:
    """把原始資料壓成要比對的狀態（可存成 json）。"""
    s: dict = {}
    rev = data.get("revenue")
    if rev is not None and len(rev):
        last = rev.iloc[-1]
        s["rev_period"], s["rev_yoy"] = str(last["期間"]), _f(last["年增"])
    q = data.get("quarterly")
    if q is not None and len(q) and "EPS" in q:
        eps = q["EPS"].dropna()
        if len(eps):
            s["eps_quarter"], s["eps"] = str(eps.index[-1]), float(eps.iloc[-1])
    t = data["trend"]
    s["phase"] = str(t["phase"])
    s["support"] = [float(v) for v in t["levels"]["support"]]
    s["resistance"] = [float(v) for v in t["levels"]["resistance"]]
    s["close"] = float(t["close"])
    return s


def _sign_flip(prev, cur) -> str:
    if prev is None or cur is None:
        return ""
    if prev >= 0 > cur:
        return "，年增率轉負"
    if prev < 0 <= cur:
        return "，年增率轉正"
    return ""


def _pct(x) -> str:
    return "—" if x is None else f"{x:+.1%}"


def changes(data: dict, old: dict, wc: dict) -> tuple[list[str], list[str]]:
    """回傳 (有變化的項目, 附帶數字)。附帶數字（外資買賣超）不算變化，只在有變化的股票底下一併列出。"""
    now = snapshot(data, wc)
    ch: list[str] = []
    if "rev_period" in now and now["rev_period"] != old.get("rev_period"):
        ch.append(f"新公布月營收 {now['rev_period']}：年增 {_pct(now['rev_yoy'])}"
                  + _sign_flip(old.get("rev_yoy"), now["rev_yoy"]))
    if "eps_quarter" in now and now["eps_quarter"] != old.get("eps_quarter"):
        qn = now["eps_quarter"]
        prev_q = f"{int(qn[:4]) - 1}{qn[4:]}"
        q = data["quarterly"]["EPS"].dropna()
        ly = float(q[prev_q]) if prev_q in q.index else None
        ch.append(f"新公布 {qn} EPS {now['eps']:.2f}"
                  + (f"（去年同季 {ly:.2f}，{'較高' if now['eps'] > ly else '較低' if now['eps'] < ly else '持平'}）"
                     if ly is not None else "（無去年同季資料）"))
    c = data["adj"].set_index("date").sort_index()["close"]
    win, rec = wc["high_low_window"], wc["recent_days"]
    if len(c) >= win:
        w, r = c.iloc[-win:], c.iloc[-rec:]
        if r.max() >= w.max():
            ch.append(f"本週收盤創近 {win} 個交易日新高")
        if r.min() <= w.min():
            ch.append(f"本週收盤創近 {win} 個交易日新低")
    if old.get("phase") and now["phase"] != old["phase"]:
        ch.append(f"趨勢階段：{old['phase']} → {now['phase']}")
    close = now["close"]
    broke = [v for v in old.get("support", []) if close < v]
    if broke:
        ch.append("收盤跌破上次記錄的支撐 " + "、".join(f"{v:,.2f}" for v in broke) + f"（收盤 {close:,.2f}）")
    up = [v for v in old.get("resistance", []) if close > v]
    if up:
        ch.append("收盤突破上次記錄的壓力 " + "、".join(f"{v:,.2f}" for v in up) + f"（收盤 {close:,.2f}）")
    extra = []
    n = wc["chips_days"]
    inst = (data.get("chips") or {}).get("inst", {}).get("外資")
    if inst:
        v = inst.get(n, inst.get(5))
        extra.append(f"外資近 {n if n in inst else 5} 日累計買賣超 {v:+,.0f} 張")
    return ch, extra


def _names(cfg: dict) -> dict:
    return {str(k): str(v) for k, v in cfg.get("dashboard", {}).get("watchlist", {}).items()}


def check(cfg: dict, stocks: list[str], write_state: bool = True, gather_fn=None) -> str:
    """比對上次狀態、回傳提醒文字；write_state 為 True 才更新 data/watch_state.json。"""
    gather_fn = gather_fn or gather
    wc, names = _wcfg(cfg), _names(cfg)
    sp = state_path(cfg)
    try:
        state = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    except (OSError, ValueError):
        state = {}
    old_stocks = state.get("stocks", {})
    first_run = not old_stocks
    new_stocks = dict(old_stocks)
    blocks, baseline_only, failed = [], [], 0
    for sid in stocks:
        label = f"{sid} {names.get(sid, '')}".strip()
        try:
            data = gather_fn(sid, cfg)
            old = old_stocks.get(sid)
            if old is None:
                baseline_only.append(label)
                new_stocks[sid] = snapshot(data, wc)
                continue
            ch, extra = changes(data, old, wc)
            new_stocks[sid] = snapshot(data, wc)
        except Exception:
            failed += 1
            blocks.append(f"{label}：{FAIL}")
            continue
        if ch:
            blocks.append("\n".join([f"{label}"] + [f"  • {x}" for x in ch + extra]))
    today = date.today().isoformat()
    if write_state:
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(dict(updated=today, stocks=new_stocks), ensure_ascii=False, indent=1),
                      encoding="utf-8")
    head = f"【自選股每週變化】{today}"
    if first_run:
        body = [f"已建立基準，下週開始比較（{len(baseline_only)} 檔）"] + ([f"{FAIL}：{failed} 檔"] if failed else [])
    else:
        body = list(blocks)
        if baseline_only:
            body.append("新加入追蹤、已建立基準：" + "、".join(baseline_only))
        if not [b for b in blocks if not b.endswith(FAIL)] and not failed:
            body.insert(0, f"本週追蹤的 {len(stocks)} 檔都沒有新的變化")
    return "\n".join([head, ""] + body + ["", FOOTER])
