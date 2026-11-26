"""預測前瞻紀錄：每天收盤後記下各模型的預測與訊號，等結果揭曉後自動評分。

這是唯一完全沒被調參「看過」的樣本外成績：紀錄一旦寫入就不改（同一天同一檔只保留第一次）。
紀錄在 data/forecast_log.csv（data/ 已 gitignore）。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .dashboard_data import analyze, current_signals, load

LOG_COLUMNS = ["date", "stock_id", "close", "lgbm_p", "lgbm_horizon", "lgbm_hurdle", "lgbm_threshold",
               "sig_ma_cross", "sig_candle_volume", "sig_lgbm",
               "r5_lo68", "r5_hi68", "r20_lo68", "r20_hi68", "recorded_at", "model_version"]

# 會影響預測的設定；任何一項改變就是新的模型版本
VERSIONED_KEYS = [("strategies", "lgbm"), ("strategies", "ma_cross"), ("strategies", "candle_volume")]


def model_version(cfg: dict) -> str:
    """設定的雜湊（8 碼）。同時把版本與設定內容記到 data/model_versions.json，方便日後對照。"""
    import hashlib
    import json
    snap = {f"{a}.{b}": cfg[a][b] for a, b in VERSIONED_KEYS}
    text = json.dumps(snap, sort_keys=True, ensure_ascii=False, default=str)
    v = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    p = log_path(cfg).parent / "model_versions.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    known = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if v not in known:
        known[v] = {"first_seen": datetime.now().strftime("%Y-%m-%d"), "config": snap}
        p.write_text(json.dumps(known, ensure_ascii=False, indent=2), encoding="utf-8")
    return v


def log_path(cfg: dict) -> Path:
    d = Path(cfg["data_dir"])
    return (d if d.is_absolute() else ROOT / d) / "forecast_log.csv"


def read_log(cfg: dict) -> pd.DataFrame:
    p = log_path(cfg)
    if not p.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(p, dtype={"stock_id": str}, parse_dates=["date"])


def snapshot_row(stock_id: str, prices: pd.DataFrame, adj: pd.DataFrame, cfg: dict) -> dict:
    a = analyze(prices, adj, cfg["evaluation"]["holdout_start"], cfg["backtest"]["trading_days_per_year"])
    sg = current_signals(adj, cfg)
    lc = cfg["strategies"]["lgbm"]
    r5, r20 = a["ranges"].iloc[0], a["ranges"].iloc[1]

    def state(s):
        return np.nan if "error" in sg[s] else sg[s]["state"]

    return dict(date=a["date"], stock_id=stock_id, close=a["last"],
                lgbm_p=sg["lgbm"].get("p_up", np.nan), lgbm_horizon=lc.get("horizon", 1),
                lgbm_hurdle=lc.get("hurdle", 0.0), lgbm_threshold=lc["threshold"],
                sig_ma_cross=state("ma_cross"), sig_candle_volume=state("candle_volume"), sig_lgbm=state("lgbm"),
                r5_lo68=r5["下緣_68"], r5_hi68=r5["上緣_68"], r20_lo68=r20["下緣_68"], r20_hi68=r20["上緣_68"],
                recorded_at=datetime.now().strftime("%Y-%m-%d %H:%M"), model_version=model_version(cfg))


def record(stock_ids: list[str], cfg: dict, start: str = "2018-01-01") -> pd.DataFrame:
    """記錄今天（各股最後交易日）的快照；已經有同一天同一檔的紀錄就跳過。回傳新增的列。"""
    log = read_log(cfg)
    have = set(zip(log["date"].dt.strftime("%Y-%m-%d"), log["stock_id"])) if len(log) else set()
    new = []
    for sid in stock_ids:
        prices, _, adj = load(sid, start, cfg)
        if prices.empty:
            continue
        row = snapshot_row(sid, prices, adj, cfg)
        if (row["date"].strftime("%Y-%m-%d"), sid) not in have:
            new.append(row)
    add = pd.DataFrame(new, columns=LOG_COLUMNS)
    if len(add):
        out = pd.concat([log, add], ignore_index=True) if len(log) else add
        p = log_path(cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)
    return add


def score(cfg: dict, start: str = "2018-01-01", log: pd.DataFrame | None = None, loader=None) -> pd.DataFrame:
    """對每筆紀錄算出已揭曉的結果（還沒到期的留 NaN）。報酬一律用還原權息價。"""
    log = read_log(cfg) if log is None else log
    if log.empty:
        return log
    loader = loader or (lambda sid: load(sid, start, cfg))
    rows = []
    for sid, g in log.groupby("stock_id"):
        prices, _, adj = loader(sid)
        c = adj.set_index("date")["close"]
        pos = {d: i for i, d in enumerate(c.index)}
        for r in g.itertuples():
            i = pos.get(pd.Timestamp(r.date))
            out = r._asdict()
            out.pop("Index")

            def fwd(h):
                return c.iloc[i + h] / c.iloc[i] - 1 if i is not None and i + h < len(c) else np.nan

            h = int(r.lgbm_horizon)
            fh, f5, f20 = fwd(h), fwd(5), fwd(20)
            out["lgbm_outcome"] = np.nan if np.isnan(fh) else float(fh > r.lgbm_hurdle)
            out["in_r5"] = np.nan if np.isnan(f5) else float(r.r5_lo68 <= r.close * (1 + f5) <= r.r5_hi68)
            out["in_r20"] = np.nan if np.isnan(f20) else float(r.r20_lo68 <= r.close * (1 + f20) <= r.r20_hi68)
            out["fwd20"] = f20
            rows.append(out)
    return pd.DataFrame(rows)


def summary(scored: pd.DataFrame) -> dict:
    """前瞻成績彙總。樣本少時（< 30）數字只供參考。"""
    if scored.empty:
        return dict(n_records=0)
    s = {"n_records": len(scored), "first_date": scored["date"].min(), "last_date": scored["date"].max()}
    m = scored.dropna(subset=["lgbm_outcome", "lgbm_p"])
    if len(m):
        hit = (m["lgbm_p"] > m["lgbm_threshold"]) == (m["lgbm_outcome"] > 0.5)
        rate = m["lgbm_outcome"].mean()
        s.update(lgbm_n=len(m), lgbm_accuracy=float(hit.mean()), lgbm_naive=float(max(rate, 1 - rate)))
    for k in ("in_r5", "in_r20"):
        v = scored[k].dropna()
        s[f"{k}_n"], s[f"{k}_coverage"] = len(v), float(v.mean()) if len(v) else np.nan
    f = scored.dropna(subset=["fwd20"])
    for strat in ("ma_cross", "candle_volume", "lgbm"):
        col = f"sig_{strat}"
        on, off = f[f[col] == 1]["fwd20"], f[f[col] == 0]["fwd20"]
        s[f"{strat}_fwd20_on"] = float(on.mean()) if len(on) else np.nan
        s[f"{strat}_fwd20_off"] = float(off.mean()) if len(off) else np.nan
        s[f"{strat}_n_on"], s[f"{strat}_n_off"] = len(on), len(off)
    return s


def summary_text(s: dict) -> str:
    if not s.get("n_records"):
        return "前瞻紀錄：尚無資料（從今天開始累積）。"
    lines = [f"前瞻紀錄：{s['n_records']} 筆（{s['first_date']:%Y-%m-%d} ~ {s['last_date']:%Y-%m-%d}）"]
    if s.get("lgbm_n"):
        lines.append(f"- LightGBM 已揭曉 {s['lgbm_n']} 筆：準確率 {s['lgbm_accuracy']:.1%}，"
                     f"一律猜多數方向 {s['lgbm_naive']:.1%}" + ("（樣本 < 30，僅供參考）" if s["lgbm_n"] < 30 else ""))
    else:
        lines.append("- LightGBM：還沒有到期的預測")
    for k, lab in (("in_r5", "5 日"), ("in_r20", "20 日")):
        if s.get(f"{k}_n"):
            lines.append(f"- {lab} 68% 波動範圍實際涵蓋率 {s[f'{k}_coverage']:.0%}（{s[f'{k}_n']} 筆，目標約 68%）")
    return "\n".join(lines)
