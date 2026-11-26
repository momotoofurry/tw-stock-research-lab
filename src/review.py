"""預測檢討報告：用前瞻紀錄（data/forecast_log.csv）檢查各模型的實際表現。

- 準確率與 95% 信賴區間（Wilson），和「一律猜多數方向」比較
- 機率校準：預測機率區間 vs 實際發生比例
- 波動範圍（68%）的實際涵蓋率
- 依大盤環境（預測當天加權指數相對季線與近 20 日漲跌）、依個股、依模型版本拆開
- 失效警示：近期表現明顯偏離時提醒（樣本太少時只標註「樣本不足」，不下結論）

報告只描述過去預測的對錯，不會自動修改模型；要改模型仍要走開發期調整 + 新資料驗證。
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from . import journal
from .config import ROOT

MIN_N = 30          # 少於這個樣本數只列數字、不下結論
RECENT_N = 40       # 失效警示看最近幾筆已揭曉的預測


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def regime_table(index_px: pd.DataFrame) -> pd.Series:
    """每個交易日的大盤環境（只用當天以前的指數）：多頭 / 空頭 / 盤整。"""
    c = index_px.set_index("date")["close"].sort_index()
    ma60 = c.rolling(60).mean()
    r20 = c / c.shift(20) - 1
    reg = pd.Series("盤整", index=c.index)
    reg[(c > ma60) & (r20 > 0.02)] = "多頭"
    reg[(c < ma60) & (r20 < -0.02)] = "空頭"
    return reg.where(ma60.notna())


def _acc_block(df: pd.DataFrame) -> dict:
    m = df.dropna(subset=["lgbm_outcome", "lgbm_p"])
    n = len(m)
    if n == 0:
        return dict(n=0)
    hit = ((m["lgbm_p"] > m["lgbm_threshold"]) == (m["lgbm_outcome"] > 0.5)).astype(float)
    rate = m["lgbm_outcome"].mean()
    lo, hi = wilson(hit.sum(), n)
    # 機率品質：Brier（越低越好）、log loss；對照組 = 每次都報這段期間實際上漲比例（事後才知道，對模型偏嚴）
    pr, y = m["lgbm_p"].clip(1e-6, 1 - 1e-6).to_numpy(), (m["lgbm_outcome"] > 0.5).astype(float).to_numpy()
    brier, brier_ref = float(np.mean((pr - y) ** 2)), float(rate * (1 - rate))
    logloss = float(-np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
    return dict(n=n, acc=float(hit.mean()), lo=lo, hi=hi, naive=float(max(rate, 1 - rate)),
                brier=brier, brier_ref=brier_ref, logloss=logloss)


def build_review(cfg: dict, scored: pd.DataFrame | None = None, index_px: pd.DataFrame | None = None) -> dict:
    scored = journal.score(cfg) if scored is None else scored
    out = {"n_records": len(scored)}
    if scored.empty:
        return out
    s = scored.copy()
    s["date"] = pd.to_datetime(s["date"])
    if "model_version" not in s:
        s["model_version"] = np.nan
    s["model_version"] = s["model_version"].fillna("未記錄（舊版）")
    if index_px is None:
        try:
            from .market import index_prices
            index_px = index_prices(cfg)
        except Exception:
            index_px = None
    s["regime"] = s["date"].map(regime_table(index_px)) if index_px is not None else np.nan
    out.update(first=s["date"].min(), last=s["date"].max(), overall=_acc_block(s))
    out["by_version"] = {v: _acc_block(g) for v, g in s.groupby("model_version")}
    out["by_regime"] = {r: _acc_block(g) for r, g in s.dropna(subset=["regime"]).groupby("regime")}
    out["by_stock"] = {k: _acc_block(g) for k, g in s.groupby("stock_id")}
    # 校準
    m = s.dropna(subset=["lgbm_outcome", "lgbm_p"])
    if len(m):
        bins = pd.cut(m["lgbm_p"], [0, .4, .5, .6, .7, 1], include_lowest=True)
        cal = m.groupby(bins, observed=True)["lgbm_outcome"].agg(["count", "mean"])
        out["calibration"] = [(str(i), int(r["count"]), float(r["mean"])) for i, r in cal.iterrows()]
    # 波動範圍涵蓋率
    for k in ("in_r5", "in_r20"):
        v = s[k].dropna()
        lo, hi = wilson(v.sum(), len(v))
        out[k] = dict(n=len(v), cov=float(v.mean()) if len(v) else np.nan, lo=lo, hi=hi)
    out["alerts"] = alerts(s)
    return out


def alerts(s: pd.DataFrame) -> list[str]:
    """近期表現的警示。樣本不足時不下結論。"""
    msgs = []
    m = s.dropna(subset=["lgbm_outcome", "lgbm_p"]).sort_values("date").tail(RECENT_N)
    if len(m) >= MIN_N:
        b = _acc_block(m)
        if b["hi"] < b["naive"]:
            msgs.append(f"LightGBM 最近 {b['n']} 筆準確率 {b['acc']:.0%}（95% 區間上限 {b['hi']:.0%}）"
                        f"低於一律猜多數方向的 {b['naive']:.0%}：模型近期可能失效")
    for k, lab in (("in_r5", "5 日"), ("in_r20", "20 日")):
        v = s.dropna(subset=[k]).sort_values("date").tail(RECENT_N)[k]
        if len(v) >= MIN_N:
            lo, hi = wilson(v.sum(), len(v))
            if hi < 0.68:
                msgs.append(f"{lab}波動範圍最近 {len(v)} 筆只涵蓋 {v.mean():.0%}（上限 {hi:.0%}）：實際波動比預估大，區間偏窄")
            elif lo > 0.68:
                msgs.append(f"{lab}波動範圍最近 {len(v)} 筆涵蓋 {v.mean():.0%}（下限 {lo:.0%}）：區間偏寬")
    return msgs


def _fmt_acc(b: dict) -> str:
    if not b.get("n"):
        return "尚無到期的預測"
    note = "" if b["n"] >= MIN_N else "，樣本不足"
    return (f"{b['acc']:.0%}（95% 區間 {b['lo']:.0%}～{b['hi']:.0%}，{b['n']} 筆{note}）"
            f"｜一律猜多數方向 {b['naive']:.0%}")


def review_text(r: dict, short: bool = False) -> str:
    if not r.get("n_records"):
        return "【預測檢討】尚無前瞻紀錄。"
    L = [f"【預測檢討】{r['first']:%Y-%m-%d} ~ {r['last']:%Y-%m-%d}，共 {r['n_records']} 筆紀錄",
         f"LightGBM 方向：{_fmt_acc(r['overall'])}"]
    o = r["overall"]
    if o.get("n"):
        L.append(f"機率品質：Brier {o['brier']:.3f}（對照：一律報實際上漲比例 {o['brier_ref']:.3f}，越低越好）、"
                 f"log loss {o['logloss']:.3f}（亂猜 50% 為 0.693）")
    for k, lab in (("in_r5", "5 日"), ("in_r20", "20 日")):
        x = r[k]
        if x["n"]:
            L.append(f"{lab}波動範圍涵蓋率：{x['cov']:.0%}（95% 區間 {x['lo']:.0%}～{x['hi']:.0%}，{x['n']} 筆；目標 68%）")
        else:
            L.append(f"{lab}波動範圍：尚無到期的預測")
    if r.get("by_regime"):
        L.append("依大盤環境：" + "；".join(f"{k} {_fmt_acc(v).split('｜')[0]}" for k, v in r["by_regime"].items()))
    if not short:
        L.append("依模型版本：" + "；".join(f"{k} {_fmt_acc(v).split('｜')[0]}" for k, v in r["by_version"].items()))
        if r.get("calibration"):
            L.append("機率校準（預測區間：次數、實際發生比例）：" +
                     "；".join(f"{b} {n} 次 {p:.0%}" for b, n, p in r["calibration"]))
        L.append("依個股：" + "；".join(f"{k} {_fmt_acc(v).split('｜')[0]}" for k, v in r["by_stock"].items()))
    L.append("")
    L += (["⚠️ " + a for a in r["alerts"]] if r["alerts"] else
          [f"沒有失效警示（近期樣本少於 {MIN_N} 筆時不判斷）。"])
    L.append("※ 檢討只描述過去預測的對錯；要改模型須用開發期調整、再用新資料驗證，不會因短期失準自動改參數。")
    return "\n".join(L)


def save_review(r: dict, cfg: dict):
    out = ROOT / cfg["reports_dir"] / f"review_{date.today():%Y%m%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(review_text(r) + "\n", encoding="utf-8")
    return out
