"""個股多面向研究資料包：基本面、估值、籌碼、技術面、模型輸出、新聞標題。

只整理「事實」（數字與出處），不下結論。多空論點由 /tw-research skill 根據這份資料包撰寫，
而且必須引用其中的數字，不給買賣建議或目標價。
"""
from __future__ import annotations

import json
import time
from datetime import date

import numpy as np
import pandas as pd

from .config import ROOT
from .dashboard_data import STRATEGY_NAMES, analyze, current_signals, lgbm_reliability, load, stock_name, target_text
from .fetch import FinMindError, _cache_path, _request


def fetch_dataset(dataset: str, stock_id: str, start: str, cfg: dict, ttl_hours: float = 12) -> pd.DataFrame:
    """小型資料集（月營收、財報、籌碼、新聞）用整包快取，ttl_hours 內不重抓。"""
    path = _cache_path(f"{stock_id}_{start}", cfg["data_dir"], dataset).with_suffix(".json")
    from .fetch import offline
    if path.exists() and (offline(cfg) or time.time() - path.stat().st_mtime < ttl_hours * 3600):
        rows = json.loads(path.read_text(encoding="utf-8"))
    elif offline(cfg):
        rows = []
    else:
        rows = _request(dataset, {"data_id": stock_id, "start_date": start}, cfg)
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        try:   # 基本面版本快照：失敗不影響原本的抓取
            from .vintages import record
            record(dataset, stock_id, rows, cfg)
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
    return df


def _ago(days: int) -> str:
    return (pd.Timestamp(date.today()) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def revenue_table(stock_id: str, cfg: dict) -> pd.DataFrame:
    df = fetch_dataset("TaiwanStockMonthRevenue", stock_id, _ago(800), cfg)
    if df.empty:
        return df
    df = df.sort_values(["revenue_year", "revenue_month"]).drop_duplicates(["revenue_year", "revenue_month"], keep="last")
    df["期間"] = df["revenue_year"].astype(str) + "-" + df["revenue_month"].astype(int).map("{:02d}".format)
    df["營收(億)"] = df["revenue"] / 1e8
    df["月增"] = df["revenue"].pct_change()
    prev = df.set_index(["revenue_year", "revenue_month"])["revenue"]
    df["年增"] = [r.revenue / prev.get((r.revenue_year - 1, r.revenue_month), np.nan) - 1 for r in df.itertuples()]
    return df[["期間", "營收(億)", "月增", "年增"]].tail(12).reset_index(drop=True)


def quarterly_table(stock_id: str, cfg: dict) -> pd.DataFrame:
    df = fetch_dataset("TaiwanStockFinancialStatements", stock_id, _ago(800), cfg)
    if df.empty:
        return df
    p = df.pivot_table(index="date", columns="type", values="value", aggfunc="last").sort_index()
    out = pd.DataFrame(index=p.index)
    out["EPS"] = p.get("EPS")
    rev = p.get("Revenue")
    if rev is not None:
        out["營收(億)"] = rev / 1e8
        for k, lab in (("GrossProfit", "毛利率"), ("OperatingIncome", "營益率"), ("IncomeAfterTaxes", "淨利率")):
            if k in p:
                out[lab] = p[k] / rev
    out.index = [f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in out.index]
    return out.tail(6)


def valuation(stock_id: str, cfg: dict) -> dict:
    df = fetch_dataset("TaiwanStockPER", stock_id, _ago(365 * 5), cfg)
    if df.empty:
        return {}
    df = df.sort_values("date")
    last = df.iloc[-1]
    out = {"date": last["date"].date()}
    for k in ("PER", "PBR", "dividend_yield"):
        s = df[k].replace(0, np.nan).dropna()
        if len(s) and last[k] > 0:
            out[k] = float(last[k])
            out[f"{k}_pct"] = float((s < last[k]).mean())   # 在近 5 年的百分位
            out[f"{k}_med"] = float(s.median())
    return out


def trailing_pe(close: float, eps: pd.Series) -> tuple[float, float] | None:
    """自算本益比 = 收盤 ÷ 近四季 EPS 合計，回傳 (本益比, EPS 合計)；不足四季或合計 ≤ 0 回傳 None。"""
    e = eps.dropna().tail(4)
    if len(e) < 4 or e.sum() <= 0:
        return None
    return close / e.sum(), float(e.sum())


def chips(stock_id: str, cfg: dict) -> dict:
    inst = fetch_dataset("TaiwanStockInstitutionalInvestorsBuySell", stock_id, _ago(60), cfg)
    out = {}
    if len(inst):
        inst["net"] = (inst["buy"] - inst["sell"]) / 1000  # 張
        group = inst["name"].map({"Foreign_Investor": "外資", "Foreign_Dealer_Self": "外資",
                                  "Investment_Trust": "投信", "Dealer_self": "自營商", "Dealer_Hedging": "自營商"})
        daily = inst.assign(g=group).dropna(subset=["g"]).pivot_table(index="date", columns="g", values="net",
                                                                      aggfunc="sum").sort_index()
        out["inst_last_date"] = daily.index[-1].date()
        out["inst"] = {g: {n: float(daily[g].iloc[-n:].sum()) for n in (1, 5, 20)} for g in daily.columns}
    mg = fetch_dataset("TaiwanStockMarginPurchaseShortSale", stock_id, _ago(60), cfg)
    if len(mg):
        mg = mg.sort_values("date")
        b = mg["MarginPurchaseTodayBalance"]
        s = mg["ShortSaleTodayBalance"]
        out["margin"] = dict(date=mg["date"].iloc[-1].date(), balance=float(b.iloc[-1]),
                             chg20=float(b.iloc[-1] - b.iloc[-21]) if len(b) > 20 else np.nan,
                             usage=float(b.iloc[-1] / mg["MarginPurchaseLimit"].iloc[-1])
                             if mg["MarginPurchaseLimit"].iloc[-1] else np.nan,
                             short=float(s.iloc[-1]),
                             short_chg20=float(s.iloc[-1] - s.iloc[-21]) if len(s) > 20 else np.nan)
    return out


def news(stock_id: str, cfg: dict, n: int = 10, days: int = 7) -> pd.DataFrame:
    """FinMind 新聞一次只能抓一天，逐日抓最近 days 天；過去的日子長期快取，今天 3 小時。"""
    parts = []
    for k in range(days):
        day = _ago(k)
        part = fetch_dataset("TaiwanStockNews", stock_id, day, cfg, ttl_hours=3 if k == 0 else 24 * 365)
        if len(part):
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["date", "source", "title", "link"])
    df = pd.concat(parts, ignore_index=True)
    df["key"] = df["title"].str.replace(r"\s*-\s*[^-]*$", "", regex=True).str.slice(0, 40)
    return df.sort_values("date", ascending=False).drop_duplicates("key").head(n)[["date", "source", "title", "link"]]


def build_research(stock_id: str, cfg: dict, start: str = "2018-01-01") -> str:
    name, industry = stock_name(stock_id, cfg)
    prices, _, adj = load(stock_id, start, cfg)
    if prices.empty:
        raise FinMindError(f"找不到 {stock_id} 的價格資料")
    hs = cfg["evaluation"]["holdout_start"]
    a = analyze(prices, adj, hs, cfg["backtest"]["trading_days_per_year"])
    close = float(prices["close"].iloc[-1])   # 未還原收盤，和官方本益比同一基準
    L = [f"# {stock_id} {name} 研究資料包（{a['date']:%Y-%m-%d} 收盤）", "",
         f"產業：{industry or '—'}。收盤 {close:,.2f}（未還原）。"
         "資料來源：FinMind（證交所 / 公開資訊觀測站 / 新聞彙整）。本檔只整理事實，不含買賣建議。", ""]
    q = pd.DataFrame()

    L += ["## 1. 營收（月）", ""]
    try:
        rv = revenue_table(stock_id, cfg)
        if rv.empty:
            L.append("無資料。")
        else:
            L += ["| 月份 | 營收（億） | 月增 | 年增 |", "|---|---|---|---|"]
            L += [f"| {r.期間} | {r._2:,.1f} | {r.月增:+.1%} | {r.年增:+.1%} |" for r in rv.itertuples()]
            L.append("")
            L.append("註：月營收在次月 10 日前公布。")
    except FinMindError as e:
        L.append(f"抓取失敗：{e}")

    L += ["", "## 2. 季度財報", ""]
    try:
        q = quarterly_table(stock_id, cfg)
        if q.empty:
            L.append("無資料。")
        else:
            cols = list(q.columns)
            L += ["| 季度 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
            for idx, r in q.iterrows():
                cells = []
                for c in cols:
                    v = r[c]
                    cells.append("—" if pd.isna(v) else (f"{v:.1%}" if "率" in c else f"{v:,.2f}"))
                L.append(f"| {idx} | " + " | ".join(cells) + " |")
    except FinMindError as e:
        L.append(f"抓取失敗：{e}")

    L += ["", "## 3. 估值（近 5 年百分位：0% = 最便宜、100% = 最貴；殖利率相反）", ""]
    try:
        v = valuation(stock_id, cfg)
        if not v:
            L.append("無資料。")
        for k, lab in (("PER", "本益比"), ("PBR", "股價淨值比"), ("dividend_yield", "殖利率 %")):
            if k in v:
                L.append(f"- {lab} {v[k]:.2f}（近 5 年中位數 {v[f'{k}_med']:.2f}，百分位 {v[f'{k}_pct']:.0%}）")
        if v.get("PER") is None and v:
            L.append("- 本益比：近四季虧損或無資料")
        if v.get("PER") and v["date"] == a["date"].date():
            L.append(f"- 上面的本益比是證交所 / 櫃買中心公布值（{v['date']}），隱含近四季 EPS "
                     f"{close / v['PER']:.2f}（以收盤 {close:,.2f} 回推）")
        own = trailing_pe(close, q["EPS"]) if "EPS" in q else None
        if own:
            line = f"- 自算本益比 {own[0]:.2f}（收盤 ÷ §2 最近四季 EPS 合計 {own[1]:.2f}）"
            if v.get("PER") and abs(own[0] / v["PER"] - 1) > 0.05:
                line += ("；和公布值差超過 5%，可能是 EPS 定義（例如歸屬母公司、加權股數）或季報更正造成，"
                         "引用時兩個都要列")
            L.append(line)
    except FinMindError as e:
        L.append(f"抓取失敗：{e}")

    L += ["", "## 4. 籌碼（單位：張）", ""]
    try:
        c = chips(stock_id, cfg)
        if "inst" in c:
            L += [f"三大法人買賣超（截至 {c['inst_last_date']}）：", "", "| 法人 | 1 日 | 5 日 | 20 日 |", "|---|---|---|---|"]
            L += [f"| {g} | {d[1]:+,.0f} | {d[5]:+,.0f} | {d[20]:+,.0f} |" for g, d in c["inst"].items()]
        if "margin" in c:
            m = c["margin"]
            L += ["", f"- 融資餘額 {m['balance']:,.0f} 張（20 日變化 {m['chg20']:+,.0f}，使用率 {m['usage']:.1%}）；"
                      f"融券餘額 {m['short']:,.0f} 張（20 日變化 {m['short_chg20']:+,.0f}）（{m['date']}）"]
    except FinMindError as e:
        L.append(f"抓取失敗：{e}")

    L += ["", "## 5. 技術面", ""] + [f"- {n}" for n in a["notes"]]
    lv = a["levels"]
    L.append("- 參考價位：" + "、".join(f"{k} {v:,.2f}" for k, v in lv.items()))
    for r in a["ranges"].itertuples():
        L.append(f"- 未來 {r.天數} 日波動範圍：68% {r.下緣_68:,.1f} ~ {r.上緣_68:,.1f}；95% {r.下緣_95:,.1f} ~ {r.上緣_95:,.1f}"
                 "（依近 60 日波動率，不是方向預測）")

    L += ["", "## 6. 相對大盤", ""]
    try:
        from . import market
        ipx = market.index_prices(cfg)
        ic = ipx.set_index("date")["close"]
        sc = adj.set_index("date")["close"]
        for n in (20, 60, 120):
            if len(sc) > n and len(ic) > n:
                sr, ir = sc.iloc[-1] / sc.iloc[-1 - n] - 1, ic.loc[:sc.index[-1]].iloc[-1] / ic.loc[:sc.index[-1]].iloc[-1 - n] - 1
                L.append(f"- 近 {n} 日：本股 {sr:+.1%}、加權指數 {ir:+.1%}（相對 {sr - ir:+.1%}）")
        b = market.beta_to_index(sc.pct_change(), ipx)
        L.append(f"- 近 120 日對加權指數 beta {b:.2f}（大盤漲跌 1%，本股平均跟著漲跌約 {b:.1f}%；描述過去，不是預測）")
        L += [f"- 大盤：{n}" for n in market.notes(market.index_state(ipx, cfg["backtest"]["trading_days_per_year"]))[:2]]
    except FinMindError as e:
        L.append(f"抓取失敗：{e}")

    L += ["", "## 7. 趨勢分析", ""]
    from .trend import analyze_trend, trend_lines
    L += [f"- {n}" for n in trend_lines(analyze_trend(adj, prices, hs))]
    L += ["", "## 8. 策略體檢（開發期歷史證據，不是推薦）", ""]
    try:
        from .checkup import checkup_lines, run_checkup
        L += checkup_lines(run_checkup(stock_id, cfg, n_random=cfg["dashboard"]["n_random"]))[1:]
    except (FinMindError, ValueError) as e:
        L.append(f"無法執行：{e}")

    L += ["", "## 9. 研究模型輸出與可信度", ""]
    sg = current_signals(adj, cfg)
    for s in ("ma_cross", "candle_volume", "lgbm"):
        v = sg[s]
        if "error" in v:
            L.append(f"- {STRATEGY_NAMES[s]}：{v['error']}")
            continue
        L.append(f"- {STRATEGY_NAMES[s]}：目前{'持有' if v['state'] else '空手'}（自 {v['since']:%Y-%m-%d}）")
    if "proba" in sg["lgbm"]:
        rel = lgbm_reliability(sg["lgbm"]["proba"], adj, cfg, before=hs)
        L.append(f"- LightGBM 預測「{target_text(cfg)}」機率 {sg['lgbm']['p_up']:.0%}；開發期樣本外準確率 "
                 f"{rel['accuracy']:.1%}，一律猜多數方向 {rel['naive']:.1%}")
    L.append("- 40 檔多檔研究中，以上策略都沒有普遍勝過買進持有，也沒有可和運氣區分的選時能力")

    L += ["", "## 10. 近一週新聞標題（未查證，僅標題）", ""]
    try:
        nw = news(stock_id, cfg)
        if nw.empty:
            L.append("無。")
        L += [f"- {r.date:%m-%d} {r.source}：[{r.title}]({r.link})" for r in nw.itertuples()]
    except FinMindError as e:
        L.append(f"抓取失敗：{e}")
    return "\n".join(L) + "\n"


def save_research(stock_id: str, cfg: dict) -> str:
    text = build_research(stock_id, cfg)
    out = ROOT / cfg["reports_dir"] / f"research_{stock_id}_{date.today():%Y%m%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return str(out)
