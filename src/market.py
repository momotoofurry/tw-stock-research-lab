"""大盤概況：加權指數趨勢、整體法人與融資、類股輪動（ETF 代理）、市場寬度（股票池代理）。

只描述最新收盤時的狀態，不預測漲跌。所有數字只用最後一個交易日（含）以前的資料。
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .adjust import adjust_for_dividends
from .fetch import _cached_rows, fetch_events, fetch_prices

INDEX_ID = "TAIEX"
# 類股代理：用 ETF 的報酬近似該族群（沒有免費的全市場類股指數）
SECTOR_PROXIES = {"0050": "大型權值（0050）", "0052": "科技（0052）", "00891": "半導體（00891）",
                  "0055": "金融（0055）", "0056": "高股息（0056）", "0051": "中型 100（0051）"}


def _adj_close(sid: str, days: int, cfg: dict) -> pd.Series:
    """還原權息與公司行動（分割、減資）後的收盤，避免 0050 分割這類事件被當成大跌。"""
    px = fetch_prices(sid, _ago(days), date.today().isoformat(), cfg)
    if px.empty:
        return pd.Series(dtype=float)
    adj = adjust_for_dividends(px, fetch_events(sid, _ago(days), date.today().isoformat(), cfg, prices=px))
    return adj.set_index("date")["close"]


def _ago(days: int) -> str:
    return (pd.Timestamp(date.today()) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def index_prices(cfg: dict, start: str = "2018-01-01") -> pd.DataFrame:
    return fetch_prices(INDEX_ID, start, date.today().isoformat(), cfg)


def index_state(px: pd.DataFrame, trading_days: int = 252) -> dict:
    """加權指數現況。px：TAIEX 日K（未還原，價格指數）。"""
    p = px.set_index("date").sort_index()
    c = p["close"]
    last = float(c.iloc[-1])
    ma = {n: float(c.rolling(n).mean().iloc[-1]) for n in (20, 60, 120, 240)}
    ret = {n: float(c.iloc[-1] / c.iloc[-1 - n] - 1) for n in (1, 5, 20, 60) if len(c) > n}
    high252 = float(c.iloc[-252:].max())
    r = np.log(c).diff()
    vol20 = r.rolling(20).std() * np.sqrt(trading_days)
    vol_pct = float((vol20.dropna() < vol20.iloc[-1]).mean())
    if ma[20] > ma[60] > ma[120]:
        align = "多頭排列（MA20 > MA60 > MA120）"
    elif ma[20] < ma[60] < ma[120]:
        align = "空頭排列（MA20 < MA60 < MA120）"
    else:
        align = "均線糾結"
    return dict(date=c.index[-1], close=last, ret=ret, ma=ma, align=align,
                dist={n: last / v - 1 for n, v in ma.items()},
                dd_high=last / high252 - 1, high252=high252,
                vol20=float(vol20.iloc[-1]), vol_pct=vol_pct,
                ma60_slope=float(c.rolling(60).mean().iloc[-1] / c.rolling(60).mean().iloc[-21] - 1))


def market_flows(cfg: dict, days: int = 45) -> dict:
    """全市場三大法人買賣超（億元）與融資融券餘額變化。"""
    out = {}
    rows = _cached_rows("TaiwanStockTotalInstitutionalInvestors", f"all_{_ago(days)}",
                        {"start_date": _ago(days)}, cfg, ttl_hours=6)
    if rows:
        df = pd.DataFrame(rows)
        df["net"] = (df["buy"] - df["sell"]) / 1e8
        grp = df["name"].map({"Foreign_Investor": "外資", "Foreign_Dealer_Self": "外資", "Investment_Trust": "投信",
                              "Dealer_self": "自營商", "Dealer_Hedging": "自營商"})
        d = df.assign(g=grp).dropna(subset=["g"]).pivot_table(index="date", columns="g", values="net",
                                                                aggfunc="sum").sort_index()
        out["inst_date"] = pd.Timestamp(d.index[-1]).date()
        out["inst"] = {g: {n: float(d[g].iloc[-n:].sum()) for n in (1, 5, 20)} for g in d.columns}
        out["foreign_streak"] = _streak(d["外資"]) if "外資" in d else 0
    rows = _cached_rows("TaiwanStockTotalMarginPurchaseShortSale", f"all_{_ago(days)}",
                        {"start_date": _ago(days)}, cfg, ttl_hours=6)
    if rows:
        m = pd.DataFrame(rows)
        money = m[m["name"] == "MarginPurchaseMoney"].sort_values("date")
        short = m[m["name"] == "ShortSale"].sort_values("date")
        if len(money):
            b = money["TodayBalance"].astype(float) / 1e8
            out["margin"] = dict(date=money["date"].iloc[-1], balance=float(b.iloc[-1]),
                                 chg1=float(b.iloc[-1] - b.iloc[-2]) if len(b) > 1 else np.nan,
                                 chg20=float(b.iloc[-1] / b.iloc[-21] - 1) if len(b) > 20 else np.nan)
        if len(short):
            s = short["TodayBalance"].astype(float)
            out["short"] = dict(balance=float(s.iloc[-1]),
                                chg20=float(s.iloc[-1] / s.iloc[-21] - 1) if len(s) > 20 else np.nan)
    return out


def _streak(s: pd.Series) -> int:
    """連續買超（正）或賣超（負）天數。"""
    sign = np.sign(s.iloc[-1])
    n = 0
    for v in s.iloc[::-1]:
        if np.sign(v) != sign or v == 0:
            break
        n += 1
    return int(sign * n)


def sector_table(cfg: dict, index_px: pd.DataFrame) -> pd.DataFrame:
    """各類股代理 ETF 近 5 / 20 / 60 日報酬（含權息）與相對加權指數（價格指數）。"""
    ic = index_px.set_index("date")["close"]
    rows = []
    for sid, name in SECTOR_PROXIES.items():
        c = _adj_close(sid, 200, cfg)
        if len(c) < 61:
            continue
        row = {"類股": name}
        for n in (5, 20, 60):
            r = c.iloc[-1] / c.iloc[-1 - n] - 1
            ir = ic.loc[:c.index[-1]].iloc[-1] / ic.loc[:c.index[-1]].iloc[-1 - n] - 1
            row[f"{n}日"], row[f"{n}日相對大盤"] = float(r), float(r - ir)
        rows.append(row)
    return pd.DataFrame(rows)


def breadth(stock_ids: list[str], cfg: dict) -> dict:
    """股票池中收盤在 20 / 60 日均線之上的比例（近似市場寬度；不是全市場）。"""
    above = {20: [], 60: []}
    for sid in stock_ids:
        c = _adj_close(sid, 150, cfg)
        if len(c) < 61:
            continue
        for n in above:
            above[n].append(c.iloc[-1] > c.rolling(n).mean().iloc[-1])
    return {n: (float(np.mean(v)) if v else np.nan) for n, v in above.items()} | {"n": len(above[20])}


def beta_to_index(returns: pd.Series, index_px: pd.DataFrame, window: int = 120) -> float:
    """日報酬對加權指數日報酬的 beta（近 window 日）：大盤漲跌 1%，平均跟著漲跌幾 %。"""
    ir = index_px.set_index("date")["close"].pct_change()
    df = pd.DataFrame({"r": returns, "i": ir}).dropna().iloc[-window:]
    return float(np.cov(df["r"], df["i"])[0, 1] / df["i"].var()) if len(df) > 20 else np.nan


def overview(cfg: dict, stock_ids: list[str] | None = None) -> dict:
    px = index_prices(cfg)
    st = index_state(px, cfg["backtest"]["trading_days_per_year"])
    st["flows"] = market_flows(cfg)
    st["sectors"] = sector_table(cfg, px)
    st["breadth"] = breadth(stock_ids or list(cfg["evaluation"]["universe"]), cfg)
    st["index_px"] = px
    return st


def notes(st: dict) -> list[str]:
    """把大盤現況寫成幾句描述（不預測）。"""
    r, d = st["ret"], st["dist"]
    L = [f"加權指數 {st['close']:,.0f}（{r.get(1, np.nan):+.2%}）；近 5 日 {r.get(5, np.nan):+.1%}、"
         f"20 日 {r.get(20, np.nan):+.1%}、60 日 {r.get(60, np.nan):+.1%}",
         f"趨勢：{st['align']}；距月線 {d[20]:+.1%}、季線 {d[60]:+.1%}、年線 {d[240]:+.1%}；"
         f"季線近 20 日{'上升' if st['ma60_slope'] > 0 else '下降'} {st['ma60_slope']:+.1%}",
         f"距近一年高點 {st['dd_high']:+.1%}；近 20 日年化波動 {st['vol20']:.0%}（在 2018 年以來第 {st['vol_pct']:.0%} 百分位）"]
    f = st.get("flows", {})
    if "inst" in f:
        i = f["inst"]
        parts = [f"{g} {v[1]:+,.0f} 億（5 日 {v[5]:+,.0f}、20 日 {v[20]:+,.0f}）" for g, v in i.items()]
        s = f.get("foreign_streak", 0)
        L.append(f"法人（{f['inst_date']}）：" + "；".join(parts) +
                 (f"；外資連續{'買' if s > 0 else '賣'}超 {abs(s)} 天" if abs(s) >= 2 else ""))
    if "margin" in f:
        m = f["margin"]
        L.append(f"融資餘額 {m['balance']:,.0f} 億（今日 {m['chg1']:+,.1f} 億、20 日 {m['chg20']:+.1%}）" +
                 (f"；融券餘額 20 日 {f['short']['chg20']:+.1%}" if "short" in f else ""))
    b = st.get("breadth", {})
    if b.get("n"):
        L.append(f"市場寬度（{b['n']} 檔大型股）：{b[20]:.0%} 在月線上、{b[60]:.0%} 在季線上")
    sec = st.get("sectors")
    if sec is not None and len(sec):
        top = sec.sort_values("20日相對大盤", ascending=False)
        fmt = lambda rows: "、".join(f"{r['類股']} {r['20日相對大盤']:+.1%}" for _, r in rows.iterrows())  # noqa: E731
        L.append(f"近 20 日相對大盤：最強 {fmt(top.head(2))}；最弱 {fmt(top.tail(2).iloc[::-1])}")
    return L


def short_lines(st: dict) -> list[str]:
    """給 LINE 收盤摘要用的兩三行。"""
    r, d = st["ret"], st["dist"]
    L = [f"加權 {st['close']:,.0f}（{r.get(1, np.nan):+.2%}）｜季線 {d[60]:+.1%}｜距一年高點 {st['dd_high']:+.1%}"]
    f = st.get("flows", {})
    parts = []
    if "inst" in f:
        parts += [f"{g} {v[1]:+,.0f} 億" for g, v in f["inst"].items()]
    if "margin" in f:
        parts.append(f"融資 {f['margin']['chg1']:+,.1f} 億")
    if parts:
        L.append("｜".join(parts))
    return L
