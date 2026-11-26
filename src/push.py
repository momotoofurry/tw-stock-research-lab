"""給 LINE 用的精簡個股走向分析（推播，不需要 webhook）。

只描述現況與歷史統計：趨勢、階段、支撐壓力、波動範圍、法人、營收、估值、相對大盤。不是買賣建議。
"""
from __future__ import annotations

import numpy as np

from .dashboard_data import analyze, load, snapshot, stock_name
from .fetch import FinMindError


def stock_message(sid: str, cfg: dict) -> str:
    from . import market
    from .research import chips, revenue_table, valuation
    from .trend import analyze_trend

    name, _ = stock_name(sid, cfg)
    prices, _, adj = load(sid, "2018-01-01", cfg)
    if prices.empty:
        return f"【{sid}】找不到資料"
    hs = cfg["evaluation"]["holdout_start"]
    snap = snapshot(sid, name, prices, adj)
    t = analyze_trend(adj, prices, hs)
    a = analyze(prices, adj, hs, cfg["backtest"]["trading_days_per_year"])
    fr = t["frames"]
    L = [f"【{sid} {name}】{snap['日期']:%m-%d} 收 {snap['收盤']:,.2f}（{snap['漲跌%']:+.1%}）",
         f"趨勢：短 {fr['短期']['label']}｜中 {fr['中期']['label']}｜長 {fr['長期']['label']}；"
         f"ADX {t['adx']:.0f} {t['strength']}，{'多方' if t['pdi'] > t['mdi'] else '空方'}較強",
         f"階段：{t['phase'].split('（')[0]}"]
    lv = t["levels"]
    res = " / ".join(f"{v:,.1f}" for v in lv["resistance"]) or "—"
    sup = " / ".join(f"{v:,.1f}" for v in lv["support"]) or "—"
    L.append(f"壓力 {res}｜支撐 {sup}")
    r5 = a["ranges"].iloc[0]
    L.append(f"5 日波動範圍 {r5['下緣_68']:,.1f}～{r5['上緣_68']:,.1f}（68%，不是方向）")
    try:
        c = chips(sid, cfg)
        if "inst" in c:
            parts = [f"{g} {d[20]:+,.0f}" for g, d in c["inst"].items() if g in ("外資", "投信")]
            L.append("法人 20 日（張）：" + "、".join(parts))
        rv = revenue_table(sid, cfg)
        v = valuation(sid, cfg)
        fund = []
        if len(rv):
            last = rv.iloc[-1]
            fund.append(f"營收年增 {last['年增']:+.0%}（{last['期間']}）")
        if v.get("PER"):
            fund.append(f"本益比 {v['PER']:.1f}（5 年第 {v['PER_pct']:.0%} 位置）")
        if fund:
            L.append("｜".join(fund))
    except FinMindError:
        pass
    try:
        ipx = market.index_prices(cfg)
        sc = adj.set_index("date")["close"]
        ic = ipx.set_index("date")["close"]
        rel = (sc.iloc[-1] / sc.iloc[-61] - 1) - (ic.loc[:sc.index[-1]].iloc[-1] / ic.loc[:sc.index[-1]].iloc[-61] - 1)
        b = market.beta_to_index(sc.pct_change(), ipx)
        if not np.isnan(b):
            L.append(f"近 60 日相對大盤 {rel:+.1%}｜beta {b:.2f}")
    except (FinMindError, IndexError):
        pass
    return "\n".join(L)


def analysis_messages(stock_ids: list[str], cfg: dict, title: str = "個股走向") -> list[str]:
    """回傳要推播的訊息（每則最多 4 檔，避免超過 LINE 長度）。"""
    blocks = []
    for sid in stock_ids:
        try:
            blocks.append(stock_message(sid, cfg))
        except FinMindError as e:
            blocks.append(f"【{sid}】資料抓取失敗：{e}")
    msgs = []
    for k in range(0, len(blocks), 4):
        head = f"【{title}】" + (f"（{k // 4 + 1}）" if len(blocks) > 4 else "")
        msgs.append(head + "\n\n" + "\n\n".join(blocks[k:k + 4]) + "\n\n※ 描述現況與歷史統計，不是買賣建議。")
    return msgs
