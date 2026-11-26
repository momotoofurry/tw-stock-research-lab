"""台股看盤與研究介面（只做研究與模擬，沒有任何下單功能）。

啟動：.venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import dashboard_data as dd
from src import portfolio as pfm
from src.config import load_config
from src.features import build_features
from src.fetch import FinMindError
from src.kline import PATTERNS, detect_patterns

UP, DOWN = "#d9363e", "#1f9d55"           # 台股慣例：紅漲綠跌
MA_COLORS = {5: "#2a78d6", 20: "#eb6834", 60: "#7b5ea7"}
STRAT, BH = "#2a78d6", "#eb6834"
BULLISH = {"hammer", "bull_engulfing", "three_white_soldiers", "long_red", "volume_breakout"}

st.set_page_config(page_title="台股看盤研究", layout="wide")
cfg = load_config()
dcfg = cfg["dashboard"]


# ---------------- 資料（快取） ----------------
@st.cache_data(ttl=3600, show_spinner="抓取資料中…")
def load(stock_id: str, start: str):
    return dd.load(stock_id, start, cfg)


@st.cache_data(ttl=3600, show_spinner="計算策略訊號中…")
def signals(stock_id: str, start: str):
    _, _, adj = load(stock_id, start)
    return dd.current_signals(adj, cfg)


@st.cache_data(ttl=6 * 3600, show_spinner="計算開發期表現與顯著性（約 10–30 秒）…")
def track(stock_id: str, strategy: str, start: str):
    return dd.track_record(stock_id, strategy, cfg, start, dcfg["n_random"])


def pct(x, digits=1):
    return "—" if pd.isna(x) else f"{x:+.{digits}%}"


def kline_figure(title: str, prices: pd.DataFrame, adj: pd.DataFrame, days: int, extras: list[str],
                 show_pat: bool) -> go.Figure:
    """K 線 + 均線 + 型態標記 + 成交量 + 副圖指標。"""
    mas = {n: prices["close"].rolling(n).mean() for n in MA_COLORS}
    feats = build_features(adj)
    pats = detect_patterns(adj)
    view = prices.iloc[-days:]
    idx = view.index
    n_sub = len(extras)
    fig = make_subplots(rows=2 + n_sub, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.6, 0.18] + [0.22 / max(n_sub, 1)] * n_sub)
    fig.add_trace(go.Candlestick(x=view["date"], open=view["open"], high=view["high"], low=view["low"],
                                 close=view["close"], name="K 線", increasing_line_color=UP,
                                 increasing_fillcolor=UP, decreasing_line_color=DOWN,
                                 decreasing_fillcolor=DOWN), row=1, col=1)
    for n, color in MA_COLORS.items():
        fig.add_trace(go.Scatter(x=view["date"], y=mas[n][idx], name=f"MA{n}", line=dict(color=color, width=1.5)),
                      row=1, col=1)
    if show_pat:
        pv = pats.iloc[-days:]
        for kind, sym, ycol, off in (("多方型態", "triangle-up", "low", 0.98), ("空方 / 中性型態", "triangle-down", "high", 1.02)):
            pts = []
            for d, row in pv.iterrows():
                names = [k for k, v in row.items() if v and ((k in BULLISH) == (kind == "多方型態")) and k != "doji"]
                if names:
                    pts.append((d, "、".join(PATTERNS[k].split("（")[0] for k in names)))
            if pts:
                px = prices.set_index("date")
                fig.add_trace(go.Scatter(
                    x=[d for d, _ in pts], y=[px.loc[d, ycol] * off for d, _ in pts], mode="markers",
                    marker=dict(symbol=sym, size=9, color="#222222" if kind == "多方型態" else "#888888"),
                    name=kind, text=[t for _, t in pts], hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>"),
                    row=1, col=1)
    vcol = [UP if c >= o else DOWN for o, c in zip(view["open"], view["close"])]
    fig.add_trace(go.Bar(x=view["date"], y=view["volume"] / 1000, marker_color=vcol, name="成交量（張）",
                         showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=view["date"], y=(prices["volume"].rolling(20).mean() / 1000)[idx],
                             name="均量20", line=dict(color="#555555", width=1)), row=2, col=1)
    f_view = feats.iloc[-days:]
    for k, ind in enumerate(extras, start=3):
        if ind == "RSI14":
            y, lines = f_view["rsi_14"], (30, 70)
        elif ind == "MFI14":
            y, lines = f_view["mfi_14"], (20, 80)
        else:
            v = prices["volume"].astype(float)
            obv = (pd.Series(adj["close"].values, index=prices.index).diff().apply(lambda x: (x > 0) - (x < 0))
                   * v).cumsum()
            y, lines = pd.Series(obv[idx].values, index=f_view.index) / 1000, None
        fig.add_trace(go.Scatter(x=f_view.index, y=y, name=ind, line=dict(color="#2a78d6", width=1.3)),
                      row=k, col=1)
        if lines:
            for lv in lines:
                fig.add_hline(y=lv, line=dict(color="#aaaaaa", width=1, dash="dot"), row=k, col=1)
        fig.update_yaxes(title_text=ind, row=k, col=1)
    fig.update_layout(height=520 + 160 * n_sub, xaxis_rangeslider_visible=False, hovermode="x unified",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.03),
                      title=title)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="價格（未還原）", row=1, col=1)
    fig.update_yaxes(title_text="量（張）", row=2, col=1)
    return fig


@st.cache_data(ttl=6 * 3600, show_spinner="策略體檢中…")
def checkup(stock_id: str, start_date: str = ""):
    from src.checkup import run_checkup
    return run_checkup(stock_id, cfg, start_date or start, n_random=dcfg["n_random"])


@st.cache_data(ttl=3 * 3600, show_spinner="整理研究資料包…")
def research_pack(stock_id: str) -> str:
    from src.research import build_research
    text = build_research(stock_id, cfg)
    return text.split("\n", 2)[2]   # 去掉標題列（頁面上已有）


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def info(stock_id: str):
    return dd.stock_name(stock_id, cfg)


# ---------------- 側邊欄 ----------------
st.sidebar.title("台股看盤研究")
st.sidebar.caption("只做研究與模擬，不提供投資建議，也沒有下單功能。")
watch = dict(dcfg["watchlist"])
extra = st.sidebar.text_input("加入其他股票代號（逗號分隔）", "")
for code in [c.strip() for c in extra.split(",") if c.strip()]:
    watch.setdefault(code, "")
start = st.sidebar.text_input("歷史資料起日", dcfg["history_start"])
if st.sidebar.button("重新抓最新資料"):
    st.cache_data.clear()
st.sidebar.markdown("---")
st.sidebar.caption(f"保留期 {cfg['evaluation']['holdout_start']} 起不納入策略績效評估；走勢與目前訊號使用最新資料。")

tab_market, tab_analysis, tab_pf, tab_overview, tab_chart, tab_signal, tab_bt = st.tabs(
    ["大盤概況", "個股分析", "我的持股", "自選股總覽", "走勢", "訊號與預測", "回測"])

# ---------------- 個股分析 ----------------
RULES = {
    "ma_cross": "5 日均線 > 20 日均線 → 持有，否則空手",
    "candle_volume": "多方 K 線型態 + 量增 + 站上 MA20 → 進場；空方型態 / 跌破 MA20 / 持有滿 20 日 → 出場",
    "lgbm": "LightGBM 預測上漲機率 > 門檻 → 持有",
}

with tab_analysis:
    hs = cfg["evaluation"]["holdout_start"]
    c1, c2 = st.columns([1, 3])
    code = c1.text_input("輸入股票代號", value="2317", max_chars=6, help="上市、上櫃股票代號，例如 2330、1216").strip()
    if code:
        name, industry = info(code)
        try:
            prices, divs, adj = load(code, start)
        except FinMindError as e:
            prices = None
            st.error(f"抓取 {code} 失敗：{e}")
        if prices is not None and prices.empty:
            st.error(f"找不到 {code} 的資料（代號錯誤、尚未上市或已下市）。")
        elif prices is not None:
            a = dd.analyze(prices, adj, hs, cfg["backtest"]["trading_days_per_year"])
            snap = dd.snapshot(code, name, prices, adj)
            c2.markdown(f"### {code} {name}　<span style='font-size:0.8em;color:#777'>{industry}</span>",
                        unsafe_allow_html=True)
            m = st.columns(5)
            m[0].metric(f"{a['date']:%Y-%m-%d} 收盤", f"{a['last']:,.2f}", pct(snap["漲跌%"]), delta_color="inverse")
            m[1].metric("近 20 日", pct(snap["20日%"]))
            m[2].metric("近 60 日", pct(snap["60日%"]))
            m[3].metric("RSI14", f"{snap['RSI14']:.0f}")
            m[4].metric("年化波動率", f"{a['vol60'] * (cfg['backtest']['trading_days_per_year'] ** 0.5):.0%}")

            left, right = st.columns([3, 2])
            with left:
                st.markdown("#### 技術面現況")
                for n in a["notes"]:
                    st.markdown(f"- {n}")
                if snap["今日型態"] != "—":
                    st.markdown(f"- 今日 K 線型態：{snap['今日型態']}")
            with right:
                st.markdown("#### 參考價位")
                lv = pd.DataFrame({"價位": a["levels"]}).rename_axis("").reset_index()
                lv["距收盤"] = lv["價位"] / a["last"] - 1
                st.dataframe(lv.style.format({"價位": "{:,.2f}", "距收盤": "{:+.1%}"}), hide_index=True, width="stretch")
            st.plotly_chart(kline_figure(f"{code} {name}", prices, adj, 126, ["RSI14"], True), width="stretch")

            st.markdown("#### 趨勢分析")
            from src import trend as _trend
            tr_info = _trend.analyze_trend(adj, prices, hs)
            for n in _trend.trend_lines(tr_info):
                st.markdown(f"- {n}")
            st.caption("回歸線：近 N 日價格（還原權息）的最佳直線；R² 越接近 1 代表走勢越像一條直線（趨勢越乾淨）。"
                       "支撐 / 壓力取近一年的轉折高低點。只描述現況，不預測。")
            with st.expander("策略體檢：哪一類方法在這檔股票上歷史表現較好"):
                if st.toggle("執行（約 10 秒）", key=f"ck_{code}"):
                    try:
                        ck = checkup(code, start)
                    except (FinMindError, ValueError) as e:
                        st.error(str(e))
                    else:
                        tbl = ck["table"].drop(columns=["key"])
                        st.dataframe(tbl.style.format({"年化": "{:.1%}", "最大回撤": "{:.1%}", "Sharpe": "{:.2f}",
                                                       "持股比例": "{:.0%}", "隨機百分位": "{:.0%}", "p": "{:.3f}"},
                                                      na_rep="—"), hide_index=True, width="stretch")
                        from src.checkup import checkup_lines
                        st.markdown(checkup_lines(ck)[-2])
                        st.caption(f"開發期 {ck['start']:%Y-%m-%d} ~ {ck['end']:%Y-%m-%d}；含手續費、證交稅、股利。"
                                   f"同時比較 3 種策略，顯著門檻修正為 p < {ck['alpha']:.3f}。"
                                   "這是歷史證據，不是推薦；過去有效不代表未來有效。")

            st.markdown("#### 預測")
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**價格波動範圍**（統計區間，不是方向預測）")
                rg = a["ranges"].rename(columns={"天數": "未來交易日", "下緣_95": "95% 下緣", "下緣_68": "68% 下緣",
                                                 "上緣_68": "68% 上緣", "上緣_95": "95% 上緣"})
                st.dataframe(rg.style.format({c: "{:,.1f}" for c in rg.columns[1:]}), hide_index=True, width="stretch")
                cov = a["coverage"]
                st.caption(f"依近 60 日波動率、假設不漲不跌推算。用同樣方法回測開發期（{hs} 前），"
                           f"實際落在 68% 區間內的比例：5 日 {cov[5]:.0%}、20 日 {cov[20]:.0%}。")
            with p2:
                st.markdown("**LightGBM 方向預測**")
                sg = signals(code, start)
                v = sg["lgbm"]
                if "error" in v:
                    st.write(v["error"])
                else:
                    rel = dd.lgbm_reliability(v["proba"], adj, cfg, before=hs)
                    st.metric(f"模型預測：{dd.target_text(cfg)}的機率", f"{v['p_up']:.0%}")
                    if rel["n"]:
                        good = rel["accuracy"] - rel["naive"] >= 0.02
                        st.markdown(
                            f"開發期樣本外 {rel['n']} 次預測：準確率 **{rel['accuracy']:.1%}**，"
                            f"一律猜多數方向的準確率 {rel['naive']:.1%} → "
                            + ("比亂猜略好，但仍需多檔驗證" if good else "**沒有比亂猜好，這個機率不具參考價值**"))
                    else:
                        st.markdown("開發期內沒有足夠的樣本外預測可評估可信度。")

            with st.expander("基本面、估值、籌碼、新聞（研究資料包）"):
                if st.toggle("載入（會抓月營收、財報、法人、融資券、新聞，約 10–20 秒）", key=f"rs_{code}"):
                    try:
                        st.markdown(research_pack(code))
                    except FinMindError as e:
                        st.error(str(e))
                st.caption("想要多空論點整理：在 Claude Code 輸入 /tw-research " + code)

            st.markdown("#### 近期策略訊號")
            st.caption("各研究策略的規則在最後一個交易日收盤後給出的狀態。按下方按鈕可計算它們在這檔股票開發期的歷史可信度。")
            sg = signals(code, start)
            show_tr = st.toggle("計算各策略歷史可信度（約 30–60 秒）", key=f"an_tr_{code}")
            rows = []
            for s in ("ma_cross", "candle_volume", "lgbm"):
                v = sg[s]
                row = {"策略": dd.STRATEGY_NAMES[s], "規則": RULES[s],
                       "目前訊號": "—" if "error" in v else ("持有" if v["state"] else "空手"),
                       "自": "—" if "error" in v else f"{v['since']:%Y-%m-%d}"}
                if show_tr:
                    try:
                        tr = track(code, s, start)
                        mt, mb, sgt = tr["metrics"], tr["bh"], tr["sig"]
                        row.update({"開發期年化": f"{mt['cagr']:.1%}", "買進持有": f"{mb['cagr']:.1%}",
                                    "p 值": f"{sgt['p_value']:.2f}",
                                    "可信度": "可能有效（單檔）" if sgt["p_value"] < 0.05 else "無法和運氣區分"})
                    except Exception as e:
                        row.update({"可信度": f"無法計算：{e}"})
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.warning("以上是研究工具的機械輸出，不是投資建議。在 40 檔股票的開發期研究裡，三個策略都沒有穩定勝過買進持有，"
                       "也沒有顯示出可以和運氣區分的選時能力；波動範圍只描述價格可能擺動的幅度，不代表方向。"
                       "是否買賣、部位大小請依自己的判斷與風險承受度決定。")

# ---------------- 我的持股 ----------------
@st.cache_data(ttl=3600, show_spinner="計算持股與風險…")
def portfolio_view(pf_json: str):
    pf = json.loads(pf_json)
    book = pfm.build_book(pf, cfg)
    risk = pfm.risk_summary(book, cfg["backtest"]["trading_days_per_year"])
    alerts = pfm.evaluate_rules(pf, book, risk)
    divs = pfm.upcoming_dividends(list(book.positions["stock_id"]), cfg)
    return book, risk, alerts, divs


with tab_pf:
    pf = pfm.load_portfolio()
    st.caption("持股與規則只存在你電腦上的 portfolio.yaml（不會上傳）。警示只代表「你自己設定的規則」被觸發，不是買賣建議。")
    with st.expander("編輯持股與規則", expanded=not any(float(h.get("shares") or 0) > 0 for h in pf["holdings"])):
        cols = ["stock_id", "shares", "avg_cost", "buy_date", "stop_loss", "take_profit",
                "trailing_stop_pct", "ma_break", "max_weight", "note"]
        hdf = pd.DataFrame(pf["holdings"], columns=cols)
        hdf["buy_date"] = pd.to_datetime(hdf["buy_date"], errors="coerce")
        edited = st.data_editor(
            hdf, num_rows="dynamic", hide_index=True, width="stretch", key="pf_editor",
            column_config={
                "stock_id": st.column_config.TextColumn("代號", required=True),
                "shares": st.column_config.NumberColumn("股數", min_value=0, step=1, help="1 張 = 1000 股"),
                "avg_cost": st.column_config.NumberColumn("平均成本", min_value=0.0, format="%.2f"),
                "buy_date": st.column_config.DateColumn("買進日", help="移動停損從這天起算最高點；留空 = 近一年"),
                "stop_loss": st.column_config.NumberColumn("停損價", format="%.2f", help="留空則用下方預設 %"),
                "take_profit": st.column_config.NumberColumn("停利價", format="%.2f", help="留空則用下方預設 %"),
                "trailing_stop_pct": st.column_config.NumberColumn("移動停損", format="%.2f",
                                                                   help="例如 0.15 = 從高點回落 15%"),
                "ma_break": st.column_config.NumberColumn("跌破均線", step=1, help="例如 60 = 收盤跌破 60 日均線"),
                "max_weight": st.column_config.NumberColumn("權重上限", format="%.2f", help="例如 0.4 = 40%"),
                "note": st.column_config.TextColumn("備註"),
            })
        st.markdown("**預設規則**（個股欄位留空時套用；留空 = 不啟用）")
        d = pf["defaults"]
        r1 = st.columns(4)
        sl = r1[0].number_input("停損：低於成本 %", value=None if d["stop_loss_pct"] is None else abs(d["stop_loss_pct"]) * 100,
                                min_value=0.0, max_value=100.0, step=1.0, placeholder="不啟用")
        tp = r1[1].number_input("停利：高於成本 %", value=None if d["take_profit_pct"] is None else abs(d["take_profit_pct"]) * 100,
                                min_value=0.0, step=1.0, placeholder="不啟用")
        ts = r1[2].number_input("移動停損：從高點回落 %", value=None if d["trailing_stop_pct"] is None
                                else abs(d["trailing_stop_pct"]) * 100, min_value=0.0, max_value=100.0, step=1.0,
                                placeholder="不啟用")
        ma = r1[3].number_input("跌破幾日均線", value=d["ma_break"], min_value=2, max_value=480, step=1, placeholder="不啟用")
        r2 = st.columns(4)
        mw = r2[0].number_input("單檔權重上限 %", value=None if d["max_weight"] is None else d["max_weight"] * 100,
                                min_value=0.0, max_value=100.0, step=5.0, placeholder="不啟用")
        pdd = r2[1].number_input("組合從高點回落 %", value=None if d["portfolio_drawdown_pct"] is None
                                 else abs(d["portfolio_drawdown_pct"]) * 100, min_value=0.0, max_value=100.0, step=1.0,
                                 placeholder="不啟用")
        near = r2[2].number_input("「接近」提醒範圍 %", value=float(d.get("near_pct") or 0.03) * 100, min_value=0.0,
                                  max_value=20.0, step=0.5)
        cash = r2[3].number_input("現金（元）", value=float(pf.get("cash") or 0), min_value=0.0, step=10000.0)
        if st.button("儲存", type="primary"):
            def clean(v):
                return None if v is None or (isinstance(v, float) and np.isnan(v)) or (v is pd.NaT) else v
            holdings = []
            for r in edited.to_dict("records"):
                if not clean(r.get("stock_id")):
                    continue
                h = {k: clean(r.get(k)) for k in cols}
                h["stock_id"] = str(h["stock_id"]).strip()
                h["shares"] = float(h["shares"] or 0)
                h["avg_cost"] = float(h["avg_cost"] or 0)
                h["buy_date"] = pd.Timestamp(h["buy_date"]).strftime("%Y-%m-%d") if h["buy_date"] else None
                for k in ("stop_loss", "take_profit", "trailing_stop_pct", "max_weight"):
                    h[k] = None if h[k] is None else float(h[k])
                h["ma_break"] = None if h["ma_break"] is None else int(h["ma_break"])
                h["note"] = h["note"] or ""
                holdings.append(h)
            pct = lambda v: None if v is None else v / 100  # noqa: E731
            pf = {"cash": float(cash), "holdings": holdings, "defaults": {
                "stop_loss_pct": None if sl is None else -sl / 100, "take_profit_pct": pct(tp),
                "trailing_stop_pct": None if ts is None else -ts / 100, "ma_break": None if ma is None else int(ma),
                "max_weight": pct(mw), "portfolio_drawdown_pct": None if pdd is None else -pdd / 100,
                "near_pct": near / 100}}
            pfm.save_portfolio(pf)
            portfolio_view.clear()
            st.success("已儲存到 portfolio.yaml")
            pf = pfm.load_portfolio()

    if not any(float(h.get("shares") or 0) > 0 for h in pf["holdings"]):
        st.info("請在上方「編輯持股與規則」填入股數與平均成本，按「儲存」。")
    else:
        try:
            book, risk, alerts, divs = portfolio_view(json.dumps(pf, sort_keys=True, default=str))
        except FinMindError as e:
            st.error(f"抓取資料失敗：{e}")
        else:
            pos = book.positions
            day_pnl = float((pos["market_value"] * pos["day_chg"] / (1 + pos["day_chg"])).sum())
            m = st.columns(5)
            m[0].metric(f"總資產（{pos['date'].max():%Y-%m-%d}）", f"{book.total:,.0f}")
            m[1].metric("今日損益", f"{day_pnl:+,.0f}", f"{day_pnl / (book.total - day_pnl):+.2%}", delta_color="inverse")
            m[2].metric("未實現損益", f"{pos['unrealized'].sum():+,.0f}",
                        f"賣出成本後 {pos['net_if_sold'].sum():+,.0f}", delta_color="off")
            m[3].metric("組合年化波動率", f"{risk['vol']:.0%}")
            m[4].metric("距近一年高點", f"{risk['dd_now']:+.1%}")

            st.markdown("#### 警示（你設定的規則）")
            if alerts.empty:
                st.success("沒有規則被觸發或接近觸發。" if pfm._has_rules(pf) else "尚未設定任何規則。")
            else:
                st.dataframe(alerts.rename(columns={"stock_id": "代號"}).style.map(lambda v: f"color: {UP}; font-weight: 600" if v == "觸發" else
                                              ("color: #b7791f" if v == "接近" else ""), subset=["狀態"]),
                             hide_index=True, width="stretch")

            st.markdown("#### 持股")
            show = pos[["stock_id", "name", "shares", "avg_cost", "close", "day_chg", "unrealized_pct", "unrealized",
                        "market_value", "weight", "dd_from_high"]].rename(columns={
                "stock_id": "代號", "name": "名稱", "shares": "股數", "avg_cost": "成本", "close": "收盤",
                "day_chg": "今日", "unrealized_pct": "損益%", "unrealized": "未實現損益", "market_value": "市值",
                "weight": "權重", "dd_from_high": "距高點"})
            st.dataframe(show.style.format({"股數": "{:,.0f}", "成本": "{:,.2f}", "收盤": "{:,.2f}", "今日": "{:+.1%}",
                                            "損益%": "{:+.1%}", "未實現損益": "{:+,.0f}", "市值": "{:,.0f}",
                                            "權重": "{:.1%}", "距高點": "{:+.1%}"})
                         .map(lambda v: f"color: {UP}" if isinstance(v, float) and v > 0 else
                              (f"color: {DOWN}" if isinstance(v, float) and v < 0 else ""),
                              subset=["今日", "損益%", "未實現損益"]),
                         hide_index=True, width="stretch")

            left, right = st.columns(2)
            with left:
                st.markdown("#### 風險（以目前權重回推）")
                st.markdown(
                    f"- 單日 95% VaR 約 **{risk['var95_1d']:,.0f}** 元（100 天約有 5 天虧損超過此數）\n"
                    f"- 近一年最大回撤 {risk['mdd_1y']:.1%}\n"
                    f"- {risk['history_start']:%Y} 年以來最差 5 日 {risk['worst5']:.1%}（{risk['worst5_date']:%Y-%m-%d}）、"
                    f"最差 20 日 {risk['worst20']:.1%}（{risk['worst20_date']:%Y-%m-%d}）\n"
                    f"- 最大單檔 {risk['max_weight_id']} 佔 {risk['max_weight']:.0%}；等效持股檔數 {risk['effective_n']:.1f}"
                    "（越接近 1 越集中）\n"
                    "- 產業：" + "、".join(f"{k} {v:.0%}" for k, v in risk["industry"].items())
                    + f"、現金 {risk['cash_weight']:.0%}")
                st.markdown("**近一年日報酬相關係數**（越接近 1，越容易同漲同跌）")
                st.dataframe(risk["corr"].style.format("{:.2f}"), width="stretch")
            with right:
                st.markdown("#### 情境試算")
                mode = st.radio("情境", ["全部持股同時變動", "單一持股變動"], horizontal=True)
                if mode == "全部持股同時變動":
                    shock = st.slider("價格變動 %", -50, 50, -20, step=5)
                    shocks = {s: shock / 100 for s in pos["stock_id"]}
                else:
                    who = st.selectbox("股票", list(pos["stock_id"]),
                                       format_func=lambda s: f"{s} {pos.set_index('stock_id').loc[s, 'name']}")
                    shock = st.slider("價格變動 %", -50, 50, -20, step=5, key="single_shock")
                    shocks = {who: shock / 100}
                tbl, chg, pct_chg = pfm.scenario(book, shocks)
                st.metric("總資產變化", f"{chg:+,.0f} 元", f"{pct_chg:+.1%}", delta_color="inverse")
                st.dataframe(tbl.rename(columns={"stock_id": "代號", "name": "名稱", "market_value": "目前市值",
                                                 "shock": "變動", "change": "損益", "new_value": "變動後市值"})
                             .style.format({"目前市值": "{:,.0f}", "變動": "{:+.0%}", "損益": "{:+,.0f}",
                                            "變動後市值": "{:,.0f}"}), hide_index=True, width="stretch")
                st.caption(f"參考：同樣權重下，歷史最差 20 日為 {risk['worst20']:.1%}，"
                           f"約 {risk['worst20'] * (book.total - book.cash):+,.0f} 元。")

            st.markdown("#### 近 90 天除權息（已公告）")
            if divs.empty:
                st.write("無。")
            else:
                st.dataframe(divs, hide_index=True, width="stretch")
            st.download_button("下載今日持股摘要（Markdown）", pfm.daily_brief(pf, cfg),
                               file_name=f"brief_{pos['date'].max():%Y%m%d}.md", mime="text/markdown")

# ---------------- 大盤概況 ----------------
@st.cache_data(ttl=3600, show_spinner="整理大盤資料…")
def market_view():
    from src import market
    stv = market.overview(cfg)
    return stv, market.notes(stv)


with tab_market:
    try:
        mk, mk_notes = market_view()
    except FinMindError as e:
        st.error(f"抓取大盤資料失敗：{e}")
    else:
        r = mk["ret"]
        mc = st.columns(5)
        mc[0].metric(f"加權指數（{mk['date']:%Y-%m-%d}）", f"{mk['close']:,.0f}", pct(r.get(1), 2), delta_color="inverse")
        mc[1].metric("近 20 日", pct(r.get(20)))
        mc[2].metric("距季線", pct(mk["dist"][60]))
        mc[3].metric("距一年高點", pct(mk["dd_high"]))
        mc[4].metric("20 日年化波動", f"{mk['vol20']:.0%}", f"2018 年來第 {mk['vol_pct']:.0%} 百分位", delta_color="off")
        st.markdown("#### 現況")
        for n in mk_notes:
            st.markdown(f"- {n}")
        st.plotly_chart(kline_figure("加權指數", mk["index_px"], mk["index_px"], 252, [], False), width="stretch")
        left, right = st.columns(2)
        with left:
            st.markdown("#### 類股相對強弱（ETF 代理，含權息）")
            sec = mk["sectors"]
            if len(sec):
                fmtc = {c: "{:+.1%}" for c in sec.columns if c != "類股"}
                st.dataframe(sec.style.format(fmtc).map(
                    lambda v: f"color: {UP}" if isinstance(v, float) and v > 0 else
                    (f"color: {DOWN}" if isinstance(v, float) and v < 0 else ""), subset=list(fmtc)),
                    hide_index=True, width="stretch")
                st.caption("沒有免費的全市場類股指數，用代表性 ETF 近似；「相對大盤」= ETF 報酬 − 加權指數報酬。")
        with right:
            st.markdown("#### 三大法人（全市場，億元）")
            fl = mk["flows"]
            if "inst" in fl:
                st.dataframe(pd.DataFrame(fl["inst"]).T.rename(columns={1: "1 日", 5: "5 日", 20: "20 日"})
                             .style.format("{:+,.0f}"), width="stretch")
            if "margin" in fl:
                m = fl["margin"]
                st.markdown(f"融資餘額 **{m['balance']:,.0f} 億**（今日 {m['chg1']:+,.1f} 億、20 日 {m['chg20']:+.1%}）")
        # 你的持股對大盤的敏感度
        pf_own = pfm.load_portfolio()
        if any(float(h.get("shares") or 0) > 0 for h in pf_own["holdings"]):
            try:
                from src import market as _mkt
                book_own, *_ = portfolio_view(json.dumps(pf_own, sort_keys=True, default=str))
                w = book_own.positions.set_index("stock_id")["market_value"] / book_own.total
                port_r = (book_own.returns[w.index].dropna() * w).sum(axis=1)
                beta = _mkt.beta_to_index(port_r, mk["index_px"])
                st.markdown("#### 你的持股和大盤")
                st.markdown(f"以目前權重回推，近 120 個交易日組合對加權指數的 **beta 約 {beta:.2f}**："
                            f"大盤漲跌 1% 時，組合平均跟著漲跌約 {beta:.1f}%（描述過去的關係，不是預測）。")
                rows = []
                for sid in w.index:
                    b = _mkt.beta_to_index(book_own.returns[sid], mk["index_px"])
                    rows.append({"代號": sid, "名稱": book_own.positions.set_index("stock_id").loc[sid, "name"],
                                 "權重": w[sid], "beta（120 日）": b})
                st.dataframe(pd.DataFrame(rows).style.format({"權重": "{:.0%}", "beta（120 日）": "{:.2f}"}),
                             hide_index=True, width="stretch")
            except FinMindError as e:
                st.error(str(e))
        st.caption("大盤概況只描述最新收盤的狀態，不預測漲跌。市場寬度用 40 檔大型股近似，不是全市場。")

# ---------------- 總覽 ----------------
with tab_overview:
    rows, sig_rows, errs = [], [], []
    for code, name in watch.items():
        try:
            prices, _, adj = load(code, start)
            if prices.empty:
                errs.append(f"{code}：沒有資料")
                continue
            rows.append(dd.snapshot(code, name, prices, adj))
            sg = signals(code, start)
            sig_rows.append({"代號": code, "名稱": name, **{
                dd.STRATEGY_NAMES[s]: ("—" if "error" in v else ("持有" if v["state"] else "空手")
                                       + (f"（P↑ {v['p_up']:.0%}）" if "p_up" in v else ""))
                for s, v in sg.items()}})
        except FinMindError as e:
            errs.append(f"{code}：{e}")
    if rows:
        df = pd.DataFrame(rows)
        st.subheader(f"自選股（最新交易日 {max(r['日期'] for r in rows)}）")
        fmt = {c: "{:+.1%}" for c in ["漲跌%", "5日%", "20日%", "60日%", "距MA20", "距MA60"]}
        fmt.update({"收盤": "{:,.2f}", "RSI14": "{:.0f}", "MFI14": "{:.0f}", "量比(÷前5日)": "{:.2f}x"})

        def color(v):
            if isinstance(v, (int, float)) and not pd.isna(v):
                return f"color: {UP}" if v > 0 else (f"color: {DOWN}" if v < 0 else "")
            return ""
        st.dataframe(df.style.format(fmt).map(color, subset=list(fmt)[:6]), hide_index=True, width="stretch")
        st.caption("報酬用除權息還原價計算；量比 = 今日量 ÷ 前 5 日均量；RSI / MFI > 70–80 偏熱、< 20–30 偏冷。")
        st.subheader("各策略目前訊號（最後交易日收盤後）")
        st.dataframe(pd.DataFrame(sig_rows), hide_index=True, width="stretch")
        st.warning("這些策略在 40 檔股票、2018–2024 的開發期研究中，都**沒有**顯示出可以和運氣區分的選時能力"
                   "（詳見「訊號與預測」各股的歷史表現）。訊號只是規則的機械輸出，不是買賣建議。")
    for e in errs:
        st.error(e)

# ---------------- 走勢 ----------------
with tab_chart:
    c1, c2, c3 = st.columns([2, 2, 3])
    code = c1.selectbox("股票", list(watch), format_func=lambda c: f"{c} {watch[c]}", key="chart_code")
    span = c2.radio("期間", ["3 個月", "6 個月", "1 年", "3 年", "全部"], index=1, horizontal=True)
    extras = c3.multiselect("副圖指標", ["RSI14", "MFI14", "OBV"], default=["RSI14"])
    show_pat = st.checkbox("標出 K 線型態", value=True)
    prices, divs, adj = load(code, start)
    if prices.empty:
        st.error("沒有資料")
    else:
        days = {"3 個月": 63, "6 個月": 126, "1 年": 252, "3 年": 756, "全部": len(prices)}[span]
        view = prices.iloc[-days:]
        st.plotly_chart(kline_figure(f"{code} {watch[code]}", prices, adj, days, extras, show_pat), width="stretch")
        snap = dd.snapshot(code, watch[code], prices, adj)
        m = st.columns(6)
        m[0].metric("收盤", f"{snap['收盤']:,.2f}", pct(snap["漲跌%"]), delta_color="inverse")
        m[1].metric("20 日", pct(snap["20日%"]))
        m[2].metric("距 MA20", pct(snap["距MA20"]))
        m[3].metric("RSI14", f"{snap['RSI14']:.0f}")
        m[4].metric("量比", f"{snap['量比(÷前5日)']:.2f}x")
        m[5].metric("今日型態", snap["今日型態"])
        st.caption("台股紅漲綠跌（上方漲跌幅顏色已配合）。K 線為未還原價；除權息日會出現缺口，"
                   f"期間內除權息 {len(divs[divs['date'] >= view['date'].iloc[0]])} 次。")

# ---------------- 訊號與預測 ----------------
with tab_signal:
    code = st.selectbox("股票", list(watch), format_func=lambda c: f"{c} {watch[c]}", key="sig_code")
    _, _, adj = load(code, start)
    sg = signals(code, start)
    st.info("每個策略都附上它在 **開發期** （保留期之前）的歷史表現與顯著性檢定。"
            "p ≥ 0.05 代表這個策略的績效和「同樣進出頻率的亂買亂賣」分不出來，它的訊號就不值得參考。")
    for s in ("lgbm", "ma_cross", "candle_volume"):
        v = sg[s]
        with st.container(border=True):
            st.markdown(f"### {dd.STRATEGY_NAMES[s]}")
            if "error" in v:
                st.write(v["error"])
                continue
            a, b = st.columns([1, 2])
            with a:
                st.metric(f"{v['as_of']:%Y-%m-%d} 收盤後訊號", "持有" if v["state"] else "空手",
                          help="= 下一個交易日開盤時，這個規則要持有或空手")
                st.caption(f"此狀態自 {v['since']:%Y-%m-%d} 起")
                if "p_up" in v:
                    st.metric(f"模型預測：{dd.target_text(cfg)}的機率", f"{v['p_up']:.0%}",
                              help=f"機率 > {v['threshold']:.0%} 時訊號為持有")
            with b:
                if st.toggle("顯示開發期歷史表現（需計算 10–30 秒）", key=f"tr_{code}_{s}"):
                    try:
                        tr = track(code, s, start)
                    except Exception as e:
                        st.error(str(e))
                    else:
                        m, mb, sig = tr["metrics"], tr["bh"], tr["sig"]
                        st.markdown(
                            f"開發期 {m['start']:%Y-%m-%d} ~ {m['end']:%Y-%m-%d}：策略年化 **{m['cagr']:.1%}**、"
                            f"買進持有 **{mb['cagr']:.1%}**；最大回撤 {m['max_drawdown']:.1%} vs {mb['max_drawdown']:.1%}；"
                            f"來回 {m['round_trips']} 次、成本 {m['total_commission'] + m['total_tax']:,.0f} 元。")
                        verdict = ("可能有選時能力（單檔，需多檔驗證）" if sig["p_value"] < 0.05
                                   else "**無法和運氣區分**")
                        st.markdown(f"勝過 {sig['percentile']:.0%} 的同頻率隨機訊號，p = {sig['p_value']:.3f} → {verdict}")
                        if tr["folds"] is not None and len(tr["folds"]):
                            st.markdown(f"walk-forward 樣本外平均準確率 **{tr['folds']['accuracy'].mean():.1%}**"
                                        "（猜「每天都漲」約 50–54%）")
            if "proba" in v:
                hs = cfg["evaluation"]["holdout_start"]
                with st.expander(f"LightGBM 預測的歷史可信度（walk-forward 樣本外預測，只統計 {hs} 之前）"):
                    cal = dd.prediction_calibration(v["proba"], adj, cfg, before=hs)
                    st.dataframe(cal.style.format({"實際發生比例": "{:.1%}"}), hide_index=True)
                    st.caption(f"預測目標：{dd.target_text(cfg)}。如果模型有用，預測機率越高的區間，實際發生比例應該越高；各區間差不多 = 預測沒有資訊。")
                    acc = dd.rolling_accuracy(v["proba"], adj, cfg, before=hs)
                    fa = go.Figure(go.Scatter(x=acc.index, y=acc.values, line=dict(color=STRAT, width=1.5),
                                              name="近 60 日準確率"))
                    fa.add_hline(y=0.5, line=dict(color="#888888", dash="dot"))
                    fa.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%",
                                     yaxis_title="近 60 日準確率")
                    st.plotly_chart(fa, width="stretch")

# ---------------- 回測 ----------------
with tab_bt:
    c1, c2 = st.columns(2)
    code = c1.selectbox("股票", list(watch), format_func=lambda c: f"{c} {watch[c]}", key="bt_code")
    strat = c2.selectbox("策略", list(dd.STRATEGY_NAMES), format_func=lambda s: dd.STRATEGY_NAMES[s])
    st.caption(f"只回測開發期（{cfg['evaluation']['holdout_start']} 之前），含成本、證交稅、股利、漲跌停與流動性限制。")
    if st.button("執行回測", type="primary"):
        try:
            tr = track(code, strat, start)
        except Exception as e:
            st.error(str(e))
        else:
            m, mb, sig = tr["metrics"], tr["bh"], tr["sig"]
            tbl = pd.DataFrame([
                {"": dd.STRATEGY_NAMES[strat], "年化": m["cagr"], "總報酬": m["total_return"], "最大回撤": m["max_drawdown"],
                 "Sharpe": m["sharpe"], "持股比例": m["exposure"], "來回": m["round_trips"],
                 "成本": m["total_commission"] + m["total_tax"], "股利": m["total_dividends"]},
                {"": "買進持有", "年化": mb["cagr"], "總報酬": mb["total_return"], "最大回撤": mb["max_drawdown"],
                 "Sharpe": mb["sharpe"], "持股比例": mb["exposure"], "來回": mb["round_trips"],
                 "成本": mb["total_commission"] + mb["total_tax"], "股利": mb["total_dividends"]},
            ])
            st.dataframe(tbl.style.format({"年化": "{:.1%}", "總報酬": "{:.1%}", "最大回撤": "{:.1%}",
                                           "Sharpe": "{:.2f}", "持股比例": "{:.0%}", "成本": "{:,.0f}",
                                           "股利": "{:,.0f}"}), hide_index=True, width="stretch")
            st.markdown(f"顯著性：勝過 {sig['percentile']:.0%} 的同頻率隨機訊號，p = {sig['p_value']:.3f}")
            s0 = m["start"]
            eq, bh = tr["equity"][tr["equity"].index >= s0], tr["bh_equity"][tr["bh_equity"].index >= s0]
            fe = go.Figure()
            fe.add_trace(go.Scatter(x=eq.index, y=eq / eq.iloc[0], name=dd.STRATEGY_NAMES[strat], line=dict(color=STRAT, width=2)))
            fe.add_trace(go.Scatter(x=bh.index, y=bh / bh.iloc[0], name="買進持有（含股利）", line=dict(color=BH, width=2)))
            fe.update_layout(height=380, hovermode="x unified", yaxis_title="資產（起點 = 1）",
                             margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fe, width="stretch")
            with st.expander("成交明細"):
                st.dataframe(tr["trades"], hide_index=True, width="stretch")

st.caption("資料來源：FinMind。僅供研究與模擬，不構成投資建議；本工具沒有任何下單或券商連線功能。")
