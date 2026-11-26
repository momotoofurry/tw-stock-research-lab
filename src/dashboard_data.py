"""看盤介面的資料層（不含任何下單功能）。

- 走勢 / 指標用到今天為止的最新資料。
- 策略「目前訊號」用最新資料；但策略的「歷史表現」只用開發期（保留期之前），不偷看保留期。
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .adjust import adjust_for_dividends
from .backtest import Backtester
from .evaluate import STRATEGIES, generate_signals, run_stock, window_metrics
from .features import build_features
from .fetch import fetch_events, fetch_prices
from .kline import PATTERNS, detect_patterns
from .significance import random_benchmark

STRATEGY_NAMES = {"ma_cross": "均線交叉", "candle_volume": "K 線 + 量能", "lgbm": "LightGBM 預測"}


def load(stock_id: str, start: str, cfg: dict, end: str | None = None):
    end = end or date.today().isoformat()
    prices = fetch_prices(stock_id, start, end, cfg)
    divs = fetch_events(stock_id, start, end, cfg, prices=prices)
    return prices, divs, adjust_for_dividends(prices, divs)


def snapshot(stock_id: str, name: str, prices: pd.DataFrame, adj: pd.DataFrame) -> dict:
    """最新一天的狀態。報酬用還原價（含權息），價格顯示用未還原價。"""
    c = adj.set_index("date")["close"]
    f = build_features(adj).iloc[-1]
    pats = detect_patterns(adj).iloc[-1]
    last = prices.iloc[-1]

    def ret(n):
        return float(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else np.nan

    return {
        "代號": stock_id, "名稱": name, "日期": last["date"].date(), "收盤": float(last["close"]),
        "漲跌%": ret(1), "5日%": ret(5), "20日%": ret(20), "60日%": ret(60),
        "距MA20": float(f["close_ma20"]), "距MA60": float(f["close_ma60"]), "RSI14": float(f["rsi_14"]),
        "量比(÷前5日)": float(f["vol_ratio_5"] + 1), "MFI14": float(f["mfi_14"]),
        "今日型態": "、".join(PATTERNS[k].split("（")[0] for k, v in pats.items() if v) or "—",
    }


def current_signals(adj: pd.DataFrame, cfg: dict) -> dict:
    """每個策略在最後一個交易日收盤後的訊號（= 下一個交易日開盤要持有或空手）。"""
    out = {}
    for s in STRATEGIES:
        try:
            if s == "lgbm":
                from .strategies import lgbm
                sig, folds, proba = lgbm.generate_signals(adj, return_details=True, **cfg["strategies"]["lgbm"])
            else:
                sig, folds = generate_signals(s, adj, cfg)
                proba = None
        except Exception as e:  # 資料太短等
            out[s] = dict(error=str(e))
            continue
        valid = sig.dropna()
        if valid.empty:
            out[s] = dict(error="資料太短，還沒有訊號")
            continue
        state = int(valid.iloc[-1])
        changed = valid[valid != state]
        since = valid.index[0] if changed.empty else valid[valid.index > changed.index[-1]].index[0]
        info = dict(state=state, as_of=valid.index[-1], since=since, signal=sig)
        if proba is not None:
            info.update(proba=proba, p_up=float(proba.dropna().iloc[-1]), folds=folds,
                        threshold=cfg["strategies"]["lgbm"]["threshold"])
        out[s] = info
    return out


def track_record(stock_id: str, strategy: str, cfg: dict, start: str = "2018-01-01", n_random: int = 100) -> dict:
    """策略在開發期的表現與顯著性（不含保留期）。"""
    run = run_stock(stock_id, strategy, start, None, cfg, holdout=False)
    m, mb = window_metrics(run, cfg)["開發期"]
    s, e = run.windows["開發期"]
    st = random_benchmark(Backtester.from_config(cfg), run.prices, run.signal, m["cagr"], n_random,
                          dividends=run.dividends, start=s, end=e,
                          trading_days=cfg["backtest"]["trading_days_per_year"])
    return dict(metrics=m, bh=mb, sig=st, equity=run.res.equity, bh_equity=run.bh.equity,
                folds=run.folds, trades=run.res.trades)


def _outcomes(adj: pd.DataFrame, cfg: dict, before=None) -> pd.Series:
    """模型標籤的實際結果（定義同 config 的 horizon / hurdle）。先把價格截在 before 之前再算，
    所以開發期最後 horizon 天的結果是 NaN，不會用到保留期的價格。"""
    from .features import make_label
    lc = cfg["strategies"]["lgbm"]
    if before is not None:
        adj = adj[adj["date"] < pd.Timestamp(before)]
    return make_label(adj, lc.get("horizon", 1), lc.get("hurdle", 0.0))


def target_text(cfg: dict) -> str:
    lc = cfg["strategies"]["lgbm"]
    h, hurdle = lc.get("horizon", 1), lc.get("hurdle", 0.0)
    if h == 1 and hurdle == 0:
        return "下一交易日收盤上漲"
    return f"未來 {h} 個交易日報酬 > {hurdle:.1%}"


def prediction_calibration(proba: pd.Series, adj: pd.DataFrame, cfg: dict, before=None,
                           bins=(0, .4, .45, .5, .55, .6, 1)) -> pd.DataFrame:
    """LightGBM 歷史預測（全部是 walk-forward 樣本外）的校準：預測機率區間 vs 實際發生比例。
    before：只統計這天之前（保留期不納入）。"""
    df = pd.DataFrame({"p": _before(proba, before), "y": _outcomes(adj, cfg, before)}).dropna()
    df["區間"] = pd.cut(df["p"], bins=list(bins), include_lowest=True)
    g = df.groupby("區間", observed=True)["y"].agg(["count", "mean"]).reset_index()
    g.columns = ["預測機率", "次數", "實際發生比例"]
    g["預測機率"] = g["預測機率"].astype(str)
    return g


def rolling_accuracy(proba: pd.Series, adj: pd.DataFrame, cfg: dict, window: int = 60,
                     before=None) -> pd.Series:
    df = pd.DataFrame({"p": _before(proba, before), "y": _outcomes(adj, cfg, before)}).dropna()
    hit = ((df["p"] > cfg["strategies"]["lgbm"]["threshold"]) == (df["y"] > 0.5)).astype(float)
    return hit.rolling(window).mean()


def _before(s: pd.Series, before) -> pd.Series:
    return s if before is None else s[s.index < pd.Timestamp(before)]


def stock_name(stock_id: str, cfg: dict) -> tuple[str, str]:
    from .fetch import fetch_stock_info
    try:
        info = fetch_stock_info(cfg)
    except Exception:
        return "", ""
    row = info[info["stock_id"] == stock_id]
    return (row["stock_name"].iloc[0], row["industry_category"].iloc[0]) if len(row) else ("", "")


def analyze(prices: pd.DataFrame, adj: pd.DataFrame, holdout_start: str, trading_days: int = 252) -> dict:
    """個股技術面摘要與「波動範圍」統計預測。全部只用最後一個交易日（含）以前的資料，只描述現況，不是買賣建議。"""
    px = prices.set_index("date")
    c_raw, h, l = px["close"], px["high"], px["low"]
    c = adj.set_index("date")["close"]
    last = float(c_raw.iloc[-1])
    ma = {n: float(c_raw.rolling(n).mean().iloc[-1]) for n in (5, 20, 60, 120)}
    ma20 = c_raw.rolling(20).mean()
    ma20_slope = float(ma20.iloc[-1] / ma20.iloc[-21] - 1) if len(ma20.dropna()) > 21 else np.nan
    f = build_features(adj).iloc[-1]

    if ma[5] > ma[20] > ma[60]:
        align = "多頭排列（MA5 > MA20 > MA60）"
    elif ma[5] < ma[20] < ma[60]:
        align = "空頭排列（MA5 < MA20 < MA60）"
    else:
        align = "均線糾結 / 盤整"

    rets = np.log(c).diff()
    vol60 = float(rets.iloc[-60:].std())
    tr = pd.concat([h - l, (h - c_raw.shift()).abs(), (l - c_raw.shift()).abs()], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1])

    # 波動範圍：零漂移、對數常態近似，用近 60 日日波動率
    ranges = []
    for hz in (5, 20):
        s = vol60 * np.sqrt(hz)
        ranges.append(dict(天數=hz, 下緣_95=last * np.exp(-1.96 * s), 下緣_68=last * np.exp(-s),
                           上緣_68=last * np.exp(s), 上緣_95=last * np.exp(1.96 * s)))
    # 開發期內同一方法的實際涵蓋率（區間只用當時以前 60 日波動率）
    dev = c[c.index < pd.Timestamp(holdout_start)]
    dr = np.log(dev).diff()
    cover = {}
    for hz in (5, 20):
        sig = dr.rolling(60).std() * np.sqrt(hz)
        fwd = np.log(dev.shift(-hz) / dev)
        ok = pd.DataFrame({"s": sig, "f": fwd}).dropna()
        cover[hz] = float((ok["f"].abs() <= ok["s"]).mean()) if len(ok) else np.nan

    notes = [
        f"趨勢：{align}；收盤距 MA20 {last / ma[20] - 1:+.1%}、距 MA60 {last / ma[60] - 1:+.1%}；"
        f"MA20 近 20 日{'上升' if ma20_slope > 0 else '下降'} {ma20_slope:+.1%}",
        f"動能：RSI14 = {f['rsi_14']:.0f}（> 70 偏熱、< 30 偏冷）；近 20 日報酬 {f['ret_20']:+.1%}",
        f"量能：5 日均量 / 20 日均量 = {f['vol_ma5_ma20'] + 1:.2f}x；OBV 20 日斜率 {f['obv_slope_20']:+.2f}"
        f"（{'上漲日量較多' if f['obv_slope_20'] > 0 else '下跌日量較多'}）；收盤 / 20 日量加權均價 {f['close_vwap20']:+.1%}",
        f"波動：近 60 日年化波動率 {vol60 * np.sqrt(trading_days):.0%}；ATR14 = {atr14:,.2f} 元（約 {atr14 / last:.1%}）",
    ]
    levels = {
        "20 日高點": float(h.iloc[-20:].max()), "20 日低點": float(l.iloc[-20:].min()),
        "60 日高點": float(h.iloc[-60:].max()), "60 日低點": float(l.iloc[-60:].min()),
        "MA20": ma[20], "MA60": ma[60], "MA120": ma[120],
        "收盤 − 2×ATR": last - 2 * atr14,
    }
    return dict(last=last, date=c_raw.index[-1], align=align, notes=notes, levels=levels,
                ranges=pd.DataFrame(ranges), coverage=cover, vol60=vol60, atr14=atr14)


def lgbm_reliability(proba: pd.Series, adj: pd.DataFrame, cfg: dict, before=None) -> dict:
    """LightGBM 樣本外預測在 before 之前的準確率，和「一律猜多數方向」比較。標籤定義同 config。"""
    df = pd.DataFrame({"p": _before(proba, before), "y": _outcomes(adj, cfg, before)}).dropna()
    if df.empty:
        return dict(n=0, accuracy=np.nan, naive=np.nan)
    acc = float(((df["p"] > cfg["strategies"]["lgbm"]["threshold"]) == (df["y"] > 0.5)).mean())
    rate = float(df["y"].mean())
    return dict(n=len(df), accuracy=acc, naive=max(rate, 1 - rate))
