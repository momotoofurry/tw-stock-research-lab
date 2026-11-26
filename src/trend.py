"""個股趨勢分析：多時間框架方向與品質、ADX 趨勢強度、支撐壓力、趨勢階段、股性（趨勢延續 / 均值回歸）。

全部只用最後一個交易日（含）以前的資料，只描述現況與歷史統計，不是買賣建議。
價格請傳入「還原權息」的價格（報酬、斜率才不會被除權息干擾）；支撐壓力另外用未還原價計算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FRAMES = {"短期": 20, "中期": 60, "長期": 240}


def regression_trend(close: pd.Series, n: int) -> dict:
    """近 n 日 log 價格線性回歸：slope = 回歸線在這 n 日的漲跌幅、R²（趨勢品質，越接近 1 越像一條直線）、方向。"""
    y = np.log(close.iloc[-n:].to_numpy(dtype=float))
    if len(y) < n:
        return dict(slope=np.nan, r2=np.nan, label="資料不足")
    x = np.arange(n)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else 0.0
    slope = float(np.exp(b * (n - 1)) - 1)
    if r2 >= 0.5 and slope > 0.02:
        label = "上升"
    elif r2 >= 0.5 and slope < -0.02:
        label = "下降"
    elif r2 >= 0.25:
        label = "偏升" if slope > 0 else "偏降"
    else:
        label = "盤整"
    return dict(slope=slope, r2=float(r2), label=label)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    """Wilder ADX / +DI / −DI。ADX > 25 一般視為趨勢明顯、< 20 視為沒有趨勢。"""
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    alpha = 1 / n
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=n).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pd.DataFrame({"adx": dx.ewm(alpha=alpha, adjust=False, min_periods=n).mean(), "pdi": pdi, "mdi": mdi})


def swing_levels(high: pd.Series, low: pd.Series, close: float, lookback: int = 250, k: int = 5) -> dict:
    """近 lookback 日的轉折高低點（前後各 k 日內的最高 / 最低）中，離收盤最近的支撐與壓力。
    轉折點需要右側 k 日確認，所以最後 k 日不會產生新的轉折點（不偷看未來）。
    注意：只能用來描述序列「最後一天」的狀態；不可對歷史每一天逐日計算或當成特徵 / 訊號
    （第 i 天的轉折點要等到 i + k 才確認，逐日使用會偷看未來）。"""
    h, l = high.iloc[-lookback:], low.iloc[-lookback:]
    win = 2 * k + 1
    piv_h = h[(h == h.rolling(win, center=True).max())].dropna()
    piv_l = l[(l == l.rolling(win, center=True).min())].dropna()
    res = sorted(v for v in piv_h if v > close)
    sup = sorted((v for v in piv_l if v < close), reverse=True)
    return dict(resistance=res[:2], support=sup[:2])


def character(close: pd.Series, before=None, weeks: int = 260) -> dict:
    """股性：週報酬的一階自我相關。顯著為正 → 趨勢延續型；顯著為負 → 均值回歸型；否則 → 接近隨機漫步。
    before：只用這天之前的資料（保留期不納入）。"""
    c = close if before is None else close[close.index < pd.Timestamp(before)]
    w = c.resample("W-FRI").last().pct_change().dropna().iloc[-weeks:]
    if len(w) < 52:
        return dict(ac=np.nan, n=len(w), label="資料不足")
    ac = float(w.autocorr(1))
    bound = 2 / np.sqrt(len(w))
    label = "趨勢延續型" if ac > bound else ("均值回歸型" if ac < -bound else "接近隨機（無明顯慣性）")
    return dict(ac=ac, n=len(w), bound=float(bound), label=label)


def phase(close: pd.Series) -> str:
    """依價格與均線的相對位置描述趨勢階段。"""
    ma20, ma60, ma120 = (close.rolling(n).mean() for n in (20, 60, 120))
    c, a, b, d = close.iloc[-1], ma20.iloc[-1], ma60.iloc[-1], ma120.iloc[-1]
    rising60 = ma60.iloc[-1] > ma60.iloc[-21]
    if c > a > b > d and rising60:
        return "上升趨勢中（價格 > 月線 > 季線 > 半年線，季線上揚）"
    if b > d and rising60 and c < a and c > b:
        return "上升趨勢中的回檔（跌破月線、仍在季線上，季線上揚）"
    if b > d and c < b:
        return "上升趨勢轉弱（跌破季線）"
    if c < a < b < d and not rising60:
        return "下降趨勢中（價格 < 月線 < 季線 < 半年線，季線下彎）"
    if b < d and not rising60 and c > a and c < b:
        return "下降趨勢中的反彈（站上月線、仍在季線下，季線下彎）"
    if b < d and c > b:
        return "下降趨勢轉強（站上季線）"
    return "盤整 / 方向不明"


def analyze_trend(adj: pd.DataFrame, raw: pd.DataFrame, before=None) -> dict:
    """adj：還原價日K；raw：未還原日K（支撐壓力用）。before：股性統計的截止日。"""
    a = adj.set_index("date").sort_index()
    r = raw.set_index("date").sort_index()
    c = a["close"]
    frames = {k: regression_trend(c, n) for k, n in FRAMES.items()}
    ad = adx(a["high"], a["low"], c).iloc[-1]
    lv = swing_levels(r["high"], r["low"], float(r["close"].iloc[-1]))
    ch = character(c, before)
    strength = ("趨勢明顯" if ad["adx"] > 25 else "沒有明顯趨勢" if ad["adx"] < 20 else "趨勢普通")
    direction = "多方較強（+DI > −DI）" if ad["pdi"] > ad["mdi"] else "空方較強（−DI > +DI）"
    return dict(date=c.index[-1], close=float(r["close"].iloc[-1]), frames=frames,
                adx=float(ad["adx"]), pdi=float(ad["pdi"]), mdi=float(ad["mdi"]), strength=strength,
                direction=direction, levels=lv, character=ch, phase=phase(c))


def trend_lines(t: dict) -> list[str]:
    fr = "；".join(f"{k} {v['label']}（回歸線 {FRAMES[k]} 日 {v['slope']:+.1%}、R² {v['r2']:.2f}）"
                  for k, v in t["frames"].items() if not np.isnan(v["slope"]))
    L = [f"多時間框架：{fr}",
         f"趨勢強度：ADX {t['adx']:.0f}（{t['strength']}），{t['direction']}",
         f"趨勢階段：{t['phase']}"]
    lv = t["levels"]
    L.append("最近壓力：" + ("、".join(f"{v:,.2f}（{v / t['close'] - 1:+.1%}）" for v in lv["resistance"]) or "近一年無")
             + "；最近支撐：" + ("、".join(f"{v:,.2f}（{v / t['close'] - 1:+.1%}）" for v in lv["support"]) or "近一年無"))
    ch = t["character"]
    if not np.isnan(ch["ac"]):
        L.append(f"股性：{ch['label']}（週報酬自我相關 {ch['ac']:+.2f}，顯著門檻 ±{ch['bound']:.2f}，{ch['n']} 週）")
    return L
