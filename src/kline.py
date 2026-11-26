"""K 線型態與量價指標。

所有型態 / 指標的第 t 列只用第 t 天收盤以前的資料（含當天 OHLCV）。
`pattern_stats` 會用到未來報酬，只用來做事後統計，絕不能拿來當特徵或訊號。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PATTERNS = {
    "doji": "十字線",
    "hammer": "錘子（下影線長，跌勢後）",
    "shooting_star": "流星（上影線長，漲勢後）",
    "bull_engulfing": "多頭吞噬",
    "bear_engulfing": "空頭吞噬",
    "three_white_soldiers": "紅三兵",
    "three_black_crows": "黑三兵",
    "long_red": "長紅（實體 > 近 20 日平均實體 2 倍）",
    "long_black": "長黑（實體 > 近 20 日平均實體 2 倍）",
    "volume_breakout": "帶量突破 20 日高點",
    "volume_breakdown": "帶量跌破 20 日低點",
}


def candle_parts(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.set_index("date").sort_index()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    p = pd.DataFrame(index=df.index)
    p["body"] = (c - o) / o
    p["upper_shadow"] = (h - np.maximum(o, c)) / c
    p["lower_shadow"] = (np.minimum(o, c) - l) / c
    p["body_ratio"] = (c - o).abs() / rng           # 實體佔整根 K 線比例
    p["close_pos"] = (c - l) / rng                  # 收在當日區間的位置（0 = 最低，1 = 最高）
    return p


def volume_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.set_index("date").sort_index()
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"].astype(float)
    f = pd.DataFrame(index=df.index)
    f["vol_ratio_5"] = v / v.rolling(5).mean().shift(1) - 1      # 今日量 vs 前 5 日均量
    f["vol_ma5_ma20"] = v.rolling(5).mean() / v.rolling(20).mean() - 1
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_slope_20"] = (obv - obv.shift(20)) / v.rolling(20).sum()  # 20 日 OBV 變化 / 20 日總量
    vwap20 = (c * v).rolling(20).sum() / v.rolling(20).sum()
    f["close_vwap20"] = c / vwap20 - 1
    f["pv_corr_20"] = c.pct_change().rolling(20).corr(v.pct_change())
    tp = (h + l + c) / 3
    mf = tp * v
    # 用相對容忍度判斷漲跌：還原價等比例縮放後，(h+l+c) 相等的兩天可能因浮點誤差變成不相等
    chg = tp / tp.shift(1) - 1
    pos = mf.where(chg > 1e-9, 0.0).rolling(14).sum()
    neg = mf.where(chg < -1e-9, 0.0).rolling(14).sum()
    f["mfi_14"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    return f.replace([np.inf, -np.inf], np.nan)


def detect_patterns(prices: pd.DataFrame) -> pd.DataFrame:
    """回傳布林表，每欄一種型態（欄名見 PATTERNS）。"""
    df = prices.set_index("date").sort_index()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    p = candle_parts(prices)
    body_abs = (c - o).abs()
    avg_body = body_abs.rolling(20).mean().shift(1)
    up_k, down_k = c > o, c < o    # 台股慣例：up_k = 紅 K，down_k = 黑（綠）K
    trend5 = c.pct_change(5)
    vol_avg20 = v.rolling(20).mean().shift(1)
    small_body = body_abs <= 0.3 * (h - l)

    out = pd.DataFrame(index=df.index)
    out["doji"] = p["body_ratio"] < 0.1
    out["hammer"] = small_body & (p["lower_shadow"] * c >= 2 * body_abs) & (p["upper_shadow"] * c <= body_abs) \
        & (trend5.shift(1) < 0)
    out["shooting_star"] = small_body & (p["upper_shadow"] * c >= 2 * body_abs) & (p["lower_shadow"] * c <= body_abs) \
        & (trend5.shift(1) > 0)
    po, pc = o.shift(1), c.shift(1)
    out["bull_engulfing"] = down_k.shift(1, fill_value=False) & up_k & (o <= pc) & (c >= po) & (body_abs > (pc - po).abs())
    out["bear_engulfing"] = up_k.shift(1, fill_value=False) & down_k & (o >= pc) & (c <= po) & (body_abs > (pc - po).abs())
    rising = (c > c.shift(1))
    falling = (c < c.shift(1))
    out["three_white_soldiers"] = (up_k & rising) & (up_k & rising).shift(1, fill_value=False) \
        & (up_k & rising).shift(2, fill_value=False)
    out["three_black_crows"] = (down_k & falling) & (down_k & falling).shift(1, fill_value=False) \
        & (down_k & falling).shift(2, fill_value=False)
    out["long_red"] = up_k & (body_abs > 2 * avg_body)
    out["long_black"] = down_k & (body_abs > 2 * avg_body)
    hi20, lo20 = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)
    out["volume_breakout"] = (c > hi20) & (v > 1.5 * vol_avg20)
    out["volume_breakdown"] = (c < lo20) & (v > 1.5 * vol_avg20)
    return out.fillna(False).astype(bool)


def pattern_stats(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """事後統計（用到未來資料，只做研究報表）：型態出現後，第 t+1 天開盤買、第 t+1+horizon 天開盤賣的報酬。"""
    df = prices.set_index("date").sort_index()
    fwd = df["open"].shift(-(horizon + 1)) / df["open"].shift(-1) - 1
    base = fwd.dropna()
    pats = detect_patterns(prices)
    rows = []
    for name, desc in PATTERNS.items():
        r = fwd[pats[name]].dropna()
        rows.append(dict(pattern=name, 說明=desc, 次數=len(r),
                         平均報酬=r.mean() if len(r) else np.nan,
                         上漲比例=(r > 0).mean() if len(r) else np.nan,
                         超額報酬=(r.mean() - base.mean()) if len(r) else np.nan))
    rows.append(dict(pattern="all_days", 說明="所有交易日（基準）", 次數=len(base), 平均報酬=base.mean(),
                     上漲比例=(base > 0).mean(), 超額報酬=0.0))
    return pd.DataFrame(rows)
