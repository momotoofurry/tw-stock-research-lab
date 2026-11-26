"""基本面資料的「時間點」對齊（point-in-time）。

每一筆基本面數字都附上「可取得日」available，只有 available <= 訊號日 的資料才能用：
- 月營收：FinMind 的 date 是次月 1 日，法定公布期限是次月 10 日 → available = date + 9 天（次月 10 日）
- 季財報：一般公司 Q1–Q3 季末 + 45 天、金融業季末 + 2 個月、年報 3/31；再加 1 天緩衝
用到的訊號日一律取「收盤後」，所以 available 當天收盤後即可使用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .research import fetch_dataset

REVENUE_LAG_DAYS = 9


def revenue_pit(stock_id: str, cfg: dict, start: str = "2016-01-01") -> pd.DataFrame:
    """月營收：period（所屬月份）、revenue、yoy、yoy3（近 3 個月合計年增）、available。"""
    df = fetch_dataset("TaiwanStockMonthRevenue", stock_id, start, cfg, ttl_hours=24 * 7)
    if df.empty:
        return pd.DataFrame(columns=["period", "revenue", "yoy", "yoy3", "available"])
    df = df.sort_values(["revenue_year", "revenue_month"]).drop_duplicates(["revenue_year", "revenue_month"], keep="last")
    df["period"] = pd.to_datetime(dict(year=df["revenue_year"], month=df["revenue_month"], day=1))
    s = df.set_index("period")["revenue"].astype(float).asfreq("MS")
    out = pd.DataFrame({"revenue": s})
    out["yoy"] = s / s.shift(12) - 1
    r3 = s.rolling(3).sum()
    out["yoy3"] = r3 / r3.shift(12) - 1
    out["available"] = out.index + pd.offsets.MonthBegin(1) + pd.Timedelta(days=REVENUE_LAG_DAYS)
    return out.reset_index().rename(columns={"index": "period"}).dropna(subset=["revenue"])


def report_available(quarter_end: pd.Timestamp, financial: bool) -> pd.Timestamp:
    """財報最晚公布日 + 1 天緩衝。一般公司 Q1–Q3 季末 + 45 天；金融業保守用季末 + 2 個月；
    年報一律 3/31（用月份位移，閏年也正確）。"""
    if quarter_end.month == 12:
        due = quarter_end + pd.DateOffset(months=3)
    elif financial:
        due = quarter_end + pd.DateOffset(months=2)
    else:
        due = quarter_end + pd.Timedelta(days=45)
    return pd.Timestamp(due) + pd.Timedelta(days=1)


def is_financial(stock_id: str, cfg: dict) -> bool:
    from .fetch import fetch_stock_info
    try:
        info = fetch_stock_info(cfg)
    except Exception:
        return stock_id.startswith("28")
    row = info[info["stock_id"] == stock_id]
    return bool(len(row)) and "金融" in str(row["industry_category"].iloc[0])


def eps_pit(stock_id: str, cfg: dict, start: str = "2016-01-01") -> pd.DataFrame:
    """季 EPS：quarter_end、eps、eps_ttm（近四季合計）、available。"""
    df = fetch_dataset("TaiwanStockFinancialStatements", stock_id, start, cfg, ttl_hours=24 * 7)
    if df.empty:
        return pd.DataFrame(columns=["quarter_end", "eps", "eps_ttm", "available"])
    e = df[df["type"] == "EPS"].sort_values("date").drop_duplicates("date", keep="last")
    out = pd.DataFrame({"quarter_end": e["date"].values, "eps": e["value"].astype(float).values})
    out["eps_ttm"] = out["eps"].rolling(4).sum()
    fin = is_financial(stock_id, cfg)
    out["available"] = [report_available(q, fin) for q in out["quarter_end"]]
    return out


def as_of(table: pd.DataFrame, day: pd.Timestamp, col: str) -> float:
    """day 收盤後可取得的最新一筆 col；沒有就 NaN。"""
    t = table[table["available"] <= day]
    return float(t[col].iloc[-1]) if len(t) and not pd.isna(t[col].iloc[-1]) else np.nan
