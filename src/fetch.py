"""從 FinMind 抓台股日K（TaiwanStockPrice），本地 CSV 快取。

- token 從環境變數 FINMIND_TOKEN 讀取；沒有也能跑（免費額度較低）。token 不寫入任何檔案。
- 已快取的日期區間不重抓，只補抓缺的頭尾。
- 遇到限流（HTTP 402/429 或 status 402）會等待並重試，用完次數拋出 FinMindRateLimitError。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import ROOT, load_config

PRICE_COLUMNS = ["date", "stock_id", "open", "high", "low", "close", "volume", "spread"]
DIVIDEND_COLUMNS = ["date", "stock_id", "before_price", "after_price", "stock_and_cache_dividend"]


class FinMindError(RuntimeError):
    pass


class FinMindRateLimitError(FinMindError):
    pass


def _cache_path(stock_id: str, data_dir: str | Path, dataset: str = "TaiwanStockPrice") -> Path:
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{dataset}_{stock_id}.csv"


def _request(dataset: str, params: dict, cfg: dict, sleep=time.sleep) -> list[dict]:
    """呼叫 FinMind API，處理限流與網路錯誤重試。"""
    fm = cfg["finmind"]
    headers = {}
    from .notify import _env
    token = _env("FINMIND_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    query = {"dataset": dataset, **params}
    wait = fm["retry_wait_seconds"]
    last_err = ""
    for attempt in range(fm["max_retries"] + 1):
        try:
            resp = requests.get(fm["url"], params=query, headers=headers, timeout=fm["timeout_seconds"])
        except requests.RequestException as e:
            last_err = f"網路錯誤：{e}"
        else:
            body = {}
            try:
                body = resp.json()
            except ValueError:
                pass
            status = body.get("status", resp.status_code)
            if resp.status_code in (402, 429) or status in (402, 429):
                last_err = f"限流（HTTP {resp.status_code}，訊息：{body.get('msg', resp.text[:200])}）"
            elif resp.status_code != 200 or status != 200:
                raise FinMindError(
                    f"FinMind 回傳錯誤 HTTP {resp.status_code} status={status} msg={body.get('msg', resp.text[:200])}"
                )
            else:
                return body.get("data", [])
        if attempt < fm["max_retries"]:
            print(f"[fetch] {last_err}；{wait} 秒後重試（第 {attempt + 1}/{fm['max_retries']} 次）")
            sleep(wait)
            wait *= 2
    hint = "" if token else "未設定 FINMIND_TOKEN，免費額度較低；可設定環境變數後再試。"
    raise FinMindRateLimitError(
        f"FinMind 請求失敗，已重試 {fm['max_retries']} 次仍失敗：{last_err}。{hint}"
    )


def _normalize_prices(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    df = pd.DataFrame(rows).rename(
        columns={"max": "high", "min": "low", "Trading_Volume": "volume"}
    )
    df["date"] = pd.to_datetime(df["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    if "spread" not in df:
        df["spread"] = float("nan")
    return df[PRICE_COLUMNS]


def _normalize_dividends(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=DIVIDEND_COLUMNS)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    return df[DIVIDEND_COLUMNS]


def _read_cache(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, dtype={"stock_id": str}, parse_dates=["date"])
    return df[columns]


def _read_coverage(meta_path: Path, legacy_start: Path, cache: pd.DataFrame):
    """已請求過的日期區間 [start, end]。沒有紀錄時用快取資料的最早 / 最晚日期。"""
    if meta_path.exists():
        a, b = meta_path.read_text().split()[:2]
        return pd.Timestamp(a), pd.Timestamp(b)
    if cache.empty:
        return None
    cmin, cmax = cache["date"].min(), cache["date"].max()
    if legacy_start.exists():
        cmin = min(cmin, pd.Timestamp(legacy_start.read_text().strip()))
    return cmin, cmax


def _fetch_cached(dataset: str, stock_id: str, start: str, end: str, cfg: dict, columns: list[str],
                  normalize, sleep, refresh: bool = False) -> pd.DataFrame:
    """只抓快取沒涵蓋到的頭尾區間。涵蓋區間記在 .range 檔，最多記到昨天（今天的資料可能還沒出來）。
    同一個尾段在 recheck_hours 內查過就不再查（省 API 額度）；refresh=True 強制重查。"""
    path = _cache_path(stock_id, cfg["data_dir"], dataset)
    meta_path = path.with_suffix(".range")
    cache = _read_cache(path, columns)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    cov = _read_coverage(meta_path, path.with_suffix(".start"), cache)
    if offline(cfg):  # 離線模式：只用快取，完全不連網
        return cache[(cache["date"] >= start_ts) & (cache["date"] <= end_ts)].reset_index(drop=True)
    if cov is None:
        ranges = [(start_ts, end_ts)]
    else:
        ranges = []
        if start_ts < cov[0]:
            ranges.append((start_ts, cov[0] - pd.Timedelta(days=1)))
        if end_ts > cov[1]:
            checked = meta_path.read_text().split()[2:3] if meta_path.exists() else []
            hours = cfg["finmind"].get("recheck_hours", 2)
            recent = checked and (pd.Timestamp.now() - pd.Timestamp(checked[0])) < pd.Timedelta(hours=hours)
            tail_is_new_days_only = (end_ts - cov[1]).days <= 4  # 只差最近幾天（含週末）才套用
            if refresh or not (recent and tail_is_new_days_only):
                ranges.append((cov[1] + pd.Timedelta(days=1), end_ts))

    new_parts = []
    for s, e in ranges:
        if s > e:
            continue
        rows = _request(dataset, {
            "data_id": stock_id, "start_date": s.strftime("%Y-%m-%d"), "end_date": e.strftime("%Y-%m-%d"),
        }, cfg, sleep=sleep)
        print(f"[fetch] {dataset} {stock_id} {s.date()}~{e.date()} 新抓 {len(rows)} 筆")
        new_parts.append(normalize(rows))

    if ranges:
        parts = [p for p in [cache, *new_parts] if not p.empty]
        if parts:
            cache = pd.concat(parts, ignore_index=True).drop_duplicates("date", keep="last") \
                .sort_values("date").reset_index(drop=True)
        cache.to_csv(path, index=False)
        yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        new_start = start_ts if cov is None else min(cov[0], start_ts)
        new_end = min(end_ts, yesterday) if cov is None else max(cov[1], min(end_ts, yesterday))
        if not cache.empty:
            new_end = max(new_end, cache["date"].max())
        meta_path.write_text(f"{new_start:%Y-%m-%d} {new_end:%Y-%m-%d} {pd.Timestamp.now():%Y-%m-%dT%H:%M}")

    return cache[(cache["date"] >= start_ts) & (cache["date"] <= end_ts)].reset_index(drop=True)


def fetch_prices(stock_id: str, start: str, end: str, cfg: dict | None = None,
                 sleep=time.sleep, refresh: bool = False) -> pd.DataFrame:
    """回傳 [start, end] 的日K（未還原）。"""
    cfg = cfg or load_config()
    out = _fetch_cached("TaiwanStockPrice", stock_id, start, end, cfg, PRICE_COLUMNS, _normalize_prices, sleep,
                        refresh)
    # 停牌 / 無成交日 FinMind 可能給 0 價格，回測時無法成交，直接排除
    out = out[(out["open"] > 0) & (out["close"] > 0)]
    return out.reset_index(drop=True)


def fetch_dividends(stock_id: str, start: str, end: str, cfg: dict | None = None,
                    sleep=time.sleep) -> pd.DataFrame:
    """除權息結果（TaiwanStockDividendResult）：date = 除權息日，before_price / after_price = 前一日收盤 / 參考價。"""
    cfg = cfg or load_config()
    out = _fetch_cached("TaiwanStockDividendResult", stock_id, start, end, cfg, DIVIDEND_COLUMNS,
                        _normalize_dividends, sleep)
    return out[(out["before_price"] > 0) & (out["after_price"] > 0)].reset_index(drop=True)


def fetch_trading_calendar(start: str, end: str, cfg: dict | None = None,
                           sleep=time.sleep) -> pd.DatetimeIndex | None:
    """台股交易日曆（TaiwanStockTradingDate）；抓不到回傳 None。"""
    cfg = cfg or load_config()
    try:
        rows = _request("TaiwanStockTradingDate", {"start_date": start, "end_date": end}, cfg, sleep=sleep)
    except FinMindError as e:
        print(f"[fetch] 無法取得交易日曆：{e}")
        return None
    if not rows:
        return None
    return pd.DatetimeIndex(sorted(pd.to_datetime([r["date"] for r in rows])))


def missing_dates(df: pd.DataFrame, start: str, end: str,
                  calendar: pd.DatetimeIndex | None = None) -> tuple[list[pd.Timestamp], str]:
    """回傳缺漏日期與比對基準。有交易日曆用日曆，否則用平日（會包含國定假日）。"""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if calendar is not None:
        expected = calendar[(calendar >= start_ts) & (calendar <= end_ts)]
        basis = "FinMind 交易日曆"
    else:
        expected = pd.bdate_range(start_ts, end_ts)
        basis = "平日（未扣除國定假日，僅供參考）"
    have = set(pd.to_datetime(df["date"]))
    return [d for d in expected if d not in have], basis


def fetch_stock_info(cfg: dict | None = None, max_age_days: int = 7, sleep=time.sleep) -> pd.DataFrame:
    """台股代號 / 名稱 / 產業（TaiwanStockInfo），快取 max_age_days 天。"""
    cfg = cfg or load_config()
    path = _cache_path("all", cfg["data_dir"], "TaiwanStockInfo")
    if path.exists() and (offline(cfg) or time.time() - path.stat().st_mtime < max_age_days * 86400):
        return pd.read_csv(path, dtype=str)
    rows = _request("TaiwanStockInfo", {}, cfg, sleep=sleep)
    df = pd.DataFrame(rows)[["stock_id", "stock_name", "industry_category", "type"]].drop_duplicates("stock_id")
    df.to_csv(path, index=False)
    return df.astype(str)


def offline(cfg: dict) -> bool:
    """config 的 finmind.offline 為 True 時，所有資料只讀本機快取（研究重跑時避免撞到每小時查詢上限）。"""
    return bool(cfg.get("finmind", {}).get("offline"))


def _cached_rows(dataset: str, key: str, params: dict, cfg: dict, ttl_hours: float, sleep=time.sleep) -> list[dict]:
    path = _cache_path(key, cfg["data_dir"], dataset).with_suffix(".json")
    if path.exists() and (offline(cfg) or time.time() - path.stat().st_mtime < ttl_hours * 3600):
        return json.loads(path.read_text(encoding="utf-8"))
    if offline(cfg):
        return []
    rows = _request(dataset, params, cfg, sleep=sleep)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def _next_trading_day(day: pd.Timestamp, dates: pd.DatetimeIndex):
    i = dates.searchsorted(day)
    return dates[i] if i < len(dates) else None


def official_corporate_events(stock_id: str, prices: pd.DataFrame, dividends: pd.DataFrame, start: str,
                              cfg: dict, ttl_hours: float = 24 * 7, sleep=time.sleep) -> pd.DataFrame:
    """官方資料的公司行動與漏列的除權息（全部有公告來源，不用 spread 推算）：
    - 減資恢復買賣參考價（TaiwanStockCapitalReductionReferencePrice）
    - 分割 / 面額變更（TaiwanStockSplitPrice、TaiwanStockParValueChange，全市場一份）
    - 股利政策公告（TaiwanStockDividend）中，除權息日不在除權息結果表裡的：現金部分記 dividend、
      股票股利記 corporate（股數 ×(1 + 每股配股元 / 10)）。公告的除權息日不是交易日（例如颱風休市）時順延。
    """
    cols = DIVIDEND_COLUMNS + ["kind", "source"]
    if prices.empty:
        return pd.DataFrame(columns=cols)
    p = prices.sort_values("date")
    dates = pd.DatetimeIndex(p["date"])
    prev_close = p.set_index("date")["close"].shift(1)
    rows = []
    for r in _cached_rows("TaiwanStockCapitalReductionReferencePrice", stock_id,
                          {"data_id": stock_id, "start_date": "2010-01-01"}, cfg, ttl_hours, sleep):
        if float(r.get("PostReductionReferencePrice") or 0) > 0:
            rows.append(dict(date=pd.Timestamp(r["date"]), before_price=float(r["ClosingPriceonTheLastTradingDay"]),
                             after_price=float(r["PostReductionReferencePrice"]), kind="corporate", source="減資"))
    for ds, b, a in (("TaiwanStockSplitPrice", "before_price", "after_price"),
                     ("TaiwanStockParValueChange", "before_close", "after_ref_close")):
        for r in _cached_rows(ds, "all", {"start_date": "2010-01-01"}, cfg, ttl_hours, sleep):
            if str(r["stock_id"]) == stock_id and float(r.get(a) or 0) > 0:
                rows.append(dict(date=pd.Timestamp(r["date"]), before_price=float(r[b]), after_price=float(r[a]),
                                 kind="corporate", source=ds))
    known = set(pd.to_datetime(dividends["date"])) if len(dividends) else set()
    for r in _cached_rows("TaiwanStockDividend", stock_id, {"data_id": stock_id, "start_date": "2010-01-01"}, cfg,
                          ttl_hours, sleep):
        cash = float(r.get("CashEarningsDistribution") or 0) + float(r.get("CashStatutorySurplus") or 0)
        stock = float(r.get("StockEarningsDistribution") or 0) + float(r.get("StockStatutorySurplus") or 0)
        for amount, key, kind in ((cash, "CashExDividendTradingDate", "dividend"),
                                  (stock, "StockExDividendTradingDate", "corporate")):
            if amount <= 0 or not r.get(key):
                continue
            day = _next_trading_day(pd.Timestamp(r[key]), dates)
            if day is None or day in known or pd.isna(prev_close.get(day, np.nan)):
                continue
            pc = float(prev_close[day])
            if kind == "dividend":
                rows.append(dict(date=day, before_price=pc, after_price=pc - amount, kind="dividend", source="股利公告"))
            else:
                rows.append(dict(date=day, before_price=1 + amount / 10, after_price=1.0, kind="corporate",
                                 source="股利公告（配股）"))
    out = pd.DataFrame(rows, columns=["date", "before_price", "after_price", "kind", "source"])
    out = out.drop_duplicates(["date", "kind"], keep="first")      # 分割與面額變更可能重複列
    out = out[(out["date"] >= dates[0]) & (out["date"] <= dates[-1])]
    out["stock_id"] = stock_id
    out["stock_and_cache_dividend"] = np.nan
    return out[cols].reset_index(drop=True)


def unexplained_gaps(prices: pd.DataFrame, events: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    """檢查用：參考價（收盤 − spread）和前一日收盤差超過 tol、卻沒有任何官方事件可解釋的日子。只記錄不套用
    （spread 的定義不一定等於官方參考價，直接拿來調整會出錯）。"""
    if prices.empty or "spread" not in prices:
        return pd.DataFrame(columns=["date", "prev_close", "ref"])
    p = prices.sort_values("date").reset_index(drop=True)
    ref = p["close"] - p["spread"]
    prev = p["close"].shift(1)
    known = set(pd.to_datetime(events["date"])) if len(events) else set()
    m = (ref / prev - 1).abs().gt(tol) & ref.gt(0) & ~p["date"].isin(known)
    return pd.DataFrame({"date": p.loc[m, "date"], "prev_close": prev[m], "ref": ref[m]}).reset_index(drop=True)


def fetch_events(stock_id: str, start: str, end: str, cfg: dict | None = None, prices: pd.DataFrame | None = None,
                 sleep=time.sleep) -> pd.DataFrame:
    """價格事件：除權息結果（dividend，現金入帳）＋ 官方公司行動與漏列除權息（見 official_corporate_events）。
    同一天先以原股數入帳現金股利，再調整股數。"""
    cfg = cfg or load_config()
    div = fetch_dividends(stock_id, start, end, cfg, sleep=sleep).assign(kind="dividend", source="除權息結果")
    prices = fetch_prices(stock_id, start, end, cfg, sleep=sleep) if prices is None else prices
    corp = official_corporate_events(stock_id, prices, div, start, cfg, sleep=sleep)
    parts = [d for d in (div, corp) if len(d)]
    if not parts:
        return pd.DataFrame(columns=DIVIDEND_COLUMNS + ["kind", "source"])
    return pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
