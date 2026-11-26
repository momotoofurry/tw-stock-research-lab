"""決策日誌：記下每筆交易當下自己寫的論點 / 預計持有 / 退出條件，事後和實際結果並列。

只描述使用者自己的紀錄與之後發生的事：不判斷論點對錯、不評論好壞、不建議下一步，沒有任何下單功能。
資料存在 data/decisions.csv（個人資料，data/ 已 gitignore），只追加；報告 reports/decisions_*.md 也 gitignore。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .dashboard_data import load
from .portfolio import trades_path

DECISION_COLUMNS = ["recorded_at", "date", "side", "stock_id", "shares", "price", "thesis", "horizon_months",
                    "exit_if"]
BENCH = "0050"
MIN_SAMPLE = 10
SIDE_ZH = {"buy": "買進", "sell": "賣出"}


def _dir(cfg: dict, key: str) -> Path:
    d = Path(cfg[key])
    return d if d.is_absolute() else ROOT / d


def decisions_path(cfg: dict) -> Path:
    return _dir(cfg, "data_dir") / "decisions.csv"


def record_decision(cfg: dict, day: str, side: str, stock_id: str, shares: float, price: float,
                    thesis: str = "", horizon_months: float | None = None, exit_if: str = "") -> bool:
    """論點 / 預計持有 / 退出條件任一有值才附加一列到 decisions.csv（只追加）；三者都空什麼都不做。"""
    if not (thesis or exit_if or horizon_months is not None):
        return False
    row = dict(recorded_at=pd.Timestamp.now().isoformat(timespec="seconds"), date=day, side=side,
               stock_id=str(stock_id), shares=shares, price=price, thesis=thesis, horizon_months=horizon_months,
               exit_if=exit_if)
    p = decisions_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    pd.DataFrame([row], columns=DECISION_COLUMNS).to_csv(p, mode="a", header=new, index=False, encoding="utf-8")
    return True


def _read(p: Path, columns: list[str]) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(p, dtype={"stock_id": str}, keep_default_na=False)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load(stock_id: str, start: str, cfg: dict):
    """(未還原價, 事件, 還原價)；還原價是向前還原，報酬用它才含息。可被測試替換。"""
    return load(stock_id, start, cfg)


def _total_return(book: tuple, day: pd.Timestamp, price: float, end: pd.Timestamp | None,
                  end_price: float | None = None):
    """含息總報酬。起點 = 實際成交價 × (決策日還原價 / 決策日未還原收盤)，把成交價換算到還原價座標；
    終點 = end（None 則最新資料日）的還原收盤；有 end_price（實際賣出成交價）則同起點方式換算。
    回傳 (報酬, 終點日) 或 (None, None)。"""
    prices, _, adj = book
    raw = prices.set_index("date")["close"]
    a = adj.set_index("date")["close"]
    i = raw.index.searchsorted(day)           # 決策日休市就取之後第一個交易日
    if i >= len(raw):
        return None, None
    d0 = raw.index[i]
    stop = a.index[-1] if end is None else a.index[a.index <= end][-1] if (a.index <= end).any() else None
    if stop is None or stop < d0:
        return None, None
    start_adj = price * a.loc[d0] / raw.loc[d0]
    end_adj = a.loc[stop] if end_price is None else end_price * a.loc[stop] / raw.loc[stop]
    return float(end_adj / start_adj - 1), stop


def _bench_return(book: tuple, day: pd.Timestamp, stop: pd.Timestamp):
    a = book[2].set_index("date")["close"]
    i = a.index.searchsorted(day)
    if i >= len(a) or a.index[i] > stop:
        return None
    return float(a.loc[stop] / a.iloc[i] - 1)


def review(cfg: dict, today: str | None = None, only: set | None = None) -> dict:
    """每筆決策對照實際結果。買進：到之後第一筆同代號賣出日（沒有就到最新資料日）；賣出：賣出後至今。
    only：只算這些 (recorded_at, stock_id)（到期提醒用，避免替每筆都抓資料）。"""
    dec = _read(decisions_path(cfg), DECISION_COLUMNS)
    if only is not None:
        dec = dec[[(a, b) in only for a, b in zip(dec["recorded_at"], dec["stock_id"])]]
    trades = _read(trades_path(cfg), ["date", "stock_id", "side"])
    today = pd.Timestamp(today or date.today().isoformat())
    cache: dict[str, tuple] = {}

    def book(sid: str, start: pd.Timestamp):
        if sid not in cache:
            cache[sid] = _load(sid, (start - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), cfg)
        return cache[sid]

    rows = []
    for r in dec.sort_values("date").itertuples(index=False):
        sid, day, price = r.stock_id, r.date, float(r.price)
        row = dict(recorded_at=r.recorded_at, date=day, side=r.side, stock_id=sid, price=price, thesis=r.thesis, exit_if=r.exit_if,
                   horizon_months=float(r.horizon_months) if str(r.horizon_months) not in ("", "nan") else None,
                   closed=False, end=None, ret=None, bench=None, excess=None, held_days=None, note="")
        start = (day - pd.Timedelta(days=10))
        try:
            b, bm = book(sid, start), book(BENCH, start)
            sells = trades[(trades.stock_id == sid) & (trades.side == "sell") & (trades.date > day)]
            end = sells["date"].min() if r.side == "buy" and len(sells) else None
            row["closed"] = end is not None
            end_price = None
            if end is not None:
                end_price = float(sells.loc[sells["date"] == end, "price"].iloc[0])
            ret, stop = _total_return(b, day, price, end, end_price)
            bench = _bench_return(bm, day, stop) if stop is not None else None
            row.update(ret=ret, bench=bench, end=stop)
            if ret is not None and bench is not None:
                row["excess"] = ret - bench
            if stop is not None:
                row["held_days"] = int((stop - day).days)
        except Exception as e:  # 單筆資料問題不要擋住整份報告，原因寫進報告
            row["note"] = f"資料不足：{type(e).__name__}"
        rows.append(row)
    df = pd.DataFrame(rows)
    buys = df[(df["side"] == "buy") & df["excess"].notna()] if len(df) else df
    summary = dict(n_buy=int((df["side"] == "buy").sum()) if len(df) else 0, n_scored=len(buys),
                   win_rate=float((buys["excess"] > 0).mean()) if len(buys) else None,
                   median_excess=float(buys["excess"].median()) if len(buys) else None)
    return dict(rows=df, summary=summary, today=today)


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.1%}"


def review_text(r: dict) -> str:
    s, df = r["summary"], r["rows"]
    L = [f"# 決策日誌回顧 {r['today']:%Y-%m-%d}", "",
         "> 只描述你自己的紀錄，不是投資建議。工具不判斷論點對錯，原文照列供你自己對照。", "",
         "## 彙總", "", f"- 買進決策 {s['n_buy']} 筆，其中算得出同期 0050 對照的 {s['n_scored']} 筆"]
    if s["n_scored"]:
        L.append(f"- 跑贏同期 0050（含息）的比例：{s['win_rate']:.0%}；超額報酬中位數 {_pct(s['median_excess'])}")
    if s["n_scored"] < MIN_SAMPLE:
        L.append(f"- 樣本太少，不足以下結論（少於 {MIN_SAMPLE} 筆）")
    L.append("- 報酬為含息總報酬（還原權息價；起點用實際成交價換算），0050 為同一區間含息總報酬，皆未扣成本")
    L.append("- 已賣出的買進決策終點用該筆賣出的實際成交價，未賣出的用最新收盤")
    buys = df[df["side"] == "buy"] if len(df) else df
    sells = df[df["side"] == "sell"] if len(df) else df
    L += ["", "## 買進決策", ""]
    for x in buys.itertuples(index=False):
        L += _entry(x, f"{x.date:%Y-%m-%d} 買進 {x.stock_id} @ {x.price:,.2f}", "持有至賣出" if x.closed else "持有至今")
    if not len(buys):
        L.append("（無）")
    L += ["", "## 賣出決策（賣出之後發生了什麼）", ""]
    for x in sells.itertuples(index=False):
        L += _entry(x, f"{x.date:%Y-%m-%d} 賣出 {x.stock_id} @ {x.price:,.2f}", "賣出後至今")
    if not len(sells):
        L.append("（無）")
    return "\n".join(L) + "\n"


def _entry(x, title: str, span: str) -> list[str]:
    hz = ""
    if x.horizon_months is not None:
        hz = f"；預計持有約 {x.horizon_months:g} 個月（約 {x.horizon_months * 30.4:.0f} 天）"
    held = f"{x.held_days} 天" if x.held_days is not None else "—"
    out = [f"### {title}", "",
           f"- 區間（{span}）至 {x.end:%Y-%m-%d}；實際經過 {held}{hz}" if x.end is not None else f"- 區間（{span}）：無資料",
           f"- 該股含息報酬 {_pct(x.ret)}；同期 0050 含息 {_pct(x.bench)}；差距 {_pct(x.excess)}"]
    if x.note:
        out.append(f"- {x.note}")
    out += [f"- 你當時寫的論點：{x.thesis or '（未填）'}", f"- 你當時寫的退出條件：{x.exit_if or '（未填）'}", ""]
    return out


def save_review(cfg: dict, today: str | None = None) -> tuple[Path, dict]:
    r = review(cfg, today)
    out = _dir(cfg, "reports_dir") / f"decisions_{r['today']:%Y%m%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(review_text(r), encoding="utf-8")
    return out, r


def reminders_path(cfg: dict) -> Path:
    return _dir(cfg, "data_dir") / "decision_reminders.json"


def _months_later(day: pd.Timestamp, months: float) -> pd.Timestamp:
    whole = int(months)
    return day + pd.DateOffset(months=whole) + pd.Timedelta(days=round((months - whole) * 30.4))


def due_reminders(cfg: dict, today: str | None = None, write_state: bool = True) -> str:
    """買進決策填了 horizon_months、今天已到「決策日 + 預計持有月數」、尚未提醒過、也還沒賣出的，產生提醒文字。
    沒有要提醒的回傳空字串。只描述事實並原文引用使用者寫的論點與退出條件，不評論、不判斷對錯。
    write_state=True 才把已提醒的 (recorded_at|stock_id) 追加到 data/decision_reminders.json。"""
    today_ts = pd.Timestamp(today or date.today().isoformat())
    dec = _read(decisions_path(cfg), DECISION_COLUMNS)
    if dec.empty:
        return ""
    rp = reminders_path(cfg)
    try:
        done = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    except (OSError, ValueError):
        done = {}
    trades = _read(trades_path(cfg), ["date", "stock_id", "side"])
    due = []
    for r in dec.itertuples(index=False):
        key = f"{r.recorded_at}|{r.stock_id}"
        if r.side != "buy" or str(r.horizon_months) in ("", "nan") or key in done:
            continue
        h = float(r.horizon_months)
        if today_ts < _months_later(r.date, h):
            continue
        if len(trades[(trades.stock_id == r.stock_id) & (trades.side == "sell") & (trades.date > r.date)]):
            continue   # 已經賣出的不提醒
        due.append((key, r, h))
    if not due:
        return ""
    rv = review(cfg, today_ts.strftime("%Y-%m-%d"), only={(r.recorded_at, r.stock_id) for _, r, _ in due})["rows"]
    res = {(x.recorded_at, x.stock_id): x for x in rv.itertuples(index=False)} if len(rv) else {}
    L = [f"【決策日誌到期提醒】{today_ts:%Y-%m-%d}", ""]
    for key, r, h in due:
        x = res.get((r.recorded_at, r.stock_id))
        held = (today_ts - r.date).days
        L += [f"{r.stock_id}：{r.date:%Y-%m-%d} 買進，預計持有 {h:g} 個月，已持有 {held} 天",
              f"  至今含息報酬 {_pct(x.ret) if x is not None else '—'}；同期 0050 含息 {_pct(x.bench) if x is not None else '—'}"
              + (f"（{x.note}）" if x is not None and x.note else ""),
              f"  你當時寫的論點：{r.thesis or '（未填）'}", f"  你當時寫的退出條件：{r.exit_if or '（未填）'}", ""]
    L.append("回頭看看當初的理由和失效條件，是否仍符合現在的情況")
    if write_state:
        rp.parent.mkdir(parents=True, exist_ok=True)
        done.update({k: today_ts.strftime("%Y-%m-%d") for k, _, _ in due})
        rp.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    return "\n".join(L)
