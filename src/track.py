"""前瞻模擬追蹤：把使用者登記的策略檔放進每日排程，用紙上交易持續記錄，紀錄寫入後不改。

只做紙上模擬與數字紀錄；不連券商、不產生下單指令、不給買賣建議。設計見 docs/前瞻追蹤設計.md。

檔案（都在 data/track/，只追加）：
- registry.csv        登記表：action(add/stop), name, spec_path, spec_hash, date, note
- <名稱>_signals.csv  訊號紀錄（選股結果凍結；之後的淨值只用這裡的選股，絕不事後重選）
- <名稱>.csv          每日淨值：策略、0050 含息、配對隨機組中位數、紙上持股
- prefetch_cursor.json 補資料的當日進度
"""
from __future__ import annotations

import csv
import json
import re
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import Costs
from .config import ROOT
from .fetch import FinMindError, fetch_events, fetch_prices
from .lab import SpecError, load_spec, read_ledger, spec_hash
from .rank_strategy import (eligible_pools, factor_table, load_universe, matched_random_targets, run_portfolio,
                            select_spec)

REG_COLS = ["action", "name", "spec_path", "spec_hash", "date", "note"]
SIG_COLS = ["signal_date", "nominal_date", "recorded_at", "kind", "stocks", "coverage", "n_eligible", "note",
            "trade_key"]       # trade_key = max(訊號日, 記錄當時的最新交易日)：淨值在它之後第一個交易日開盤成交
FUND_NOTE = "基本面為記錄當下的 FinMind 版本"
NAV_COLS = ["date", "nav_strategy", "nav_0050", "nav_random_median", "holdings", "recorded_at", "restated_diff"]
RESTATE_WARN = 0.005          # 重算後和當初記錄的淨值差超過這個比例就在摘要標註
DATA_START = "2016-01-01"      # 和 bulk / radar 相同，讓快取共用
MIN_DAYS = 252                 # 未滿這麼多個交易日不下結論、不顯示 Sharpe
MIN_BENCH_DRAWDOWN = 0.10      # 期間 0050 至少要從高點回檔這麼多，才算經歷過一次大盤回檔
GAP_DAYS = 5                   # 持股資料中斷超過這麼多個交易日就在摘要提示
NAME_RE = re.compile(r"[A-Za-z0-9_]+")


# ---------------- 路徑與設定 ----------------
def track_dir(cfg: dict) -> Path:
    return ROOT / cfg["data_dir"] / "track"


def _tc(cfg: dict) -> dict:
    d = dict(min_coverage=0.95, max_defer_days=5, prefetch_days=5, prefetch_minutes=60, random_groups=5,
             seed=20261126)
    d.update(cfg.get("track") or {})
    return d


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ts(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def _append(path: Path, cols: list[str], rows: list[dict]) -> None:
    """只追加；檔案不存在才寫表頭。"""
    if not rows:
        return
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not new:
        with open(path, encoding="utf-8-sig", newline="") as f:
            head = next(csv.reader(f), [])
        if head != cols:
            raise ValueError(f"{path.name} 的表頭和預期欄位不一致（檔案 {head}，預期 {cols}）；"
                             "不會寫入錯位的列，請先處理這個檔案")
    with open(path, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerows({k: r.get(k, "") for k in cols} for r in rows)


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")


# ---------------- 登記表 ----------------
def read_registry(cfg: dict) -> pd.DataFrame:
    return _read(track_dir(cfg) / "registry.csv", REG_COLS)


def active(cfg: dict) -> dict[str, dict]:
    """追蹤中的策略：{名稱: 最後一列 add}（之後有 stop 的不算）。"""
    out: dict[str, dict] = {}
    for r in read_registry(cfg).to_dict("records"):
        if r["action"] == "add":
            out[r["name"]] = r
        elif r["action"] == "stop":
            out.pop(r["name"], None)
    return out


def add(cfg: dict, spec_path, today=None, note: str = "") -> dict:
    """開始追蹤：記下規則雜湊與開始日（今天）。同名（含已停止的）一律拒絕，避免覆蓋舊紀錄。"""
    spec = load_spec(spec_path)        # SpecError 往外丟
    name = spec["name"]
    if not NAME_RE.fullmatch(str(name)):
        raise SpecError(f"策略名稱只能用英數與底線，收到 {name!r}")
    names = set(read_registry(cfg)["name"])
    if name in active(cfg):
        raise ValueError(f"{name} 已在追蹤中")
    if name in names:
        raise ValueError(f"{name} 曾經追蹤過（已停止）；紀錄不能覆蓋，請另存新檔、用新名稱")
    day = _ts(today or date.today())
    p = Path(spec_path)
    try:
        shown = p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(p)
    row = dict(action="add", name=name, spec_path=shown, spec_hash=spec_hash(spec, cfg),
               date=f"{day:%Y-%m-%d}", note=note)
    _append(track_dir(cfg) / "registry.csv", REG_COLS, [row])
    return row


def stop(cfg: dict, name: str, today=None, note: str = "手動停止") -> dict:
    cur = active(cfg)
    if name not in cur:
        raise ValueError(f"{name} 不在追蹤中")
    day = _ts(today or date.today())
    row = dict(action="stop", name=name, spec_path=cur[name]["spec_path"], spec_hash=cur[name]["spec_hash"],
               date=f"{day:%Y-%m-%d}", note=note)
    _append(track_dir(cfg) / "registry.csv", REG_COLS, [row])
    return row


# ---------------- 資料存取（測試時 monkeypatch 這幾個） ----------------
def _calendar(cfg: dict, start, end) -> pd.DatetimeIndex:
    """交易日（以基準 0050 有成交的日子為準）。"""
    px = fetch_prices(cfg["rank"]["benchmark"], f"{_ts(start):%Y-%m-%d}", f"{_ts(end):%Y-%m-%d}", cfg)
    return pd.DatetimeIndex(sorted(pd.to_datetime(px["date"])))


def _is_trading_day(cfg: dict, day) -> bool:
    """day 是不是交易日（交易日曆；查不到時以平日代替）。測試時 monkeypatch。"""
    from .daily import is_trading_day
    try:
        r = is_trading_day(_ts(day).date(), cfg)
    except Exception:
        r = None
    return _ts(day).weekday() < 5 if r is None else bool(r)


def _load(ids, start, end, cfg, ends=None, fundamentals=True):
    return load_universe(list(ids), start, end, cfg, fundamentals=fundamentals, ends=ends)


def _price_dates(cfg: dict, sid: str) -> pd.DatetimeIndex:
    """只讀本機快取：這檔股票有價格的日期（不連網）。"""
    from .fetch import _cache_path
    p = _cache_path(sid, cfg["data_dir"], "TaiwanStockPrice")
    if not p.exists():
        return pd.DatetimeIndex([])
    df = pd.read_csv(p, usecols=["date", "open", "close"], parse_dates=["date"])
    return pd.DatetimeIndex(df.loc[(df["open"] > 0) & (df["close"] > 0), "date"])


def _fetch_stock(sid: str, end: str, cfg: dict) -> None:
    """補一檔股票到 end：日K、事件、基本面（都走快取，只補缺的）。"""
    from .fundamentals import eps_pit, revenue_pit
    px = fetch_prices(sid, DATA_START, end, cfg)
    if not px.empty:
        fetch_events(sid, DATA_START, end, cfg, prices=px)
    revenue_pit(sid, cfg)
    eps_pit(sid, cfg)


class _Ctx:
    """一次 update 內的暫存：快取讀過的價格日期。"""
    def __init__(self):
        self.dates: dict[str, pd.DatetimeIndex] = {}
        self.prefetched = False

    def get(self, cfg, sid):
        if sid not in self.dates:
            self.dates[sid] = _price_dates(cfg, sid)
        return self.dates[sid]


# ---------------- 候選與涵蓋率 ----------------
def candidates(cfg: dict, spec: dict, day) -> tuple[list[str], dict | None]:
    """訊號日的候選股票。full：full_candidates.csv 中 end ≥ 訊號日 − 10 天者；
    名單若過舊（所有 end 都早於訊號日 − 10 天）就改用名單裡最新 end 往前 10 天，避免整份名單被排除。"""
    day = _ts(day)
    if spec["universe"] != "full":
        return [str(s) for s in cfg["evaluation"]["universe"]], None
    p = ROOT / cfg["data_dir"] / "full_candidates.csv"
    if not p.exists():
        return [], None
    c = pd.read_csv(p, dtype=str)
    if c.empty:
        return [], None
    e = pd.to_datetime(c["end"])
    cut = e.max() - pd.Timedelta(days=10)
    c = c[e >= min(day, e.max()) - pd.Timedelta(days=10)]
    # ends 只放已下市的（end 明顯早於名單建立日）；上市中的 end 只是名單建立時的最後資料日，放進去會把之後的資料截掉
    return list(c["stock_id"]), {s: x for s, x in zip(c["stock_id"], c["end"]) if pd.Timestamp(x) < cut}


def delisted_ids(cfg: dict) -> set:
    """官方已下市的股票（full_candidates.csv 中 end 明顯早於名單內最大 end）；只有這些才在淨值模擬中出清。"""
    p = ROOT / cfg["data_dir"] / "full_candidates.csv"
    if not p.exists():
        return set()
    c = pd.read_csv(p, dtype=str)
    if c.empty:
        return set()
    e = pd.to_datetime(c["end"])
    return set(c.loc[e < e.max() - pd.Timedelta(days=10), "stock_id"])


def coverage(cfg: dict, ids: list[str], day, ctx: _Ctx) -> float:
    if not ids:
        return 0.0
    day = _ts(day)
    return sum(day in ctx.get(cfg, s) for s in ids) / len(ids)


# ---------------- 補資料 ----------------
def prefetch(cfg: dict, ids: list[str], today, latest, clock=time.monotonic) -> dict:
    """依序對候選股補資料到 today；快取最後日期已到 latest（最新交易日）的略過。
    當日已試過的記在游標檔，下次接著補；遇到 FinMind 錯誤或超過時間預算就停。"""
    tc = _tc(cfg)
    path = track_dir(cfg) / "prefetch_cursor.json"
    day_s = f"{_ts(today):%Y-%m-%d}"
    state = {"day": day_s, "attempted": []}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("day") == day_s:
                state = old
        except (ValueError, OSError):
            pass
    tried = set(state["attempted"])
    deadline = clock() + tc["prefetch_minutes"] * 60
    ctx_dates: dict[str, pd.DatetimeIndex] = {}
    fetched, stopped = 0, None
    for sid in ids:
        if sid in tried:
            continue
        d = ctx_dates.setdefault(sid, _price_dates(cfg, sid))
        if len(d) and d.max() >= _ts(latest):
            continue
        if clock() > deadline:
            stopped = "time"
            break
        try:
            _fetch_stock(sid, day_s, cfg)
        except FinMindError as e:
            stopped = str(e) or type(e).__name__
            break
        tried.add(sid)
        state["attempted"].append(sid)
        fetched += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    out = {"fetched": fetched}
    if stopped:
        out["stopped"] = stopped
    return out


# ---------------- 訊號 ----------------
def _signals(cfg: dict, name: str) -> pd.DataFrame:
    return _read(track_dir(cfg) / f"{name}_signals.csv", SIG_COLS)


def _stocks(s: str) -> list[str]:
    return [x for x in s.split(";") if x]


def compute_signal(cfg: dict, spec: dict, day, ids: list[str], ends: dict | None) -> tuple[list[str], list[str]]:
    """第 day 天收盤後的選股與當天 eligible pool；資料只載到 day（再由 factor_table 只取 ≤ day）。"""
    day = _ts(day)
    rc = cfg["rank"]
    mt = rc["full_min_turnover"] if spec["universe"] == "full" else rc["min_turnover"]
    U = _load(ids, DATA_START, f"{day:%Y-%m-%d}", cfg, ends=ends)
    picks = select_spec(factor_table(U, day, spec["lookback"]), spec, min_price=rc["min_price"], min_turnover=mt)
    pool = eligible_pools(U, [day], spec["lookback"], min_price=rc["min_price"], min_turnover=mt)[day]
    return picks, pool


def _random_pick(prev_day, day, prev_s: list[str], prev_r: list[str], picks: list[str], pool: list[str], rng) -> list[str]:
    """沿用 matched_random_targets：同檔數、和上一期的重疊檔數與策略相同。
    key 用兩個 Timestamp（上一期訊號日、本期訊號日），排序才會先處理上一期。"""
    t, p = {}, {}
    if (prev_s or prev_r) and prev_day is not None and pd.Timestamp(prev_day) < pd.Timestamp(day):
        t[pd.Timestamp(prev_day)], p[pd.Timestamp(prev_day)] = prev_s, prev_r
    t[pd.Timestamp(day)], p[pd.Timestamp(day)] = picks, pool
    return matched_random_targets(t, p, rng)[pd.Timestamp(day)]


def _next_pos(sigs: pd.DataFrame, cal: pd.DatetimeIndex, R: int) -> int:
    """下一個名目訊號日在 cal 的位置：第一期 = 開始日；之後 = 上一個已記錄名目訊號日往後數 R 個交易日。
    （不從開始日重數：交易日曆事後補進缺的日子時，格線才不會位移、產生重複訊號。）"""
    nom = sigs["nominal_date"]
    nom = nom[nom != ""]
    return 0 if nom.empty else int(cal.searchsorted(pd.Timestamp(nom.max()))) + R


def process_signals(cfg: dict, name: str, spec: dict, start, cal: pd.DatetimeIndex, ctx: _Ctx, today=None) -> int:
    """依排程補記該記而未記的訊號；回傳新寫入的列數。cal = 開始日以後（到今天）的交易日。
    選股只用 ≤ 訊號日的資料；但若記錄時已晚於訊號日，淨值成交要等記錄日之後（trade_key），不能回頭用過去的開盤價。"""
    tc = _tc(cfg)
    R = int(spec["rebalance_days"])
    path = track_dir(cfg) / f"{name}_signals.csv"
    written = 0
    today = _ts(today if today is not None else cal[-1])
    # 記錄日當天若是交易日，淨值最早在記錄日「之後」的交易日開盤成交（即使 0050 快取還沒有今天的資料）
    latest = today if _is_trading_day(cfg, today) else cal[-1]
    while True:
        sigs = _signals(cfg, name)
        pos0 = _next_pos(sigs, cal, R)
        if pos0 > len(cal) - 1:
            break
        nominal = f"{cal[pos0]:%Y-%m-%d}"
        chosen, waiting, last_cov = None, False, 0.0
        for j in range(tc["max_defer_days"] + 1):
            pos = pos0 + j
            if pos > len(cal) - 1:
                waiting = True
                break
            ids, ends = candidates(cfg, spec, cal[pos])
            cov = coverage(cfg, ids, cal[pos], ctx)
            last_cov = cov
            if ids and cov >= tc["min_coverage"]:
                chosen = (cal[pos], ids, ends, cov)
                break
        if waiting and chosen is None:
            break                       # 資料還沒齊，明天再看
        rec = _now()
        if chosen is None:
            _append(path, SIG_COLS, [dict(signal_date=nominal, nominal_date=nominal, recorded_at=rec, kind="skipped",
                                          stocks="", coverage=f"{last_cov:.4f}", n_eligible="",
                                          note=f"資料涵蓋率不足（最後一次 {last_cov:.1%} < {tc['min_coverage']:.0%}），"
                                               f"順延 {tc['max_defer_days']} 個交易日仍不足，這一期不記訊號")])
            written += 1
            continue
        day, ids, ends, cov = chosen
        picks, pool = compute_signal(cfg, spec, day, ids, ends)
        ordinal = int((sigs["kind"] == "strategy").sum())
        notes = [FUND_NOTE] + ([f"順延自 {nominal}"] if day != cal[pos0] else [])
        base = dict(signal_date=f"{day:%Y-%m-%d}", nominal_date=nominal, recorded_at=rec,
                    coverage=f"{cov:.4f}", n_eligible=str(len(pool)), note="；".join(notes),
                    trade_key=f"{max(day, latest):%Y-%m-%d}")
        rows = [dict(base, kind="strategy", stocks=";".join(picks))]
        prev = sigs[sigs["kind"] == "strategy"]
        prev_s = _stocks(prev.iloc[-1]["stocks"]) if len(prev) else []
        prev_day = prev.iloc[-1]["signal_date"] if len(prev) else None
        for g in range(1, int(tc["random_groups"]) + 1):
            pr = sigs[sigs["kind"] == f"random_{g}"]
            prev_r = _stocks(pr.iloc[-1]["stocks"]) if len(pr) else []
            rng = np.random.default_rng([int(tc["seed"]), ordinal, g])
            rows.append(dict(base, kind=f"random_{g}",
                             stocks=";".join(_random_pick(prev_day, day, prev_s, prev_r, picks, pool, rng))))
        _append(path, SIG_COLS, rows)
        written += len(rows)
    return written


# ---------------- 淨值 ----------------
def _targets(sigs: pd.DataFrame, kind: str) -> dict:
    d = sigs[sigs["kind"] == kind]
    return {pd.Timestamp(r.get("trade_key") or r["signal_date"]): _stocks(r["stocks"]) for r in d.to_dict("records")}


def _holdings(trades: pd.DataFrame, day) -> str:
    t = trades[trades["date"] <= day]
    if t.empty:
        return ""
    net = (np.where(t["side"] == "buy", 1.0, -1.0) * t["shares"].to_numpy())
    s = pd.Series(net, index=t["stock_id"].to_numpy()).groupby(level=0).sum()
    return ";".join(sorted(s[s > 0].index))


def update_nav(cfg: dict, name: str, today) -> int:
    """用凍結的訊號從第一個訊號日跑到今天，只追加還沒記過的日期。回傳新寫入列數。"""
    sigs = _signals(cfg, name)
    strat = _targets(sigs, "strategy")
    if not strat:
        return 0
    bench = cfg["rank"]["benchmark"]
    rand_kinds = sorted(set(k for k in sigs["kind"] if k.startswith("random_")))
    rand = {k: _targets(sigs, k) for k in rand_kinds}
    ids = sorted({s for t in [strat, *rand.values()] for v in t.values() for s in v} - {bench})
    first = min(strat)
    U = _load(ids + [bench], f"{first - pd.Timedelta(days=60):%Y-%m-%d}", f"{_ts(today):%Y-%m-%d}", cfg,
              fundamentals=False)
    bt, costs = cfg["backtest"], Costs.from_config(cfg)
    kw = dict(max_participation=bt.get("max_volume_participation"), volume_window=bt.get("volume_window", 20))
    start, end = f"{first:%Y-%m-%d}", f"{_ts(today):%Y-%m-%d}"
    cash0 = bt["initial_cash"]
    def snap(t):    # trade_key 不在日曆上（當天資料還沒進來）就對到之後第一個有資料的日子：只會更晚成交，不會更早
        out = {}
        for k, v in sorted(t.items()):
            i = U.calendar.searchsorted(k)
            if i < len(U.calendar):
                out[U.calendar[i]] = v
        return out
    strat, rand = snap(strat), {k: snap(t) for k, t in rand.items()}
    if not strat:
        return 0
    first = min(strat)
    kw["delisted"] = delisted_ids(cfg)      # 只有官方下市者才出清；資料中斷的持股沿用最後收盤
    main = run_portfolio(U, strat, costs, cash0, start, end, **kw)
    b = run_portfolio(U, {first: [bench]}, costs, cash0, start, end, **kw)
    rnd = [run_portfolio(U, t, costs, cash0, start, end, **kw).equity / cash0 for t in rand.values()]
    med = pd.concat(rnd, axis=1).median(axis=1) if rnd else pd.Series(np.nan, index=main.equity.index)
    path = track_dir(cfg) / f"{name}.csv"
    old = _read(path, NAV_COLS)
    have = set(old["date"])
    restated = ""      # 今天重算出的「上一個已記錄日」淨值相對當初記錄的差
    if len(old) and pd.Timestamp(old.iloc[-1]["date"]) in main.equity.index:
        was = float(old.iloc[-1]["nav_strategy"])
        restated = f"{main.equity[pd.Timestamp(old.iloc[-1]['date'])] / cash0 / was - 1:.6f}" if was else ""
    rec, rows = _now(), []
    for d in main.equity.index:
        if f"{d:%Y-%m-%d}" in have:
            continue
        rows.append(dict(date=f"{d:%Y-%m-%d}", nav_strategy=f"{main.equity[d] / cash0:.6f}",
                         nav_0050=f"{b.equity.get(d, np.nan) / cash0:.6f}",
                         nav_random_median=f"{med.get(d, np.nan):.6f}", holdings=_holdings(main.trades, d),
                         recorded_at=rec, restated_diff=restated))
    _append(path, NAV_COLS, rows)
    return len(rows)


# ---------------- 每日更新 ----------------
def _update_one(cfg: dict, name: str, row: dict, today, ctx: _Ctx, res: dict) -> None:
    tc = _tc(cfg)
    try:
        p = Path(row["spec_path"])
        spec = load_spec(p if p.is_absolute() else ROOT / p)
        changed = spec_hash(spec, cfg) != row["spec_hash"]
    except (OSError, SpecError):
        changed = True
        spec = None
    if changed:
        stop(cfg, name, today, note="策略檔被修改（或讀不到）：規則雜湊和登記時不同，自動停止；要改規則請另開新追蹤")
        res["stopped"].append(name)
        return
    start = _ts(row["date"])
    cal = _calendar(cfg, start, today)
    cal = cal[cal >= start]
    if len(cal) == 0:
        return
    R = int(spec["rebalance_days"])
    if _next_pos(_signals(cfg, name), cal, R) - (len(cal) - 1) <= tc["prefetch_days"] and not ctx.prefetched:
        ctx.prefetched = True
        ids, _ = candidates(cfg, spec, cal[-1])
        res["prefetch"] = prefetch(cfg, ids, today, cal[-1])
        ctx.dates.clear()
    res["signals"] += process_signals(cfg, name, spec, start, cal, ctx, today)
    res["nav"] += update_nav(cfg, name, today)
    res["updated"].append(name)


def update(cfg: dict, today=None) -> dict:
    """每日更新所有追蹤中的策略。單一策略出錯只記在 errors，不影響其他策略。"""
    today = _ts(today or date.today())
    res = dict(updated=[], stopped=[], signals=0, nav=0, errors=[])
    ctx = _Ctx()
    for name, row in active(cfg).items():
        try:
            _update_one(cfg, name, row, today, ctx, res)
        except Exception as e:   # 追蹤失敗不影響其他步驟；下次排程會接著補
            res["errors"].append(f"{name}：{type(e).__name__}")
    return res


# ---------------- 摘要 ----------------
def lab_verdict(cfg: dict, h: str) -> str:
    led = read_ledger(ROOT / cfg["reports_dir"] / "lab" / "trials.csv")
    led = led[led["spec_hash"] == h]
    return led.iloc[-1]["verdict"] if len(led) else "未經實驗室"


def _nav(cfg: dict, name: str) -> pd.DataFrame:
    df = _read(track_dir(cfg) / f"{name}.csv", NAV_COLS)
    if "restated_diff" not in df:
        df["restated_diff"] = ""
    for c in ("nav_strategy", "nav_0050", "nav_random_median"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _mdd(s: pd.Series) -> float:
    return float((s / s.cummax() - 1).min())


def _gap_stocks(cfg: dict, held: list[str], nav_dates) -> list[str]:
    """持股中最後一筆價格比淨值最後日期早超過 GAP_DAYS 個交易日（以淨值紀錄的日期數計）的股票。"""
    dates = pd.to_datetime(pd.Series(list(nav_dates)))
    out = []
    for sid in held:
        d = _price_dates(cfg, sid)
        last = d.max() if len(d) else pd.Timestamp.min
        if int((dates > last).sum()) > GAP_DAYS:
            out.append(sid)
    return out


def _block(cfg: dict, name: str, row: dict) -> list[str]:
    head = f"◆ {name}（{row['date']} 開始；實驗室判定：{lab_verdict(cfg, row['spec_hash'])}）"
    nav = _nav(cfg, name)
    if nav.empty:
        return [head, "  尚無淨值紀錄（等第一個訊號日）"]
    n = len(nav)
    cum = lambda s: float(s.iloc[-1] / s.iloc[0] - 1)    # noqa: E731
    s, b, r = nav["nav_strategy"], nav["nav_0050"], nav["nav_random_median"]
    L = [head, f"  追蹤 {n} 個交易日｜累積報酬 {cum(s):+.1%}｜最大回撤 {_mdd(s):.1%}",
         f"  同期 0050 含息 {cum(b):+.1%}（策略差 {(cum(s) - cum(b)) * 100:+.1f} 個百分點）；"
         f"配對隨機中位數 {cum(r):+.1%}（策略差 {(cum(s) - cum(r)) * 100:+.1f} 個百分點）"]
    tdays = cfg["backtest"]["trading_days_per_year"]
    if n >= MIN_DAYS:
        d = s.pct_change().dropna()
        sh = float(d.mean() / d.std() * np.sqrt(tdays)) if d.std() > 0 else 0.0
        L.append(f"  Sharpe {sh:.2f}")
    hold = _stocks(nav.iloc[-1]["holdings"])
    L.append("  目前紙上持股：" + ("、".join(hold) if hold else "無"))
    gap = _gap_stocks(cfg, hold, nav["date"])
    if gap:
        L.append(f"  ⚠ 持股中有資料中斷超過 {GAP_DAYS} 個交易日的股票（可能已下市，請重跑 bulk-fetch 更新下市名單）："
                 + "、".join(gap))
    rd = pd.to_numeric(nav["restated_diff"], errors="coerce").dropna()
    if len(rd) and rd.abs().max() > RESTATE_WARN:
        L.append(f"  ⚠ 淨值曾因資料更正而重算，與當時紀錄差 {rd.loc[rd.abs().idxmax()]:+.1%}")
    why = []
    if n < MIN_DAYS:
        why.append(f"追蹤未滿 {MIN_DAYS} 個交易日（目前 {n}）")
    if _mdd(b) > -MIN_BENCH_DRAWDOWN:
        why.append(f"期間 0050 未曾從高點回檔 {MIN_BENCH_DRAWDOWN:.0%} 以上（最深 {_mdd(b):.1%}）")
    if why:
        L.append("  ⚠ 樣本不足，不能下結論：" + "；".join(why))
    return L


def summary_text(cfg: dict, today=None) -> str:
    """各追蹤中策略的淨值摘要；只描述數字。沒有追蹤中的策略（也沒有今天自動停止的）回傳空字串。"""
    today = _ts(today or date.today())
    act = active(cfg)
    reg = read_registry(cfg)
    stopped = reg[(reg["action"] == "stop") & (reg["date"] == f"{today:%Y-%m-%d}") & reg["note"].str.startswith("策略檔")]
    if not act and stopped.empty:
        return ""
    L = [f"【前瞻追蹤】{today:%Y-%m-%d}（紙上模擬的數字紀錄）"]
    for r in stopped.to_dict("records"):
        L.append(f"⚠ {r['name']} 已自動停止：策略檔被修改，紀錄保留")
    for name, row in act.items():
        L += _block(cfg, name, row)
    L.append("※ 僅為紙上模擬的歷史數字，不含任何下單功能。")
    return "\n".join(L)


def list_text(cfg: dict, today=None) -> str:
    reg = read_registry(cfg)
    if reg.empty:
        return "尚未登記任何追蹤（python -m src.cli track add strategies/lab/<名稱>.yaml）"
    act = active(cfg)
    L = ["名稱 | 狀態 | 開始日 | 實驗室判定"]
    for name in dict.fromkeys(reg["name"]):
        first = reg[(reg["name"] == name) & (reg["action"] == "add")].iloc[0]
        st = "追蹤中" if name in act else "已停止（" + reg[(reg["name"] == name) & (reg["action"] == "stop")].iloc[-1]["note"] + "）"
        L.append(f"{name} | {st} | {first['date']} | {lab_verdict(cfg, first['spec_hash'])}")
    s = summary_text(cfg, today)
    return "\n".join(L) + ("\n\n" + s if s else "")
