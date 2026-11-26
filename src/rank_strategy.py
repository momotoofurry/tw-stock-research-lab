"""選股排名策略（多檔投資組合）＋ 多檔回測引擎。

策略（參考 FinLab「動能 + 品質」排名的做法，改用 FinMind 免費資料）：
每 rebalance_days 個交易日，在第 t 天收盤後，從股票池中篩選
  近 63 日漲幅 > 0、近 lookback 日漲幅 > 0、近 3 個月營收年增 > 0、近四季 EPS > 0、股價 > min_price、
  近 20 日平均成交金額 > min_turnover
再依近 lookback 日漲幅排名，取前 top_n 檔等權重，第 t+1 天開盤調整部位。
所有基本面只用 t 收盤前已公布的資料（src/fundamentals.py）。

引擎規則同單檔回測：t+1 開盤成交、手續費（最低 20 元）、賣出證交稅、開盤漲停買不到 / 跌停賣不掉（之後每天重試，
直到下一次調整）、股利入帳。以零股（1 股）為單位，才能把資金平均分配到多檔高價股。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import BacktestResult, Costs, limit_prices
from .fetch import fetch_events, fetch_prices
from .fundamentals import as_of, eps_pit, revenue_pit


@dataclass
class StockData:
    prices: pd.DataFrame      # 未還原，index = date
    adj_close: pd.Series      # 向前還原收盤（只用來算報酬）
    dividends: pd.Series      # 每股現金股利價值（before − after），index = 除權息日
    share_mult: pd.Series     # 公司行動（減資 / 分割）股數乘數 before / after，index = 生效日
    revenue: pd.DataFrame
    eps: pd.DataFrame


@dataclass
class Universe:
    stocks: dict[str, StockData]
    calendar: pd.DatetimeIndex
    names: dict = field(default_factory=dict)


def load_universe(stock_ids: list[str], start: str, end: str, cfg: dict, fundamentals: bool = True,
                  ends: dict | None = None) -> Universe:
    """ends：個股的資料截止日（已下市股票用下市日，避免每次都去查下市後不存在的資料）。"""
    from .adjust import adjust_for_dividends
    stocks = {}
    for sid in stock_ids:
        e = min(end, ends[sid]) if ends and sid in ends else end
        px = fetch_prices(sid, start, e, cfg)
        if px.empty:
            continue
        dv = fetch_events(sid, start, e, cfg, prices=px)
        adj = adjust_for_dividends(px, dv)
        cashd = dv[dv["kind"] == "dividend"] if len(dv) else dv
        corp = dv[dv["kind"] == "corporate"] if len(dv) else dv
        ps = (cashd.assign(ps=cashd["before_price"] - cashd["after_price"]).groupby("date")["ps"].sum()
              if len(cashd) else pd.Series(dtype=float))
        mult = ((corp["before_price"] / corp["after_price"]).groupby(corp["date"]).prod()
                if len(corp) else pd.Series(dtype=float))
        stocks[sid] = StockData(
            prices=px.set_index("date"), adj_close=adj.set_index("date")["close"], dividends=ps, share_mult=mult,
            revenue=revenue_pit(sid, cfg) if fundamentals else pd.DataFrame(),
            eps=eps_pit(sid, cfg) if fundamentals else pd.DataFrame())
    cal = pd.DatetimeIndex(sorted(set().union(*[set(s.prices.index) for s in stocks.values()])))
    return Universe(stocks, cal)


# ---------------- 因子與選股 ----------------
def factor_table(u: Universe, day: pd.Timestamp, lookback: int = 126) -> pd.DataFrame:
    """第 day 天收盤後可算出的因子。只用 day（含）以前的價格、day 以前已公布的基本面。"""
    rows = []
    for sid, s in u.stocks.items():
        ac = s.adj_close[s.adj_close.index <= day]
        px = s.prices[s.prices.index <= day]
        if len(ac) <= max(lookback, 63) or ac.index[-1] != day:   # 當天沒交易（停牌）就不列入
            continue
        turnover = float((px["close"] * px["volume"]).iloc[-20:].mean())
        eps_ttm = as_of(s.eps, day, "eps_ttm") if len(s.eps) else np.nan
        close = float(px["close"].iloc[-1])
        dv = (s.dividends[(s.dividends.index > day - pd.Timedelta(days=365)) & (s.dividends.index <= day)]
              if len(s.dividends) else s.dividends)
        m = 1.0   # 財報季末之後（到 day 為止）的股數乘數；eps_ttm 是用當時股數算的
        if len(s.share_mult) and len(s.eps) and "quarter_end" in s.eps:
            avail = s.eps[s.eps["available"] <= day]
            if len(avail):
                q = avail["quarter_end"].iloc[-1]
                m = float(s.share_mult[(s.share_mult.index > q) & (s.share_mult.index <= day)].prod())
        rows.append(dict(stock_id=sid, close=close,
                         pe=close * m / eps_ttm if eps_ttm > 0 and close > 0 else np.nan,   # NaN 或 ≤ 0 → NaN
                         dy=float(sum(v / (s.share_mult[(s.share_mult.index >= d) & (s.share_mult.index <= day)].prod()
                                           if len(s.share_mult) else 1.0) for d, v in dv.items())) / close if close > 0 else np.nan,
                         mom63=float(ac.iloc[-1] / ac.iloc[-64] - 1),
                         mom=float(ac.iloc[-1] / ac.iloc[-1 - lookback] - 1),
                         yoy3=as_of(s.revenue, day, "yoy3") if len(s.revenue) else np.nan,
                         eps_ttm=eps_ttm,
                         turnover=turnover))
    return pd.DataFrame(rows)


def select(f: pd.DataFrame, top_n: int = 8, min_price: float = 10, min_turnover: float = 2e7,
           use_fundamentals: bool = True) -> list[str]:
    if f.empty:
        return []
    ok = (f["mom63"] > 0) & (f["mom"] > 0) & (f["close"] > min_price) & (f["turnover"] > min_turnover)
    if use_fundamentals:
        ok &= (f["yoy3"] > 0) & (f["eps_ttm"] > 0)
    return list(f[ok].sort_values("mom", ascending=False)["stock_id"].head(top_n))


_OPS = {">": "gt", ">=": "ge", "<": "lt", "<=": "le"}


def select_spec(f: pd.DataFrame, spec: dict, min_price: float = 10, min_turnover: float = 2e7) -> list[str]:
    """策略檔版選股（策略實驗室用）：價格 / 流動性門檻，再套 spec["filters"]（全部 AND），
    依 spec["rank_by"] 排名取前 spec["top_n"] 檔。因子缺值（NaN）的比較一律不成立。"""
    if f.empty:
        return []
    ok = (f["close"] > min_price) & (f["turnover"] > min_turnover)
    for flt in spec["filters"]:
        ok &= getattr(f[flt["factor"]], _OPS[flt["op"]])(flt["value"])
    rb = spec["rank_by"]
    g = f[ok].dropna(subset=[rb["factor"]])
    return list(g.sort_values(rb["factor"], ascending=rb["order"] == "asc")["stock_id"].head(spec["top_n"]))


def rebalance_days(u: Universe, start: str, end: str, every: int) -> list[pd.Timestamp]:
    cal = u.calendar[(u.calendar >= pd.Timestamp(start)) & (u.calendar <= pd.Timestamp(end))]
    return list(cal[::every])


def strategy_targets(u: Universe, start: str, end: str, lookback: int = 126, top_n: int = 8,
                     every: int = 63, use_fundamentals: bool = True, **kw) -> dict:
    return {d: select(factor_table(u, d, lookback), top_n, use_fundamentals=use_fundamentals, **kw)
            for d in rebalance_days(u, start, end, every)}


def eligible_pools(u: Universe, days, lookback: int = 126, min_price: float = 10,
                   min_turnover: float = 2e7) -> dict:
    """每個調整日「當天」可投資的股票：有足夠歷史、當天有交易、股價與 20 日均成交金額達標（只用當天以前的資料）。
    隨機選股與等權對照組都從這裡抽，才和策略面對同一個股票池。"""
    pools = {}
    for d in days:
        f = factor_table(u, d, lookback)
        pools[d] = ([] if f.empty else
                    list(f.loc[(f["close"] > min_price) & (f["turnover"] > min_turnover), "stock_id"]))
    return pools


def matched_random_targets(targets: dict, pools: dict, rng) -> dict:
    """和策略配對的隨機選股：每次調整挑的檔數 = 策略當次實際檔數；續抱的檔數 = 策略當次和上次重疊的檔數
    （從隨機組上次持股中仍可投資的股票抽），其餘從當天股票池抽。這樣持股數與換手率都和策略相同。"""
    out, prev_s, prev_r = {}, [], []
    for d in sorted(targets):
        picks, pool = list(targets[d]), pools.get(d, [])
        n_keep = len(set(picks) & set(prev_s))
        keep_cands = [s for s in prev_r if s in set(pool)]
        keep = ([str(x) for x in rng.choice(keep_cands, size=min(n_keep, len(keep_cands)), replace=False)]
                if n_keep and keep_cands else [])
        rest = [s for s in pool if s not in set(keep)]
        k = min(len(picks) - len(keep), len(rest))
        new = [str(x) for x in rng.choice(rest, size=k, replace=False)] if k > 0 else []
        out[d] = keep + new
        prev_s, prev_r = picks, out[d]
    return out


# ---------------- 多檔引擎 ----------------
def run_portfolio(u: Universe, targets: dict, costs: Costs, initial_cash: float = 5_000_000,
                  start: str | None = None, end: str | None = None, cash_buffer: float = 0.01,
                  max_participation: float | None = None, volume_window: int = 20,
                  delist_recovery: float = 1.0, delisted: set | None = None) -> BacktestResult:
    """targets：{訊號日: [股票...]}（等權重）。訊號日收盤後決定，下一個交易日開盤開始調整。
    max_participation：每檔每天買進股數 ≤ 成交日「之前」volume_window 日均量 × 比例（None = 不限制）；
    沒買足的部分之後每天繼續買，直到下一次調整。
    delist_recovery：資料提早結束（下市）時仍持有的部位，以最後收盤 × 這個比例出清。1 = 最後收盤（偏樂觀，
    合併下市大致如此）；0 = 全部歸零（破產下市的最壞情況）。資料無法分辨下市原因，研究兩種都報。
    delisted：只有這個集合內的股票才會在資料結束時出清（None = 所有股票，舊行為）；其餘股票資料中斷時沿用最後收盤。"""
    cal = u.calendar
    if start is not None:
        cal = cal[cal >= pd.Timestamp(start)]
    if end is not None:
        cal = cal[cal <= pd.Timestamp(end)]
    sids = list(u.stocks)
    op = pd.DataFrame({s: u.stocks[s].prices["open"] for s in sids}).reindex(cal)
    cl = pd.DataFrame({s: u.stocks[s].prices["close"] for s in sids}).reindex(cal)
    sp = pd.DataFrame({s: u.stocks[s].prices["spread"] for s in sids}).reindex(cal)
    dv = pd.DataFrame({s: u.stocks[s].dividends for s in sids}).reindex(cal).fillna(0.0)
    mu = pd.DataFrame({s: u.stocks[s].share_mult for s in sids}).reindex(cal).fillna(1.0)
    last_close = cl.ffill()
    # 前 volume_window 個交易日（不含當天）的平均成交量：只用個股自己的交易日，再對到日曆
    avgv = pd.DataFrame({s: u.stocks[s].prices["volume"].astype(float).rolling(volume_window, min_periods=1)
                         .mean().shift(1) for s in sids}).reindex(cal).to_numpy()
    O, C, S, D, LC, MU = (op.to_numpy(), cl.to_numpy(), sp.to_numpy(), dv.to_numpy(), last_close.to_numpy(),
                          mu.to_numpy())
    sig_days = {pd.Timestamp(d) for d in targets}
    # 每檔最後一個交易日（下市 / 資料結束）；之後仍持有就以最後收盤價扣成本出清（對下市破產的股票偏樂觀）
    last_day = np.array([cal.searchsorted(u.stocks[s].prices.index.max(), side="right") - 1 for s in sids])
    data_end = len(cal) - 1
    can_delist = np.array([delisted is None or s in delisted for s in sids], dtype=bool)
    col = {s: k for k, s in enumerate(sids)}

    cash, shares = float(initial_cash), np.zeros(len(sids))
    pending: dict[int, float] = {}
    equity = np.empty(len(cal))
    trades, skipped, divs = [], [], []
    for i, day in enumerate(cal):
        # 1. 股利（以原股數）與公司行動（股數調整）：前一日收盤持有
        for k in np.flatnonzero((shares > 0) & (D[i] > 0)):
            amt = shares[k] * D[i, k] * (1 - costs.dividend_tax_rate)
            cash += amt
            divs.append(dict(date=day, shares=shares[k], per_share=D[i, k], amount=amt))
        shares = shares * MU[i]
        # 2. 前一天是訊號日 → 依前一天收盤計算目標股數
        if i > 0 and cal[i - 1] in sig_days:
            picks = targets[cal[i - 1]]
            # 今天的參考價：前一日收盤扣掉今天的股利、再除以股數乘數（股數已在上面調整過），單位才一致
            refp = (LC[i - 1] - D[i]) / MU[i]
            total = cash + float(np.nansum(shares * refp))
            per = total * (1 - cash_buffer) / len(picks) if picks else 0.0
            pending = {}
            for k in range(len(sids)):
                tgt = math.floor(per / refp[k]) if sids[k] in picks and refp[k] > 0 else 0
                if tgt != shares[k]:
                    pending[k] = tgt
        # 3. 開盤執行：先賣後買
        if pending:
            for k in sorted(pending, key=lambda k: pending[k] - shares[k]):
                tgt, px = pending[k], O[i, k]
                if np.isnan(px):
                    continue
                prev = LC[i - 1, k] if i > 0 and not np.isnan(LC[i - 1, k]) else px
                ref = C[i, k] - S[i, k] if not np.isnan(S[i, k]) else prev
                if not (ref > 0 and abs(ref / prev - 1) <= costs.price_limit):
                    ref = prev
                up, down = limit_prices(ref, costs.price_limit)
                diff = tgt - shares[k]
                if diff < 0:
                    if px <= down + 1e-9:
                        skipped.append(dict(date=day, side="sell", price=px, reason=f"{sids[k]} 開盤跌停，賣不掉"))
                        continue
                    value = -diff * px
                    fee, tax = costs.commission(value), costs.sell_tax(value)
                    cash += value - fee - tax
                    shares[k] = tgt
                    trades.append(dict(date=day, signal_date=None, side="sell", price=px, shares=-diff, value=value,
                                       commission=fee, tax=tax, capped=False, stock_id=sids[k]))
                    del pending[k]
                elif diff > 0:
                    if px >= up - 1e-9:
                        skipped.append(dict(date=day, side="buy", price=px, reason=f"{sids[k]} 開盤漲停，買不到"))
                        continue
                    qty, capped = int(diff), False
                    if max_participation is not None:
                        cap = 0 if np.isnan(avgv[i, k]) else math.floor(avgv[i, k] * max_participation)
                        if cap < qty:
                            qty, capped = cap, True
                    if qty * px + costs.commission(qty * px) > cash:   # 現金不夠就買能買的最大股數
                        rate = costs.commission_rate * costs.commission_discount
                        qty = max(math.floor((cash - costs.min_commission) / (px * (1 + rate))), 0)
                        while qty > 0 and qty * px + costs.commission(qty * px) > cash:
                            qty -= 1
                    if qty <= 0:
                        continue
                    value = qty * px
                    fee = costs.commission(value)
                    cash -= value + fee
                    shares[k] += qty
                    trades.append(dict(date=day, signal_date=None, side="buy", price=px, shares=qty, value=value,
                                       commission=fee, tax=0, capped=capped, stock_id=sids[k]))
                    if shares[k] >= tgt:
                        del pending[k]
                else:
                    del pending[k]
        for k in np.flatnonzero((shares > 0) & (last_day == i) & (last_day < data_end) & can_delist):
            value = shares[k] * LC[i, k] * delist_recovery
            fee, tax = (costs.commission(value), costs.sell_tax(value)) if value > 0 else (0, 0)
            cash += value - fee - tax
            trades.append(dict(date=day, signal_date=None, side="sell", price=LC[i, k] * delist_recovery,
                               shares=shares[k], value=value,
                               commission=fee, tax=tax, capped=False, stock_id=sids[k]))
            skipped.append(dict(date=day, side="sell", price=LC[i, k], reason=f"{sids[k]} 資料結束（下市），以最後收盤出清"))
            shares[k] = 0
            pending.pop(k, None)
        equity[i] = cash + float(np.nansum(shares * LC[i]))
    tr = pd.DataFrame(trades, columns=["date", "signal_date", "side", "price", "shares", "value", "commission", "tax",
                                       "capped", "stock_id"])
    return BacktestResult(
        equity=pd.Series(equity, index=cal, name="equity"), trades=tr,
        round_trips=pd.DataFrame(columns=["entry_date", "exit_date", "shares", "cost", "proceeds", "dividends",
                                          "pnl", "ret"]),
        skipped=pd.DataFrame(skipped, columns=["date", "side", "price", "reason"]),
        total_commission=float(tr["commission"].sum()), total_tax=float(tr["tax"].sum()),
        dividends=pd.DataFrame(divs, columns=["date", "shares", "per_share", "amount"]),
        total_dividends=float(sum(d["amount"] for d in divs)),
        params={"col": col})


def turnover(res: BacktestResult) -> float:
    """年化週轉率：每年買進金額 / 平均資產。"""
    years = max(len(res.equity) / 252, 1e-9)
    return float(res.trades.loc[res.trades["side"] == "buy", "value"].sum() / res.equity.mean() / years)
