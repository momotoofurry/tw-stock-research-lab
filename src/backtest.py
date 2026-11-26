"""台股回測引擎（只做模擬，無任何下單功能）。

時序規則：
- 訊號 signals[t] 由第 t 天收盤後（含）的資料產生，代表「希望持有的部位」（1 = 持有，0 = 空手）。
- 在第 t+1 天以「開盤價」成交。絕不在第 t 天用收盤價成交。
- 開盤即漲停買不到、開盤即跌停賣不掉：跳過並記錄於 skipped，之後的交易日若訊號仍成立會再嘗試。
成本：手續費 commission_rate * discount（無條件捨去至整數元，最低 min_commission），賣出另收證交稅 sell_tax_rate。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LOT_SIZES = {"board": 1000, "odd": 1}


@dataclass
class Costs:
    commission_rate: float = 0.001425
    commission_discount: float = 1.0
    min_commission: float = 20
    sell_tax_rate: float = 0.003
    price_limit: float = 0.10
    dividend_tax_rate: float = 0.0   # 股利稅負（二代健保 / 所得稅依個人狀況，預設不計）

    @classmethod
    def from_config(cls, cfg: dict) -> "Costs":
        return cls(**cfg["costs"])

    def commission(self, value: float) -> int:
        fee = math.floor(value * self.commission_rate * self.commission_discount)
        return int(max(fee, self.min_commission))

    def sell_tax(self, value: float) -> int:
        return int(math.floor(value * self.sell_tax_rate))


def tick_size(price: float) -> float:
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    if price < 100:
        return 0.1
    if price < 500:
        return 0.5
    if price < 1000:
        return 1.0
    return 5.0


def limit_prices(ref: float, limit: float) -> tuple[float, float]:
    """依參考價算漲停 / 跌停價（漲停向下取到檔位，跌停向上取到檔位）。"""
    up_raw, down_raw = ref * (1 + limit), ref * (1 - limit)
    tu, td = tick_size(up_raw), tick_size(down_raw)
    up = math.floor(round(up_raw / tu, 6)) * tu
    down = math.ceil(round(down_raw / td, 6)) * td
    return round(up, 2), round(down, 2)


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame          # 每筆成交
    round_trips: pd.DataFrame     # 每次買進到賣出（pnl 含持有期間領到的股利）
    skipped: pd.DataFrame         # 漲跌停 / 資金不足 / 流動性不足而跳過的委託
    total_commission: float
    total_tax: float
    dividends: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["date", "shares", "per_share", "amount"]))
    total_dividends: float = 0.0
    params: dict = field(default_factory=dict)


class Backtester:
    # 以下兩個屬性定義成交時點；只有 oracle 測試的「錯誤版本」會改動它們
    execution_lag = 1          # 訊號後第幾個交易日成交
    execution_field = "open"   # 成交價欄位

    def __init__(self, costs: Costs, initial_cash: float = 1_000_000, lot_mode: str = "board",
                 max_volume_participation: float | None = None, volume_window: int = 20):
        self.costs = costs
        self.initial_cash = float(initial_cash)
        self.lot = LOT_SIZES[lot_mode]
        self.lot_mode = lot_mode
        # 每筆買進股數上限 = 前一交易日為止 volume_window 日均量 × 比例（None = 不限制）
        self.max_volume_participation = max_volume_participation
        self.volume_window = volume_window

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "Backtester":
        bt = cfg["backtest"]
        kw = dict(costs=Costs.from_config(cfg), initial_cash=bt["initial_cash"], lot_mode=bt["lot_mode"],
                  max_volume_participation=bt.get("max_volume_participation"),
                  volume_window=bt.get("volume_window", 20))
        kw.update(overrides)
        return cls(**kw)

    def _reference_price(self, close: float, spread: float, prev_close: float) -> float:
        # 當日參考價 = 收盤 - 漲跌價差（FinMind spread），開盤前即公告；除權息日與前一日收盤不同。
        # 缺 spread 時退回前一日收盤。
        # 與前一日收盤差距超過漲跌幅（資料異常）時也退回前一日收盤。
        if not np.isnan(spread):
            ref = close - spread
            if ref > 0 and abs(ref / prev_close - 1) <= self.costs.price_limit:
                return ref
        return prev_close

    def _shares_affordable(self, cash: float, price: float) -> int:
        lots = int(cash // (price * self.lot * (1 + self.costs.commission_rate * self.costs.commission_discount)))
        while lots > 0:
            value = lots * self.lot * price
            if value + self.costs.commission(value) <= cash:
                return lots * self.lot
            lots -= 1
        return 0

    def _volume_cap(self, avg_volume: float) -> int | None:
        if self.max_volume_participation is None:
            return None
        if np.isnan(avg_volume):
            return 0
        return int(avg_volume * self.max_volume_participation // self.lot) * self.lot

    def run(self, prices: pd.DataFrame, signals: pd.Series,
            dividends: pd.DataFrame | None = None) -> BacktestResult:
        """dividends：價格事件（date, before_price, after_price, kind）。前一日收盤仍持有者：
        kind=dividend 在除權息日領到每股 before − after 的現金（配股也折成現金），扣 dividend_tax_rate；
        kind=corporate（減資、分割、面額變更）股數乘上 before / after，以參考價計值不變。"""
        df = prices.sort_values("date").reset_index(drop=True)
        dates = pd.DatetimeIndex(df["date"])
        n = len(df)
        sig = signals.reindex(dates).to_numpy(dtype=float)  # 沒有訊號的日子 = NaN = 維持現狀
        exec_px = df[self.execution_field].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        spread = df["spread"].to_numpy(dtype=float) if "spread" in df else np.full(n, np.nan)
        # 成交日之前（不含當天）的平均成交量：成交當下還不知道當天的量
        avg_vol = df["volume"].astype(float).rolling(self.volume_window, min_periods=1).mean().shift(1).to_numpy()
        div_ps, share_mult = np.zeros(n), np.ones(n)
        if dividends is not None and len(dividends):
            kind = dividends["kind"] if "kind" in dividends else pd.Series("dividend", index=dividends.index)
            cashd = dividends[kind == "dividend"]
            d = cashd.assign(ps=cashd["before_price"] - cashd["after_price"])
            d = d[d["ps"] > 0].groupby("date")["ps"].sum()
            div_ps = d.reindex(dates).fillna(0.0).to_numpy()
            corp = dividends[kind == "corporate"]
            if len(corp):
                r = (corp["before_price"] / corp["after_price"]).groupby(corp["date"]).prod()
                share_mult = r.reindex(dates).fillna(1.0).to_numpy()
        lag = self.execution_lag
        is_open = self.execution_field == "open"

        cash, shares = self.initial_cash, 0
        entry_cost, entry_date, entry_divs = 0.0, None, 0.0
        equity = np.empty(n)
        trades, rounds, skipped, divs = [], [], [], []
        tot_comm = tot_tax = tot_div = 0.0

        want, want_date = None, None  # 最近一次訊號要求的部位；被漲跌停擋下後會持續再試
        for i in range(n):
            if shares > 0 and div_ps[i] > 0:  # 前一日收盤持有 → 除權息日入帳（以原股數）
                amount = shares * div_ps[i] * (1 - self.costs.dividend_tax_rate)
                cash += amount
                tot_div += amount
                entry_divs += amount
                divs.append(dict(date=dates[i], shares=shares, per_share=div_ps[i], amount=amount))
            if shares > 0 and share_mult[i] != 1.0:  # 減資 / 分割 / 配股：以參考價計值不變，調整股數
                shares = shares * share_mult[i]
            j = i - lag  # 產生訊號的那一天
            if j >= 0 and not np.isnan(sig[j]):
                want, want_date = int(sig[j] > 0), dates[j]
            if want is not None and want != int(shares > 0):
                price = exec_px[i]
                prev_close = close[i - 1] if i > 0 else close[i]
                up, down = limit_prices(self._reference_price(close[i], spread[i], prev_close),
                                        self.costs.price_limit)

                if want == 1 and shares == 0:
                    if is_open and price >= up - 1e-9:
                        skipped.append(dict(date=dates[i], side="buy", price=price, reason="開盤漲停，買不到"))
                    else:
                        qty = self._shares_affordable(cash, price)
                        cap = self._volume_cap(avg_vol[i])
                        if qty == 0:
                            skipped.append(dict(date=dates[i], side="buy", price=price, reason="資金不足一個交易單位"))
                        elif cap == 0:
                            skipped.append(dict(date=dates[i], side="buy", price=price,
                                                reason="流動性不足：均量上限不到一個交易單位"))
                            qty = 0
                        capped = cap is not None and 0 < cap < qty
                        if capped:
                            qty = cap
                        if qty > 0:
                            value = qty * price
                            fee = self.costs.commission(value)
                            cash -= value + fee
                            shares = qty
                            tot_comm += fee
                            entry_cost, entry_date, entry_divs = value + fee, dates[i], 0.0
                            trades.append(dict(date=dates[i], signal_date=want_date, side="buy", price=price,
                                               shares=qty, value=value, commission=fee, tax=0, capped=capped))
                elif want == 0 and shares > 0:
                    if is_open and price <= down + 1e-9:
                        skipped.append(dict(date=dates[i], side="sell", price=price, reason="開盤跌停，賣不掉"))
                    else:
                        value = shares * price
                        fee = self.costs.commission(value)
                        tax = self.costs.sell_tax(value)
                        proceeds = value - fee - tax
                        cash += proceeds
                        tot_comm += fee
                        tot_tax += tax
                        trades.append(dict(date=dates[i], signal_date=want_date, side="sell", price=price,
                                           shares=shares, value=value, commission=fee, tax=tax, capped=False))
                        pnl = proceeds + entry_divs - entry_cost
                        rounds.append(dict(entry_date=entry_date, exit_date=dates[i], shares=shares,
                                           cost=entry_cost, proceeds=proceeds, dividends=entry_divs,
                                           pnl=pnl, ret=pnl / entry_cost))
                        shares = 0
            equity[i] = cash + shares * close[i]

        return BacktestResult(
            equity=pd.Series(equity, index=dates, name="equity"),
            trades=pd.DataFrame(trades, columns=["date", "signal_date", "side", "price", "shares", "value",
                                                 "commission", "tax", "capped"]),
            round_trips=pd.DataFrame(rounds, columns=["entry_date", "exit_date", "shares", "cost", "proceeds",
                                                      "dividends", "pnl", "ret"]),
            skipped=pd.DataFrame(skipped, columns=["date", "side", "price", "reason"]),
            total_commission=tot_comm,
            total_tax=tot_tax,
            dividends=pd.DataFrame(divs, columns=["date", "shares", "per_share", "amount"]),
            total_dividends=tot_div,
            params=dict(lot_mode=self.lot_mode, initial_cash=self.initial_cash, costs=vars(self.costs)),
        )


def buy_and_hold_signals(prices: pd.DataFrame, start=None) -> pd.Series:
    """買進持有：從 start（訊號日）起每天訊號都是 1。"""
    dates = pd.DatetimeIndex(prices["date"])
    s = pd.Series(np.nan, index=dates)
    s[dates >= (pd.Timestamp(start) if start is not None else dates[0])] = 1.0
    return s
