import numpy as np
import pandas as pd

from src.backtest import Backtester, Costs, limit_prices, tick_size
from src.strategies import ma_cross


def _prices(opens, closes, spread=None):
    dates = pd.bdate_range("2024-01-01", periods=len(opens))
    return pd.DataFrame({"date": dates, "open": opens, "high": np.maximum(opens, closes),
                         "low": np.minimum(opens, closes), "close": closes, "volume": 1000,
                         "spread": spread if spread is not None else np.nan})


def test_limit_prices_follow_tick_rules():
    assert limit_prices(100, 0.1) == (110.0, 90.0)
    assert limit_prices(48.3, 0.1) == (53.1, 43.5)      # 53.13 → 53.1（0.1 檔）, 43.47 → 43.5（0.05 檔）
    assert tick_size(600) == 1.0 and tick_size(1200) == 5.0


def test_limit_up_open_cannot_buy_then_buys_next_day():
    p = _prices([100, 110, 112, 113], [100, 110, 112, 113])
    sig = pd.Series([1.0, np.nan, np.nan, np.nan], index=p["date"])
    res = Backtester(Costs(), 1_000_000).run(p, sig)
    assert len(res.skipped) == 1 and res.skipped.iloc[0].side == "buy"
    assert "漲停" in res.skipped.iloc[0].reason
    # 訊號仍是持有 → 隔天開盤 112（未漲停）買進
    sig2 = pd.Series([1.0, 1.0, np.nan, np.nan], index=p["date"])
    res2 = Backtester(Costs(), 1_000_000).run(p, sig2)
    assert res2.trades.iloc[0].date == p["date"][2] and res2.trades.iloc[0].price == 112


def test_blocked_order_retries_without_new_signal():
    p = _prices([100, 110, 112, 113], [100, 110, 112, 113])
    res = Backtester(Costs(), 1_000_000).run(p, pd.Series([1.0, np.nan, np.nan, np.nan], index=p["date"]))
    assert len(res.skipped) == 1
    assert res.trades.iloc[0].date == p["date"][2] and res.trades.iloc[0].signal_date == p["date"][0]


def test_limit_down_open_cannot_sell():
    p = _prices([100, 100, 90, 95], [100, 100, 90, 95])
    sig = pd.Series([1.0, 0.0, 0.0, np.nan], index=p["date"])
    res = Backtester(Costs(), 1_000_000).run(p, sig)
    assert list(res.skipped.side) == ["sell"]
    sells = res.trades[res.trades.side == "sell"]
    assert sells.iloc[0].date == p["date"][3] and sells.iloc[0].price == 95


def test_reference_price_uses_spread_on_ex_dividend_day():
    # 除息日：前一日收盤 100，參考價 95（spread 以參考價計算）→ 漲停 104.5，開盤 104.5 買不到
    p = _prices([100, 104.5], [100, 104.5], spread=[0.0, 9.5])
    res = Backtester(Costs(), 1_000_000).run(p, pd.Series([1.0, np.nan], index=p["date"]))
    assert len(res.skipped) == 1


def test_odd_lot_mode_buys_single_shares():
    p = _prices([100, 100, 100], [100, 100, 100])
    res = Backtester(Costs(), 50_000, "odd").run(p, pd.Series([1.0, np.nan, np.nan], index=p["date"]))
    assert res.trades.iloc[0].shares == 499       # 499*100 + 71 手續費 <= 50000
    res_b = Backtester(Costs(), 50_000, "board").run(p, pd.Series([1.0, np.nan, np.nan], index=p["date"]))
    assert res_b.trades.empty and res_b.skipped.iloc[0].reason.startswith("資金不足")


def test_never_trades_on_signal_day():
    p = _prices(list(range(100, 130)), [x + 0.5 for x in range(100, 130)])
    sig = ma_cross.generate_signals(p, 3, 5)
    res = Backtester(Costs(), 1_000_000).run(p, sig)
    assert (res.trades["date"] > res.trades["signal_date"]).all()
    assert len(res.trades) > 0


def test_ma_cross_uses_only_past():
    p = _prices(list(range(100, 160)), list(range(100, 160)))
    full = ma_cross.generate_signals(p)
    part = ma_cross.generate_signals(p.iloc[:40])
    pd.testing.assert_series_equal(full.iloc[:40], part)
