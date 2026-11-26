import numpy as np
import pandas as pd

from src import oracles
from src.backtest import Backtester, Costs
from src.kline import detect_patterns, pattern_stats, volume_indicators
from src.strategies import candle_volume


def _bars(rows, volume=1000):
    """rows: [(open, high, low, close), ...]"""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)
    df.insert(0, "date", pd.bdate_range("2024-01-01", periods=len(df)))
    df["volume"] = volume
    df["spread"] = np.nan
    return df


def _pad(n=25, price=100.0):
    return [(price, price + 1, price - 1, price + 0.2)] * n


def test_bull_and_bear_engulfing():
    bull = detect_patterns(_bars(_pad() + [(101, 101.5, 98, 98.5), (98, 102.5, 97.5, 102)]))
    assert bull["bull_engulfing"].iloc[-1] and not bull["bear_engulfing"].iloc[-1]
    bear = detect_patterns(_bars(_pad() + [(99, 102, 98.5, 101.5), (102, 102.5, 98, 98.5)]))
    assert bear["bear_engulfing"].iloc[-1] and not bear["bull_engulfing"].iloc[-1]


def test_doji_and_hammer():
    falling = [(110 - i, 111 - i, 108 - i, 109 - i) for i in range(25)]
    df = _bars(falling + [(85, 85.6, 80, 85.4), (86, 88, 84, 86.05)])
    pats = detect_patterns(df)
    assert pats["hammer"].iloc[-2]          # 長下影線、小實體、跌勢後
    assert pats["doji"].iloc[-1]            # 實體 0.05 / 區間 4


def test_volume_breakout_needs_volume():
    rows = _pad() + [(100.5, 106, 100, 105.5)]
    quiet = _bars(rows, volume=[1000] * 26)
    loud = _bars(rows, volume=[1000] * 25 + [5000])
    assert not detect_patterns(quiet)["volume_breakout"].iloc[-1]
    assert detect_patterns(loud)["volume_breakout"].iloc[-1]


def test_kline_and_volume_have_no_lookahead():
    for fn in (detect_patterns, volume_indicators, lambda p: candle_volume.generate_signals(p).to_frame()):
        r = oracles.check_feature_no_lookahead(lambda p: fn(p).astype(float))
        assert r.passed, r.detail


def test_lookahead_check_catches_shifted_patterns():
    r = oracles.check_feature_no_lookahead(lambda p: detect_patterns(p).astype(float).shift(-1))
    assert not r.passed


def test_candle_volume_signal_is_binary_and_trades_next_open():
    prices = oracles.synthetic_prices(800, seed=5)
    sig = candle_volume.generate_signals(prices)
    assert set(sig.dropna().unique()) <= {0.0, 1.0}
    res = Backtester(Costs(), 1_000_000).run(prices, sig)
    assert len(res.trades) > 0
    assert (res.trades["date"] > res.trades["signal_date"]).all()


def test_candle_volume_respects_max_hold():
    prices = oracles.synthetic_prices(800, seed=5)
    sig = candle_volume.generate_signals(prices, max_hold=3).fillna(0)
    runs = (sig != sig.shift()).cumsum()[sig == 1]
    assert runs.value_counts().max() <= 3


def test_volume_cap_limits_buy_size_and_skips_illiquid():
    df = _bars(_pad(5), volume=10_000)       # 均量 1 萬股，5% = 500 股
    sig = pd.Series([1.0] + [np.nan] * 4, index=df["date"])
    odd = Backtester(Costs(), 1_000_000, "odd", max_volume_participation=0.05).run(df, sig)
    assert odd.trades.iloc[0].shares == 500 and odd.trades.iloc[0].capped
    board = Backtester(Costs(), 1_000_000, "board", max_volume_participation=0.05).run(df, sig)
    assert board.trades.empty
    assert board.skipped["reason"].str.startswith("流動性不足").all() and len(board.skipped) == 4


def test_volume_cap_uses_only_past_volume():
    # 成交當天爆量不能讓當天買更多：上限只看前幾日均量
    df = _bars(_pad(3), volume=[10_000, 10_000, 1_000_000])
    sig = pd.Series([np.nan, 1.0, np.nan], index=df["date"])
    res = Backtester(Costs(), 1_000_000, "odd", max_volume_participation=0.05).run(df, sig)
    assert res.trades.iloc[0].shares == 500


def test_pattern_stats_has_baseline_row():
    stats = pattern_stats(oracles.synthetic_prices(600, seed=2))
    assert stats.iloc[-1]["pattern"] == "all_days" and stats.iloc[-1]["次數"] > 500
