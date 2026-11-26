"""Oracle 測試（最重要）：正確版本全部通過，三個錯誤版本都必須被擋下。"""
import numpy as np
import pandas as pd
import pytest

from src import oracles
from src.backtest import Backtester
from src.features import build_features
from src.strategies import lgbm
from tests.mutants import NoTaxBacktester, SameDayCloseBacktester, lookahead_features

ENGINE_CHECKS = [oracles.check_execution_timing, oracles.check_cost_hand_calc]


# ---------- 1. 作弊模型 ----------
def test_cheat_model_is_absurdly_good():
    r = oracles.check_cheat_detected(Backtester)
    assert r.passed, r.detail


def test_cheat_model_absurd_in_odd_lot_mode_too():
    r = oracles.check_cheat_detected(Backtester, lot_mode="odd", initial_cash=100_000)
    assert r.passed, r.detail


# ---------- 2. 三個錯誤版本都要被擋下 ----------
@pytest.mark.parametrize("check", ENGINE_CHECKS)
def test_correct_engine_passes(check):
    r = check(Backtester)
    assert r.passed, r.detail


def test_mutant_same_day_close_is_caught():
    assert not oracles.check_execution_timing(SameDayCloseBacktester).passed


def test_mutant_no_sell_tax_is_caught():
    assert not oracles.check_cost_hand_calc(NoTaxBacktester).passed


def test_correct_features_have_no_lookahead():
    r = oracles.check_feature_no_lookahead(build_features)
    assert r.passed, r.detail


def test_mutant_lookahead_features_is_caught():
    assert not oracles.check_feature_no_lookahead(lookahead_features).passed


def test_lookahead_features_make_lgbm_suspiciously_good():
    """額外佐證：偷看未來的特徵讓模型準確率異常高，正確特徵則接近擲硬幣。"""
    prices = oracles.synthetic_prices(1200, seed=3)
    kw = dict(train_size=400, test_size=100, gap=5, params=dict(n_estimators=100, verbose=-1, random_state=0))
    _, honest, _ = lgbm.generate_signals(prices, return_details=True, **kw)
    _, cheat, _ = lgbm.generate_signals(prices, return_details=True, feature_fn=lookahead_features, **kw)
    assert honest["accuracy"].mean() < 0.58
    assert cheat["accuracy"].mean() > 0.75


# ---------- 3. 手算成本（細項）----------
def test_hand_calculated_board_lot_trade():
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"date": dates, "open": [100.0, 100.0, 110.0], "high": [101, 106, 111],
                           "low": [99, 99, 109], "close": [100.0, 105.0, 110.0], "volume": 1000, "spread": np.nan})
    res = Backtester(oracles.Costs(), 200_000, "board").run(prices, pd.Series([1.0, 0.0, np.nan], index=dates))
    buy, sell = res.trades.iloc[0], res.trades.iloc[1]
    assert (buy.price, buy.shares, buy.commission, buy.tax) == (100.0, 1000, 142, 0)
    assert (sell.price, sell.shares, sell.commission, sell.tax) == (110.0, 1000, 156, 330)
    assert res.total_commission == 298 and res.total_tax == 330
    assert res.round_trips.iloc[0].pnl == 109_514 - 100_142 == 9_372
    assert res.equity.iloc[-1] == 209_372


def test_commission_discount_and_minimum():
    c = oracles.Costs(commission_discount=0.6)
    assert c.commission(100_000) == 85            # floor(142.5 * 0.6 = 85.5)
    assert c.commission(10_000) == 20             # floor(8.55) → 最低 20
    assert c.sell_tax(100_000) == 300


# ---------- 4. 隨機訊號 ----------
def test_random_signals_lose_to_buy_and_hold_after_costs():
    r = oracles.check_random_vs_bh(Backtester)
    assert r.passed, r.detail


# ---------- 5. walk-forward ----------
def test_walk_forward_train_always_before_test():
    prices = oracles.synthetic_prices(900, seed=1)
    gap = 5
    _, folds, _ = lgbm.generate_signals(prices, train_size=300, test_size=100, gap=gap,
                                        params=dict(n_estimators=20, verbose=-1), return_details=True)
    r = oracles.check_walk_forward(folds, pd.DatetimeIndex(prices["date"]), gap)
    assert r.passed, r.detail
    assert (folds["train_end"] < folds["test_start"]).all()


def test_walk_forward_check_catches_overlap():
    dates = pd.bdate_range("2024-01-01", periods=20)
    bad = pd.DataFrame([dict(fold=0, train_start=dates[0], train_end=dates[10], test_start=dates[8],
                             test_end=dates[15])])
    assert not oracles.check_walk_forward(bad, dates, gap=1).passed
    no_gap = pd.DataFrame([dict(fold=0, train_start=dates[0], train_end=dates[10], test_start=dates[11],
                                test_end=dates[15])])
    assert not oracles.check_walk_forward(no_gap, dates, gap=5).passed


def test_walk_forward_splits_reject_zero_gap():
    with pytest.raises(ValueError):
        lgbm.walk_forward_splits(100, 50, 10, gap=0)


def test_walk_forward_splits_indices():
    for f in lgbm.walk_forward_splits(200, 50, 20, gap=3):
        assert f.train_idx.max() + 3 < f.test_idx.min()
        assert len(f.train_idx) == 50
