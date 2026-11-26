"""第 2 步（降低換手）與看盤介面資料層。"""
import numpy as np
import pandas as pd
import pytest

from src import dashboard_data as dd
from src import oracles
from src.features import make_label
from src.strategies import lgbm


# ---------- 標籤與 walk-forward ----------
def test_label_horizon_and_hurdle():
    dates = pd.bdate_range("2024-01-01", periods=6)
    p = pd.DataFrame({"date": dates, "close": [100, 101, 99, 103, 104, 102.0]})
    y = make_label(p, horizon=2, hurdle=0.02)
    # t=0: 99/100-1 = -1% → 0；t=1: 103/101-1 = +1.98% → 0；t=2: 104/99-1 = +5% → 1；最後 2 列未知
    assert list(y.iloc[:4]) == [0.0, 0.0, 1.0, 0.0]
    assert y.iloc[-2:].isna().all()


def test_walk_forward_requires_gap_at_least_horizon():
    with pytest.raises(ValueError):
        lgbm.walk_forward_splits(300, 100, 20, gap=4, horizon=5)
    for f in lgbm.walk_forward_splits(300, 100, 20, gap=5, horizon=5):
        # 訓練最後一列的標籤用到 train_end + horizon，必須早於測試第一列
        assert f.train_idx.max() + 5 < f.test_idx.min()


def test_long_horizon_walk_forward_passes_oracle():
    prices = oracles.synthetic_prices(900, seed=1)
    _, folds, _ = lgbm.generate_signals(prices, train_size=300, test_size=100, gap=10, horizon=10, hurdle=0.01,
                                        params=dict(n_estimators=20, verbose=-1), return_details=True)
    r = oracles.check_walk_forward(folds, pd.DatetimeIndex(prices["date"]), gap=10)
    assert r.passed, r.detail


# ---------- 遲滯 ----------
def test_hysteresis_enter_exit_and_min_hold():
    p = pd.Series([0.5, 0.6, 0.5, 0.42, 0.3, 0.3, 0.7, 0.4, np.nan, 0.4])
    s = lgbm.hysteresis(p, enter=0.55, exit_=0.45, min_hold=1)
    assert list(s.iloc[:8]) == [0, 1, 1, 0, 0, 0, 1, 0]
    assert np.isnan(s.iloc[8])
    s3 = lgbm.hysteresis(p, enter=0.55, exit_=0.45, min_hold=3)
    assert list(s3.iloc[:6]) == [0, 1, 1, 1, 0, 0]   # 0.42 時只持有 2 天，要撐到第 3 天才出場


def test_default_hysteresis_equals_plain_threshold():
    rng = np.random.default_rng(0)
    p = pd.Series(rng.random(500))
    plain = (p > 0.5).astype(float)
    assert (lgbm.hysteresis(p, 0.5, 0.5, 1) == plain).all()


def test_hysteresis_reduces_turnover():
    rng = np.random.default_rng(1)
    p = pd.Series(0.5 + 0.08 * rng.standard_normal(1000))
    flips = lambda s: int((s.diff().abs() > 0).sum())  # noqa: E731
    assert flips(lgbm.hysteresis(p, 0.55, 0.45, 5)) < flips(lgbm.hysteresis(p, 0.5, 0.5, 1)) / 2


# ---------- 看盤資料層 ----------
def test_analyze_levels_come_from_recent_window():
    prices = oracles.synthetic_prices(600, seed=6)
    a = dd.analyze(prices, prices, "2030-01-01")
    assert a["date"] == prices["date"].iloc[-1]
    assert a["levels"]["20 日高點"] == prices["high"].iloc[-20:].max()
    assert a["levels"]["60 日低點"] == prices["low"].iloc[-60:].min()
    assert a["levels"]["MA20"] == pytest.approx(prices["close"].iloc[-20:].mean())


def test_volatility_range_is_calibrated_on_random_walk():
    prices = oracles.synthetic_prices(1500, drift=0.0, seed=8)
    a = dd.analyze(prices, prices, "2030-01-01")
    assert 0.6 < a["coverage"][5] < 0.76 and 0.6 < a["coverage"][20] < 0.76
    r = a["ranges"].iloc[0]
    assert r["下緣_95"] < r["下緣_68"] < a["last"] < r["上緣_68"] < r["上緣_95"]


def test_lgbm_reliability_respects_holdout_cut():
    prices = oracles.synthetic_prices(800, seed=2)
    dates = pd.DatetimeIndex(prices["date"])
    proba = pd.Series(0.6, index=dates)
    cfg = {"strategies": {"lgbm": {"threshold": 0.5, "horizon": 1, "hurdle": 0.0}}}
    full = dd.lgbm_reliability(proba, prices, cfg)
    cut = dd.lgbm_reliability(proba, prices, cfg, before=dates[400])
    assert cut["n"] == 399 and full["n"] == 799   # 第 400 天的結果要看第 401 天（保留期內）→ 不計
    assert 0.4 < cut["accuracy"] < 0.6 and cut["naive"] >= 0.5


def test_reliability_outcomes_never_use_holdout_prices():
    """開發期最後一天的「結果」要用到隔天收盤；隔天在保留期內，所以必須是未知，而不是偷看保留期。"""
    prices = oracles.synthetic_prices(300, seed=4)
    dates = pd.DatetimeIndex(prices["date"])
    cut = dates[200]
    cfg = {"strategies": {"lgbm": {"threshold": 0.5, "horizon": 1, "hurdle": 0.0}}}
    y = dd._outcomes(prices, cfg, before=cut)
    assert y.index.max() < cut and np.isnan(y.iloc[-1])
    proba = pd.Series(0.6, index=dates)
    assert dd.lgbm_reliability(proba, prices, cfg, before=cut)["n"] == 199
    # 改動保留期的價格，開發期統計不可改變
    shocked = prices.copy()
    shocked.loc[200:, "close"] *= 3
    for fn in (lambda p: dd.lgbm_reliability(proba, p, cfg, before=cut)["accuracy"],
               lambda p: dd.prediction_calibration(proba, p, cfg, before=cut)["實際發生比例"].tolist(),
               lambda p: dd.rolling_accuracy(proba, p, cfg, window=20, before=cut).dropna().tolist()):
        assert fn(prices) == fn(shocked)


def test_walk_forward_oracle_checks_horizon():
    dates = pd.bdate_range("2024-01-01", periods=40)
    folds = pd.DataFrame([dict(fold=0, train_start=dates[0], train_end=dates[10], test_start=dates[20],
                               test_end=dates[30])])
    assert oracles.check_walk_forward(folds, dates, gap=5, horizon=5).passed
    assert not oracles.check_walk_forward(folds, dates, gap=5, horizon=10).passed
