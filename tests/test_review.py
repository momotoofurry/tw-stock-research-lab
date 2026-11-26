"""預測檢討：信賴區間、大盤環境、失效警示、版本拆分。不連網。"""
import numpy as np
import pandas as pd
import pytest

from src import review


def _scored(n, acc, start="2026-01-01", version="v1", cover=0.68, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    outcome = rng.integers(0, 2, n).astype(float)
    hit = rng.random(n) < acc
    pred_up = np.where(hit, outcome > 0.5, outcome < 0.5)
    return pd.DataFrame({"date": dates, "stock_id": "X", "lgbm_p": np.where(pred_up, 0.7, 0.3),
                         "lgbm_threshold": 0.55, "lgbm_outcome": outcome,
                         "in_r5": (rng.random(n) < cover).astype(float), "in_r20": np.nan,
                         "model_version": version})


def test_wilson_interval():
    lo, hi = review.wilson(50, 100)
    assert lo == pytest.approx(0.404, abs=0.002) and hi == pytest.approx(0.596, abs=0.002)
    assert all(np.isnan(review.wilson(0, 0)))


def test_regime_uses_only_past():
    c = np.r_[np.linspace(100, 130, 100), np.linspace(130, 100, 60)]
    px = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=160), "close": c})
    reg = review.regime_table(px)
    assert reg.iloc[95] == "多頭" and reg.iloc[-1] == "空頭" and pd.isna(reg.iloc[10])
    cut = review.regime_table(px.iloc[:120])
    assert (cut.dropna() == reg.iloc[:120].dropna()).all()


def test_no_conclusion_with_small_sample():
    r = review.build_review({}, scored=_scored(10, 0.2), index_px=None)
    assert r["alerts"] == [] and "樣本不足" in review.review_text(r)


def test_alert_when_recent_accuracy_clearly_below_naive():
    s = pd.concat([_scored(60, 0.6, seed=1), _scored(40, 0.15, start="2026-04-01", seed=2)], ignore_index=True)
    r = review.build_review({}, scored=s, index_px=None)
    assert any("可能失效" in a for a in r["alerts"])


def test_alert_when_range_too_narrow_and_quiet_when_ok():
    r = review.build_review({}, scored=_scored(60, 0.55, cover=0.3, seed=3), index_px=None)
    assert any("區間偏窄" in a for a in r["alerts"])
    ok = review.build_review({}, scored=_scored(60, 0.55, cover=0.68, seed=4), index_px=None)
    assert not any("波動範圍" in a for a in ok["alerts"])


def test_split_by_version():
    s = pd.concat([_scored(30, 0.5, version="aaa", seed=5), _scored(30, 0.5, start="2026-03-01", version="bbb", seed=6)])
    r = review.build_review({}, scored=s, index_px=None)
    assert set(r["by_version"]) == {"aaa", "bbb"} and r["by_version"]["aaa"]["n"] == 30


def test_brier_and_logloss():
    s = _scored(40, 1.0, seed=7)                                     # 全對，機率 0.7 / 0.3
    b = review.build_review({}, scored=s, index_px=None)["overall"]
    assert b["brier"] == pytest.approx(0.09) and b["logloss"] == pytest.approx(-np.log(0.7))
    assert "Brier" in review.review_text(review.build_review({}, scored=s, index_px=None))
