import pandas as pd

from src.research import trailing_pe


def test_trailing_pe_uses_last_four_quarters():
    eps = pd.Series([-1.32, 0.50, 3.58, 8.41, 14.66])
    pe, total = trailing_pe(454.5, eps)
    assert total == 27.15
    assert abs(pe - 454.5 / 27.15) < 1e-9


def test_trailing_pe_none_when_short_or_loss():
    assert trailing_pe(100, pd.Series([1.0, 2.0, 3.0])) is None
    assert trailing_pe(100, pd.Series([-5.0, 1.0, 1.0, 1.0])) is None
