"""故意埋錯的版本。oracle 檢查必須把每一個都擋下來。"""
from __future__ import annotations

from src.backtest import Backtester, Costs
from src.features import build_features


class SameDayCloseBacktester(Backtester):
    """錯誤 1：用第 t 天收盤價成交（應為第 t+1 天開盤價）。"""
    execution_lag = 0
    execution_field = "close"


class _NoTaxCosts(Costs):
    def sell_tax(self, value: float) -> int:
        return 0


class NoTaxBacktester(Backtester):
    """錯誤 2：忘記扣證交稅。"""

    def __init__(self, costs: Costs, *args, **kwargs):
        super().__init__(_NoTaxCosts(**vars(costs)), *args, **kwargs)


def lookahead_features(prices):
    """錯誤 3：特徵往未來偏移一天（第 t 列拿到第 t+1 天的特徵）。"""
    return build_features(prices).shift(-1)
