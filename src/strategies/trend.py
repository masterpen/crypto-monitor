import pandas as pd
import numpy as np
from typing import Optional
from .base import BaseStrategy


class TrendStrategy(BaseStrategy):
    """趋势跟踪策略 - 双均线交叉"""

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__("TrendStrategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.set_params(fast_period=fast_period, slow_period=slow_period)

    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs) -> str:
        if index < self.slow_period:
            return 'hold'

        fast_ma = data['close'].iloc[index - self.fast_period + 1:index + 1].mean()
        slow_ma = data['close'].iloc[index - self.slow_period + 1:index + 1].mean()

        prev_fast_ma = data['close'].iloc[index - self.fast_period:index].mean()
        prev_slow_ma = data['close'].iloc[index - self.slow_period:index].mean()

        golden_cross = prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma
        death_cross = prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma

        if position_side == 'long' and death_cross:
            return 'close'
        if position_side == 'short' and golden_cross:
            return 'close'

        if position_side == 'flat':
            if golden_cross:
                return 'long'
            if death_cross:
                return 'short'

        return 'hold'