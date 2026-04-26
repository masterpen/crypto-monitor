import pandas as pd
from typing import Optional
from .base import BaseStrategy
from .indicators import calculate_kdj


class KDJCrossStrategy(BaseStrategy):
    """KDJ金叉死叉策略"""

    def __init__(self, period: int = 9, overbought: float = 80, oversold: float = 20):
        super().__init__("KDJCrossStrategy")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.set_params(period=period, overbought=overbought, oversold=oversold)

    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs) -> str:
        if index < self.period + 5:
            return 'hold'

        k, d, j = calculate_kdj(data, self.period)

        current_k = k.iloc[index]
        current_d = d.iloc[index]
        current_j = j.iloc[index]
        prev_k = k.iloc[index - 1]
        prev_d = d.iloc[index - 1]

        if position_side == 'long':
            if prev_k >= prev_d and current_k < current_d:
                return 'close'
            if current_k > self.overbought:
                return 'close'
            return 'hold'

        if position_side == 'short':
            if prev_k <= prev_d and current_k > current_d:
                return 'close'
            if current_k < self.oversold:
                return 'close'
            return 'hold'

        if prev_k <= prev_d and current_k > current_d and current_d < self.oversold:
            return 'long'
        if prev_k >= prev_d and current_k < current_d and current_d > self.overbought:
            return 'short'

        if current_k < 0 or current_j < 0:
            return 'long'

        return 'hold'