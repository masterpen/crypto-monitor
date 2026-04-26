import pandas as pd
from typing import Optional
from .base import BaseStrategy
from .indicators import calculate_kdj


class KDJStrategy(BaseStrategy):
    """KDJ超卖策略 - J值负值买入"""

    def __init__(self, period: int = 9, k_period: int = 3, d_period: int = 3):
        super().__init__("KDJStrategy")
        self.period = period
        self.k_period = k_period
        self.d_period = d_period
        self.set_params(period=period, k_period=k_period, d_period=d_period)

    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs) -> str:
        if index < self.period + 5:
            return 'hold'

        k, d, j = calculate_kdj(data, self.period, self.k_period, self.d_period)

        current_j = j.iloc[index]
        prev_j = j.iloc[index - 1]

        if position_side == 'long':
            return 'close' if prev_j < 0 and current_j >= 0 else 'hold'

        if position_side == 'flat' and current_j < 0:
            return 'long'

        return 'hold'