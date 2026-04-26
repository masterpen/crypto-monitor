import pandas as pd
from typing import Optional
from .base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略 - 布林带"""

    def __init__(self, period: int = 20, std_multiplier: float = 2.0):
        super().__init__("MeanReversionStrategy")
        self.period = period
        self.std_multiplier = std_multiplier
        self.set_params(period=period, std_multiplier=std_multiplier)

    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs) -> str:
        if index < self.period:
            return 'hold'

        recent = data['close'].iloc[index - self.period + 1:index + 1]
        middle = recent.mean()
        std = recent.std()

        upper_band = middle + self.std_multiplier * std
        lower_band = middle - self.std_multiplier * std

        current_price = data['close'].iloc[index]

        if position_side == 'long':
            return 'close' if current_price >= middle else 'hold'
        if position_side == 'short':
            return 'close' if current_price <= middle else 'hold'

        if current_price <= lower_band:
            return 'long'
        if current_price >= upper_band:
            return 'short'

        return 'hold'