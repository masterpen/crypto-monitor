import pandas as pd
from typing import Optional
from .base import BaseStrategy
from .indicators import calculate_rsi


class RSIStrategy(BaseStrategy):
    """RSI均值回归策略"""

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSIStrategy")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.set_params(period=period, oversold=oversold, overbought=overbought)

    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs) -> str:
        if index < self.period:
            return 'hold'

        close_series = data['close'].iloc[:index + 1]
        rsi = calculate_rsi(close_series, self.period)

        if position_side == 'long':
            return 'close' if rsi >= 40 else 'hold'
        if position_side == 'short':
            return 'close' if rsi <= 60 else 'hold'

        if rsi < self.oversold:
            return 'long'
        if rsi > self.overbought:
            return 'short'

        return 'hold'