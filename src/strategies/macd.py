import pandas as pd
from typing import Optional
from .base import BaseStrategy
from .indicators import calculate_macd


class MACDStrategy(BaseStrategy):
    """MACD趋势策略"""

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACDStrategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.set_params(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)

    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs) -> str:
        if index < self.slow_period:
            return 'hold'

        close_series = data['close'].iloc[:index + 1]
        macd, signal, histogram = calculate_macd(
            close_series, self.fast_period, self.slow_period, self.signal_period
        )

        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]

        bullish = prev_hist < 0 and current_hist > 0
        bearish = prev_hist > 0 and current_hist < 0

        if position_side == 'long' and bearish:
            return 'close'
        if position_side == 'short' and bullish:
            return 'close'

        if position_side == 'flat':
            if bullish:
                return 'long'
            if bearish:
                return 'short'

        return 'hold'