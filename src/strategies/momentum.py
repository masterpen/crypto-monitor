import pandas as pd
from typing import Optional
from .base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """动量策略"""

    def __init__(self, lookback_period: int = 20, threshold: float = 0.02):
        super().__init__("MomentumStrategy")
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.set_params(lookback_period=lookback_period, threshold=threshold)

    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs) -> str:
        if index < self.lookback_period:
            return 'hold'

        current_price = data['close'].iloc[index]
        past_price = data['close'].iloc[index - self.lookback_period]
        momentum = (current_price - past_price) / past_price

        if position_side == 'long':
            return 'close' if momentum < self.threshold * 0.5 else 'hold'
        if position_side == 'short':
            return 'close' if momentum > -self.threshold * 0.5 else 'hold'

        if momentum > self.threshold:
            return 'long'
        if momentum < -self.threshold:
            return 'short'

        return 'hold'