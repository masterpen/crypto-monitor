import pandas as pd
from typing import Optional
from .base import BaseStrategy


class ArbitrageStrategy(BaseStrategy):
    """三角套利策略（简化版）"""

    def __init__(self, spread_threshold: float = 0.01, min_profit: float = 0.001):
        super().__init__("ArbitrageStrategy")
        self.spread_threshold = spread_threshold
        self.min_profit = min_profit
        self.set_params(spread_threshold=spread_threshold, min_profit=min_profit)
        self.last_signal = 'hold'

    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs) -> str:
        if len(data.columns) < 3:
            return 'hold'
        if index < 20:
            return 'hold'

        recent = data['close'].iloc[index - 20:index + 1]
        volatility = recent.std() / recent.mean()

        if volatility < self.spread_threshold:
            return 'hold'

        return 'hold'

    @staticmethod
    def calculate_triangular_arb(pair1_price: float, pair2_price: float,
                                  pair3_price: float, direction: str = "forward") -> float:
        if direction == "forward":
            btc_amount = 1 / pair1_price
            eth_amount = btc_amount * pair2_price
            final_usdt = eth_amount * pair3_price
        else:
            eth_amount = 1 / pair3_price
            btc_amount = eth_amount / pair2_price
            final_usdt = btc_amount * pair1_price
        return final_usdt - 1