import pandas as pd
from typing import Optional
from .base import BaseStrategy
from .indicators import calculate_kdj


class KDJExitAtK50(BaseStrategy):
    """方案A：宽松出场 - 等待K值回升到50以上才平仓"""

    def __init__(self, period: int = 9):
        super().__init__("KDJExitAtK50")
        self.period = period

    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs) -> str:
        if index < self.period + 5:
            return 'hold'

        k, d, j = calculate_kdj(data, self.period)
        current_j = j.iloc[index]
        current_k = k.iloc[index]

        if position_side == 'long':
            return 'close' if current_k >= 50 else 'hold'

        if position_side == 'flat' and current_j < 0:
            return 'long'

        return 'hold'


class KDJExitOverbought(BaseStrategy):
    """方案B：超买出场 - 持到 J 值进入超买区间才卖出"""

    def __init__(self, period: int = 9, overbought_j: float = 80):
        super().__init__("KDJExitOverbought")
        self.period = period
        self.overbought_j = overbought_j

    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs) -> str:
        if index < self.period + 5:
            return 'hold'

        k, d, j = calculate_kdj(data, self.period)
        current_j = j.iloc[index]

        if position_side == 'long':
            return 'close' if current_j >= self.overbought_j else 'hold'

        if position_side == 'flat' and current_j < 0:
            return 'long'

        return 'hold'


class KDJExitWithSLTP(BaseStrategy):
    """方案C：止盈止损护航 - 原版 J 转正出场，叠加止盈止损"""

    def __init__(self, period: int = 9, take_profit_pct: float = 0.04, stop_loss_pct: float = 0.02):
        super().__init__("KDJExitWithSLTP")
        self.period = period
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct

    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs) -> str:
        if index < self.period + 5:
            return 'hold'

        k, d, j = calculate_kdj(data, self.period)
        current_j = j.iloc[index]
        prev_j = j.iloc[index - 1]
        current_price = data['close'].iloc[index]

        if position_side == 'long' and entry_price is not None:
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct <= -self.stop_loss_pct:
                return 'close'
            if pnl_pct >= self.take_profit_pct:
                return 'close'
            if prev_j < 0 and current_j >= 0:
                return 'close'
            return 'hold'

        if position_side == 'flat' and current_j < 0:
            return 'long'

        return 'hold'