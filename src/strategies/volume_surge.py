import pandas as pd
from typing import Optional
from .base import BaseStrategy


class VolumeSurgeStrategy(BaseStrategy):
    """成交量放大策略 - 现货成交量暴增时买入"""

    def __init__(
        self,
        volume_ratio_threshold: float = 5.0,
        entry_mode: str = 'limit',
        limit_price_pct: float = -0.02,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.08,
        trailing_activate: float = 0.03,
        trailing_pct: float = 0.02,
        max_hold_bars: int = 36,
    ):
        super().__init__("VolumeSurgeStrategy")
        self.volume_ratio_threshold = volume_ratio_threshold
        self.entry_mode = entry_mode
        self.limit_price_pct = limit_price_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_activate = trailing_activate
        self.trailing_pct = trailing_pct
        self.max_hold_bars = max_hold_bars
        self.set_params(
            volume_ratio_threshold=volume_ratio_threshold,
            entry_mode=entry_mode,
            limit_price_pct=limit_price_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_activate=trailing_activate,
            trailing_pct=trailing_pct,
            max_hold_bars=max_hold_bars,
        )
        self._entry_bar = 0
        self._highest_price = 0.0

    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs) -> str:
        if index < 1:
            return 'hold'

        current_volume = data['volume'].iloc[index]
        prev_volume = data['volume'].iloc[index - 1]
        current_close = data['close'].iloc[index]
        current_high = data['high'].iloc[index]

        if position_side == 'long' and entry_price is not None:
            if self._highest_price == 0 or current_high > self._highest_price:
                self._highest_price = current_high

            pnl_pct = (current_close - entry_price) / entry_price

            if pnl_pct <= -self.stop_loss_pct:
                self._reset_state()
                return 'close'
            if pnl_pct >= self.take_profit_pct:
                self._reset_state()
                return 'close'

            if self._highest_price >= entry_price * (1 + self.trailing_activate):
                drawdown_from_peak = (self._highest_price - current_close) / self._highest_price
                if drawdown_from_peak >= self.trailing_pct:
                    self._reset_state()
                    return 'close'

            bars_held = index - self._entry_bar
            if bars_held >= self.max_hold_bars:
                self._reset_state()
                return 'close'

            return 'hold'

        if position_side == 'flat' and prev_volume > 0:
            volume_ratio = current_volume / prev_volume
            if volume_ratio >= self.volume_ratio_threshold:
                self._entry_bar = index
                self._highest_price = 0.0
                return 'long'

        return 'hold'

    def _reset_state(self):
        self._highest_price = 0.0

    def calc_limit_price(self, signal_bar_close: float) -> float:
        return signal_bar_close * (1 + self.limit_price_pct)