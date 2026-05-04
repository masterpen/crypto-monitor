import pandas as pd
import numpy as np
from typing import Optional
import logging

from .base import BaseStrategy
from ..exceptions import StrategyParameterError, SignalGenerationError

logger = logging.getLogger(__name__)


class TrendStrategy(BaseStrategy):
    """趋势跟踪策略 - 双均线交叉"""

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__("TrendStrategy")
        
        # 验证参数
        if fast_period <= 0:
            raise StrategyParameterError(
                f"快速均线周期必须为正数，当前值: {fast_period}",
                strategy_name=self.name,
                parameter_name="fast_period",
                parameter_value=fast_period
            )
        if slow_period <= 0:
            raise StrategyParameterError(
                f"慢速均线周期必须为正数，当前值: {slow_period}",
                strategy_name=self.name,
                parameter_name="slow_period",
                parameter_value=slow_period
            )
        if fast_period >= slow_period:
            raise StrategyParameterError(
                f"快速均线周期必须小于慢速均线周期，当前值: fast={fast_period}, slow={slow_period}",
                strategy_name=self.name,
                parameter_name="fast_period",
                parameter_value=fast_period
            )
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.set_params(fast_period=fast_period, slow_period=slow_period)

    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs) -> str:
        """
        生成交易信号
        
        Args:
            data: K线数据
            index: 当前索引
            position_side: 引擎传入的实际持仓状态 ('long', 'short', 'flat')
            entry_price: 引擎传入的开仓价格
            
        Returns:
            'long', 'short', 'close', 'hold'
            
        Raises:
            SignalGenerationError: 信号生成失败
        """
        try:
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
        except Exception as e:
            if isinstance(e, SignalGenerationError):
                raise
            logger.error(f"趋势策略信号生成异常: {e}")
            raise SignalGenerationError(
                f"趋势策略信号生成异常: {str(e)}",
                strategy_name=self.name,
                index=index
            )