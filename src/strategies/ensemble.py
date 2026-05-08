import pandas as pd
from typing import Dict, List, Any, Optional
import logging

from .base import BaseStrategy
from ..exceptions import StrategyError, SignalGenerationError

logger = logging.getLogger(__name__)


class StrategyEnsemble:
    """多策略组合"""

    def __init__(self):
        self.strategies: List[BaseStrategy] = []
        self.weights: Dict[str, float] = {}

    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0) -> None:
        """
        添加策略
        
        Args:
            strategy: 策略实例
            weight: 权重
        """
        if weight <= 0:
            raise StrategyError(
                f"策略权重必须为正数，当前值: {weight}",
                strategy_name=strategy.name
            )
        
        self.strategies.append(strategy)
        self.weights[strategy.name] = weight

    def generate_signal(self, data: pd.DataFrame, index: int, **kwargs) -> str:
        """
        生成组合信号
        
        Args:
            data: K线数据
            index: 当前索引
            **kwargs: 其他参数（如 position_side, entry_price）
            
        Returns:
            组合信号
        """
        if not self.strategies:
            return 'hold'

        votes: Dict[str, float] = {'long': 0, 'short': 0, 'hold': 0, 'close': 0}

        for strategy in self.strategies:
            try:
                signal: str = strategy.generate_signal(data, index, **kwargs)
                weight: float = self.weights.get(strategy.name, 1.0)
                if signal in votes:
                    votes[signal] += weight
            except Exception as e:
                logger.warning(f"策略 {strategy.name} 信号生成失败: {e}")
                continue

        best_signal: str = 'hold'
        best_weight: float = votes['hold']
        for sig in ('long', 'short', 'close'):
            if votes[sig] > best_weight:
                best_weight = votes[sig]
                best_signal = sig

        return best_signal

    def get_signal_details(self, data: pd.DataFrame, index: int, **kwargs) -> Dict[str, Any]:
        """
        获取信号详情
        
        Args:
            data: K线数据
            index: 当前索引
            **kwargs: 其他参数
            
        Returns:
            信号详情字典
        """
        details: Dict[str, Any] = {}
        for strategy in self.strategies:
            try:
                signal: str = strategy.generate_signal(data, index, **kwargs)
                details[strategy.name] = {
                    'signal': signal,
                    'weight': self.weights.get(strategy.name, 1.0),
                    'params': strategy.get_params()
                }
            except Exception as e:
                logger.warning(f"策略 {strategy.name} 信号生成失败: {e}")
                details[strategy.name] = {
                    'signal': 'error',
                    'weight': self.weights.get(strategy.name, 1.0),
                    'params': strategy.get_params(),
                    'error': str(e)
                }
        return details