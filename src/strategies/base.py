import pandas as pd
from typing import Dict, Optional, List, Any, Union
from abc import ABC, abstractmethod
import logging

from ..exceptions import StrategyError, StrategyParameterError, SignalGenerationError

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """策略基类"""

    def __init__(self, name: str = "BaseStrategy"):
        self.name: str = name
        self.params: Dict[str, Any] = {}

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                        entry_price: Optional[float] = None, **kwargs: Any) -> str:
        """
        生成交易信号

        Args:
            data: K线数据
            index: 当前索引
            position_side: 引擎传入的实际持仓状态 ('long', 'short', 'flat')
            entry_price: 引擎传入的开仓价格
            **kwargs: 其他参数

        Returns:
            'long', 'short', 'close', 'hold'
            
        Raises:
            SignalGenerationError: 信号生成失败
            StrategyParameterError: 策略参数错误
        """
        pass

    def set_params(self, **params: Any) -> None:
        """设置策略参数"""
        self.params.update(params)

    def get_params(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self.params.copy()

    def validate_data(self, data: pd.DataFrame, index: int) -> None:
        """
        验证输入数据
        
        Args:
            data: K线数据
            index: 当前索引
            
        Raises:
            SignalGenerationError: 数据验证失败
        """
        if data is None or data.empty:
            raise SignalGenerationError(
                f"策略 {self.name}: 输入数据为空",
                strategy_name=self.name,
                index=index
            )
        
        if index < 0 or index >= len(data):
            raise SignalGenerationError(
                f"策略 {self.name}: 索引超出范围，index={index}, 数据长度={len(data)}",
                strategy_name=self.name,
                index=index
            )
        
        required_columns: List[str] = ['open', 'high', 'low', 'close', 'volume']
        missing_columns: set = set(required_columns) - set(data.columns)
        if missing_columns:
            raise SignalGenerationError(
                f"策略 {self.name}: 数据缺少必需的列: {missing_columns}",
                strategy_name=self.name,
                index=index
            )

    def validate_position_side(self, position_side: str) -> None:
        """
        验证持仓方向
        
        Args:
            position_side: 持仓方向
            
        Raises:
            StrategyParameterError: 参数错误
        """
        valid_sides: List[str] = ['long', 'short', 'flat']
        if position_side not in valid_sides:
            raise StrategyParameterError(
                f"策略 {self.name}: 无效的持仓方向: {position_side}",
                strategy_name=self.name,
                parameter_name="position_side",
                parameter_value=position_side
            )

    def safe_generate_signal(self, data: pd.DataFrame, index: int, position_side: str = 'flat',
                            entry_price: Optional[float] = None, **kwargs: Any) -> str:
        """
        安全的信号生成方法，包含错误处理
        
        Args:
            data: K线数据
            index: 当前索引
            position_side: 引擎传入的实际持仓状态 ('long', 'short', 'flat')
            entry_price: 引擎传入的开仓价格
            **kwargs: 其他参数
            
        Returns:
            'long', 'short', 'close', 'hold'
        """
        try:
            self.validate_data(data, index)
            self.validate_position_side(position_side)
            return self.generate_signal(data, index, position_side, entry_price, **kwargs)
        except Exception as e:
            if isinstance(e, (SignalGenerationError, StrategyParameterError)):
                raise
            logger.error(f"策略 {self.name} 信号生成异常: {e}")
            raise SignalGenerationError(
                f"策略 {self.name}: 信号生成异常 - {str(e)}",
                strategy_name=self.name,
                index=index
            )