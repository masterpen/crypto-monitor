import pandas as pd
from typing import Dict, Optional
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """策略基类"""

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.params: Dict = {}

    @abstractmethod
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
        """
        pass

    def set_params(self, **params):
        self.params.update(params)

    def get_params(self) -> Dict:
        return self.params.copy()