"""
因子基类 - 所有因子必须继承此类并实现 calculate 方法
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


class FactorBase:
    """
    因子基类

    子类必须实现:
        calculate(data: pd.DataFrame) -> pd.Series
            输入: 包含 open/high/low/close/volume 列的 DataFrame
            输出: 与 data 等长的 Series，值为因子值（NaN 表示无效）

    可选实现:
        calculate_batch(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]
            批量计算多币种因子，默认逐个调用 calculate
    """

    # 因子元信息（子类应覆盖）
    name: str = "unknown"
    description: str = ""
    category: str = ""          # 类别: volume / momentum / volatility / mean_reversion / ...
    lookback: int = 0           # 需要的最小回望K线数
    params: Dict[str, Any] = {}  # 因子参数

    def __init__(self, **kwargs):
        # 允许运行时覆盖默认参数
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.params[k] = v

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值

        Args:
            data: K线数据，至少包含 open/high/low/close/volume 列

        Returns:
            pd.Series: 因子值序列，索引与 data 相同
        """
        raise NotImplementedError(f"{self.__class__.__name__} 必须实现 calculate()")

    def calculate_batch(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        批量计算多个币种的因子值（默认逐个调用 calculate，性能敏感因子可重写此方法）

        Args:
            data_dict: {symbol: DataFrame} 币种 -> K线数据

        Returns:
            {symbol: Series} 币种 -> 因子值序列
        """
        results = {}
        for symbol, data in data_dict.items():
            try:
                results[symbol] = self.calculate(data)
            except Exception as e:
                results[symbol] = pd.Series(dtype=float, name=self.name)
        return results

    def validate_data(self, data: pd.DataFrame) -> bool:
        """检查数据是否满足因子计算所需的最小行数和列"""
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(data.columns)):
            return False
        if len(data) < self.lookback + 1:
            return False
        return True

    def normalize(self, factor_values: pd.Series, method: str = 'zscore') -> pd.Series:
        """
        因子标准化

        Args:
            factor_values: 原始因子值
            method: 标准化方法 - 'zscore' / 'minmax' / 'rank'
        """
        if method == 'zscore':
            mean = factor_values.mean()
            std = factor_values.std()
            if std == 0 or pd.isna(std):
                return pd.Series(0, index=factor_values.index)
            return (factor_values - mean) / std
        elif method == 'minmax':
            min_v, max_v = factor_values.min(), factor_values.max()
            if max_v == min_v:
                return pd.Series(0.5, index=factor_values.index)
            return (factor_values - min_v) / (max_v - min_v)
        elif method == 'rank':
            return factor_values.rank(pct=True)
        return factor_values

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name}, category={self.category})>"
