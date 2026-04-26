"""
复合因子 - 多因子组合与信号生成

支持:
- 等权合成
- 加权合成（按IC权重）
- 因子筛选（按IC/IR阈值自动筛选有效因子）
- 生成交易信号
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.factors.base import FactorBase
from src.factors.registry import FactorRegistry


class CompositeFactor(FactorBase):
    """
    复合因子：将多个子因子合成为一个综合因子

    Usage:
        cf = CompositeFactor(
            factor_names=['volume_ratio', 'rsi_divergence'],
            weights=[0.6, 0.4],
            method='weighted'
        )
        composite_values = cf.calculate(data)
    """

    name = "composite"
    description = "多因子组合"
    category = "composite"
    lookback = 0

    def __init__(self, factor_names: Optional[List[str]] = None,
                 factors: Optional[List[FactorBase]] = None,
                 weights: Optional[List[float]] = None,
                 method: str = 'equal',
                 normalize: str = 'zscore',
                 **kwargs):
        """
        Args:
            factor_names: 已注册因子名列表
            factors: 因子实例列表（与 factor_names 二选一）
            weights: 因子权重列表（method='weighted' 时使用）
            method: 合成方法 - 'equal' (等权) / 'weighted' (加权) / 'rank_avg' (排名平均)
            normalize: 合成前的标准化方法 - 'zscore' / 'minmax' / 'rank' / None
        """
        super().__init__(**kwargs)
        self.method = method
        self.normalize_method = normalize

        # 构建因子列表
        if factors is not None:
            self._factors = factors
        elif factor_names is not None:
            self._factors = []
            for fn in factor_names:
                f = FactorRegistry.get(fn)
                if f is None:
                    raise ValueError(f"未注册的因子: {fn}")
                self._factors.append(f)
        else:
            raise ValueError("必须提供 factor_names 或 factors")

        # 权重
        if weights is not None:
            if len(weights) != len(self._factors):
                raise ValueError("权重数量必须与因子数量一致")
            total = sum(abs(w) for w in weights)
            self._weights = [w / total for w in weights] if total > 0 else [1.0 / len(self._factors)] * len(self._factors)
        else:
            self._weights = [1.0 / len(self._factors)] * len(self._factors)

        # 更新 lookback 为所有子因子中最大的
        self.lookback = max((f.lookback for f in self._factors), default=0)

        self.name = f"composite({'+'.join(f.name for f in self._factors)})"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算复合因子值"""
        if len(self._factors) == 0:
            return pd.Series(dtype=float, index=data.index)

        # 1. 计算所有子因子
        sub_values: Dict[str, pd.Series] = {}
        for f in self._factors:
            sub_values[f.name] = f.calculate(data)

        # 2. 标准化
        if self.normalize_method:
            for name in sub_values:
                f_instance = next(f for f in self._factors if f.name == name)
                sub_values[name] = f_instance.normalize(sub_values[name], method=self.normalize_method)

        # 3. 合成
        if self.method == 'equal':
            result = sum(sub_values.values()) / len(sub_values)
        elif self.method == 'weighted':
            result = sum(w * sub_values[f.name] for w, f in zip(self._weights, self._factors))
        elif self.method == 'rank_avg':
            ranks = [sub_values[f.name].rank(pct=True) for f in self._factors]
            result = sum(w * r for w, r in zip(self._weights, ranks))
        else:
            raise ValueError(f"未知的合成方法: {self.method}")

        result.name = self.name
        return result

    def generate_signal(self, data: pd.DataFrame, index: int,
                        position_side: str = 'flat', entry_price: Optional[float] = None,
                        threshold: float = 1.0, **kwargs) -> str:
        """
        基于复合因子值生成交易信号

        Args:
            threshold: 因子值超过此阈值时做多，低于负阈值时做空

        Returns:
            'long' / 'short' / 'close' / 'hold'
        """
        if index < self.lookback:
            return 'hold'

        factor_values = self.calculate(data)
        if index >= len(factor_values):
            return 'hold'

        current_value = factor_values.iloc[index]
        if pd.isna(current_value):
            return 'hold'

        # 做多信号
        if position_side == 'flat' and current_value > threshold:
            return 'long'

        # 做空信号
        if position_side == 'flat' and current_value < -threshold:
            return 'short'

        # 平仓信号
        if position_side == 'long' and current_value < 0:
            return 'close'
        if position_side == 'short' and current_value > 0:
            return 'close'

        return 'hold'
