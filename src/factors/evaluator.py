"""
因子评估器 - 计算因子的预测能力指标

支持:
- IC (Information Coefficient): 因子值与未来收益的秩相关系数
- IR (Information Ratio): IC均值 / IC标准差
- 分层回测: 按因子值分5组，对比各组收益
- 因子衰减分析: 不同预测周期的IC变化
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.factors.base import FactorBase
from src.factors.registry import FactorRegistry


class FactorEvaluator:
    """
    因子评估器

    Usage:
        ev = FactorEvaluator(data, factor_name='volume_ratio')
        ic_series = ev.calc_ic(forward_period=24)
        ic_mean, ic_std, ir = ev.evaluate(forward_period=24)
        layer_result = ev.layer_backtest(n_layers=5, forward_period=24)
    """

    def __init__(self, data: pd.DataFrame, factor_name: Optional[str] = None,
                 factor: Optional[FactorBase] = None, **factor_kwargs):
        """
        Args:
            data: K线数据 (open/high/low/close/volume)
            factor_name: 已注册的因子名（与 factor 二选一）
            factor: 因子实例（与 factor_name 二选一）
        """
        self.data = data
        if factor is not None:
            self.factor = factor
        elif factor_name is not None:
            self.factor = Registry_get(factor_name, **factor_kwargs)
            if self.factor is None:
                raise ValueError(f"未注册的因子: {factor_name}")
        else:
            raise ValueError("必须提供 factor_name 或 factor")

        self._factor_values: Optional[pd.Series] = None

    @property
    def factor_values(self) -> pd.Series:
        """缓存因子值"""
        if self._factor_values is None:
            self._factor_values = self.factor.calculate(self.data)
        return self._factor_values

    def _forward_return(self, period: int) -> pd.Series:
        """计算未来 period 根K线的收益率"""
        return self.data['close'].shift(-period) / self.data['close'] - 1

    def calc_ic(self, forward_period: int = 24, method: str = 'spearman') -> pd.Series:
        """
        滚动IC序列（逐行计算因子值与未来收益的秩相关系数）

        Args:
            forward_period: 预测未来N根K线的收益
            method: 'spearman' (秩相关) 或 'pearson' (线性相关)

        Returns:
            IC序列（每行对应一个时间点的IC值，使用滚动窗口）
        """
        fv = self.factor_values.dropna()
        fr = self._forward_return(forward_period).dropna()

        # 对齐索引
        common_idx = fv.index.intersection(fr.index)
        fv = fv.loc[common_idx]
        fr = fr.loc[common_idx]

        if method == 'spearman':
            # 秩相关 = 对排名后的序列计算 pearson 相关（无需 scipy）
            ic = fv.rank().corr(fr.rank(), method='pearson')
            return pd.Series([ic], index=[fv.index[-1] if len(fv) > 0 else 0])
        else:
            ic = fv.corr(fr, method='pearson')
            return pd.Series([ic], index=[fv.index[-1] if len(fv) > 0 else 0])

    def evaluate(self, forward_period: int = 24, method: str = 'spearman') -> Tuple[float, float, float]:
        """
        因子评估（单币种单次评估）

        Returns:
            (ic_mean, ic_std, ir): IC均值、IC标准差、信息比率
        """
        ic_series = self.calc_ic(forward_period=forward_period, method=method)
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ir = ic_mean / ic_std if ic_std != 0 and not pd.isna(ic_std) else 0.0
        return ic_mean, ic_std, ir

    def evaluate_rolling(self, forward_period: int = 24, window: int = 60,
                         method: str = 'spearman') -> pd.Series:
        """
        滚动窗口IC（需足够数据量，适用于截面评估后的时序IC）

        Args:
            forward_period: 预测周期
            window: 滚动窗口大小
            method: 相关方法

        Returns:
            滚动IC序列
        """
        fv = self.factor_values
        fr = self._forward_return(forward_period)
        common_idx = fv.index.intersection(fr.dropna().index)
        fv, fr = fv.loc[common_idx], fr.loc[common_idx]

        rolling_ics = []
        for i in range(window, len(fv)):
            fv_slice = fv.iloc[i - window:i]
            fr_slice = fr.iloc[i - window:i]
            if method == 'spearman':
                ic = fv_slice.rank().corr(fr_slice.rank(), method='pearson')
            else:
                ic = fv_slice.corr(fr_slice, method='pearson')
            rolling_ics.append({'time': fv.index[i], 'ic': ic})

        if not rolling_ics:
            return pd.Series(dtype=float)
        return pd.Series({r['time']: r['ic'] for r in rolling_ics})

    def layer_backtest(self, n_layers: int = 5, forward_period: int = 24) -> pd.DataFrame:
        """
        因子分层回测（将因子值排序分为n层，对比每层未来收益）

        Args:
            n_layers: 分层数
            forward_period: 预测周期

        Returns:
            DataFrame: 每层的平均因子值、平均未来收益、样本数
        """
        fv = self.factor_values.dropna()
        fr = self._forward_return(forward_period).dropna()
        common_idx = fv.index.intersection(fr.index)
        fv, fr = fv.loc[common_idx], fr.loc[common_idx]

        if len(fv) < n_layers:
            return pd.DataFrame()

        # 按因子值排序分层
        df = pd.DataFrame({'factor': fv, 'forward_return': fr})
        df['layer'] = pd.qcut(df['factor'], n_layers, labels=False, duplicates='drop')

        result = df.groupby('layer').agg(
            avg_factor=('factor', 'mean'),
            avg_return=('forward_return', 'mean'),
            count=('forward_return', 'count'),
        )
        return result

    def decay_analysis(self, periods: List[int] = None) -> pd.DataFrame:
        """
        因子衰减分析：计算不同预测周期的IC

        Args:
            periods: 预测周期列表，默认 [1, 3, 6, 12, 24, 48, 72]

        Returns:
            DataFrame: period / ic / abs_ic
        """
        if periods is None:
            periods = [1, 3, 6, 12, 24, 48, 72]

        rows = []
        for p in periods:
            try:
                ic_mean, ic_std, ir = self.evaluate(forward_period=p)
                rows.append({
                    'period': p,
                    'ic': ic_mean,
                    'abs_ic': abs(ic_mean),
                    'ir': ir,
                })
            except Exception:
                rows.append({'period': p, 'ic': 0, 'abs_ic': 0, 'ir': 0})

        return pd.DataFrame(rows)


def Registry_get(name, **kwargs):
    """辅助函数，避免循环引用"""
    return FactorRegistry.get(name, **kwargs)
