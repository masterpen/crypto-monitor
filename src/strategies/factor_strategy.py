"""
因子组合策略 - 基于IC/IR评估结果构建
用于前10加密货币回测验证
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from src.factors import FactorRegistry, FactorBase
from src.factors.builtin import register_builtin_factors

register_builtin_factors()

logger = logging.getLogger(__name__)


@dataclass
class FactorSignal:
    """因子信号"""
    factor_name: str
    value: float
    signal: str  # 'long', 'short', 'neutral'
    strength: float  # 0-1


class FactorStrategy:
    """
    因子组合策略
    
    基于评估结果:
    - atr_ratio: |IC|=0.137, IR=0.33 (最佳)
    - volatility: |IC|=0.05, 波动率预测
    - 使用4h/1d周期
    """
    
    def __init__(
        self,
        # 因子权重
        atr_ratio_weight: float = 0.6,
        volatility_weight: float = 0.4,
        # 进入阈值
        entry_threshold: float = 0.5,
        # 退出阈值
        exit_threshold: float = 0.0,
        # 止损止盈
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
        # 仓位管理
        max_position_pct: float = 0.3,
        risk_per_trade: float = 0.02,
        # 因子参数
        atr_period: int = 14,
        volatility_period: int = 20,
        lookback_period: int = 50,
    ):
        self.atr_ratio_weight = atr_ratio_weight
        self.volatility_weight = volatility_weight
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_pct = max_position_pct
        self.risk_per_trade = risk_per_trade
        self.atr_period = atr_period
        self.volatility_period = volatility_period
        self.lookback_period = lookback_period
        
        # 获取因子
        self.atr_factor = FactorRegistry.get('atr_ratio', period=atr_period)
        self.vol_factor = FactorRegistry.get('volatility', period=volatility_period)
        
        # 缓存
        self._factor_cache = {}
    
    def _calculate_factors(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算因子值"""
        cache_key = id(data)
        if cache_key in self._factor_cache:
            return self._factor_cache[cache_key]
        
        factors = {
            'atr_ratio': self.atr_factor.calculate(data),
            'volatility': self.vol_factor.calculate(data),
        }
        
        self._factor_cache[cache_key] = factors
        return factors
    
    def _normalize_factor(self, series: pd.Series, window: int = 50) -> pd.Series:
        """因子标准化（z-score）"""
        mean = series.rolling(window=window, min_periods=20).mean()
        std = series.rolling(window=window, min_periods=20).std()
        return ((series - mean) / std).clip(-3, 3)
    
    def _calculate_composite_score(self, data: pd.DataFrame, index: int) -> float:
        """
        计算综合因子得分
        
        Returns:
            得分范围 [-1, 1]，正值做多，负值做空
        """
        factors = self._calculate_factors(data)
        
        # 标准化因子
        atr_norm = self._normalize_factor(factors['atr_ratio'])
        vol_norm = self._normalize_factor(factors['volatility'])
        
        if index >= len(atr_norm) or pd.isna(atr_norm.iloc[index]):
            return 0.0
        
        # 综合得分
        # atr_ratio: IC正向，值越高越看多
        # volatility: 值越高，市场越活跃
        atr_score = atr_norm.iloc[index]
        vol_score = vol_norm.iloc[index]
        
        # 加权组合
        composite = (
            self.atr_ratio_weight * atr_score +
            self.volatility_weight * vol_score
        )
        
        # 归一化到 [-1, 1]
        return np.clip(composite / 2, -1, 1)
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        index: int,
        position_side: str = 'flat',
        entry_price: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        生成交易信号
        
        Args:
            data: K线数据
            index: 当前索引
            position_side: 当前持仓方向
            entry_price: 入场价格
            
        Returns:
            'long', 'short', 'close', 'hold'
        """
        if index < self.lookback_period:
            return 'hold'
        
        # 计算综合得分
        score = self._calculate_composite_score(data, index)
        
        # 当前价格
        current_price = data['close'].iloc[index]
        
        # 计算ATR用于止损
        atr = self._calculate_atr(data, index)
        
        # ============================================================
        # 进入信号
        # ============================================================
        if position_side == 'flat':
            # 做多信号：得分 > 阈值
            if score > self.entry_threshold:
                return 'long'
            
            # 做空信号：得分 < -阈值
            if score < -self.entry_threshold:
                return 'short'
        
        # ============================================================
        # 退出信号
        # ============================================================
        elif position_side == 'long':
            # 止损
            if entry_price and current_price < entry_price * (1 - self.stop_loss_pct):
                return 'close'
            
            # 止盈
            if entry_price and current_price > entry_price * (1 + self.take_profit_pct):
                return 'close'
            
            # 信号反转退出
            if score < self.exit_threshold:
                return 'close'
            
            # 得分大幅下降退出
            if score < -0.3:
                return 'close'
        
        elif position_side == 'short':
            # 止损
            if entry_price and current_price > entry_price * (1 + self.stop_loss_pct):
                return 'close'
            
            # 止盈
            if entry_price and current_price < entry_price * (1 - self.take_profit_pct):
                return 'close'
            
            # 信号反转退出
            if score > -self.exit_threshold:
                return 'close'
            
            # 得分大幅上升退出
            if score > 0.3:
                return 'close'
        
        return 'hold'
    
    def _calculate_atr(self, data: pd.DataFrame, index: int) -> float:
        """计算ATR"""
        if index < self.atr_period:
            return 0.0
        
        high = data['high'].iloc[max(0, index-self.atr_period):index+1]
        low = data['low'].iloc[max(0, index-self.atr_period):index+1]
        close = data['close'].iloc[max(0, index-self.atr_period):index+1]
        
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        return tr.mean()
    
    def calculate_position_size(
        self,
        equity: float,
        price: float,
        atr: float
    ) -> float:
        """计算仓位大小"""
        if atr <= 0:
            return 0
        
        # 基于ATR的仓位
        risk_amount = equity * self.risk_per_trade
        position_size = risk_amount / (atr * 2)
        
        # 限制最大仓位
        max_position = equity * self.max_position_pct / price
        
        return min(position_size, max_position)
    
    def get_analysis(self, data: pd.DataFrame, index: int) -> Dict:
        """获取分析详情"""
        if index < self.lookback_period:
            return {'score': 0, 'factors': {}}
        
        factors = self._calculate_factors(data)
        atr_norm = self._normalize_factor(factors['atr_ratio'])
        vol_norm = self._normalize_factor(factors['volatility'])
        
        return {
            'score': self._calculate_composite_score(data, index),
            'atr_ratio': factors['atr_ratio'].iloc[index],
            'atr_ratio_norm': atr_norm.iloc[index] if index < len(atr_norm) else 0,
            'volatility': factors['volatility'].iloc[index],
            'volatility_norm': vol_norm.iloc[index] if index < len(vol_norm) else 0,
        }


class FactorStrategyWithTrend(FactorStrategy):
    """
    因子策略 + 趋势确认
    
    增加趋势过滤，提高胜率
    """
    
    def __init__(
        self,
        trend_period: int = 50,
        trend_threshold: float = 0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.trend_period = trend_period
        self.trend_threshold = trend_threshold
    
    def _check_trend(self, data: pd.DataFrame, index: int) -> str:
        """检查趋势方向"""
        if index < self.trend_period:
            return 'neutral'
        
        close = data['close'].iloc[index]
        ma = data['close'].iloc[index-self.trend_period:index+1].mean()
        
        pct_diff = (close - ma) / ma
        
        if pct_diff > self.trend_threshold:
            return 'up'
        elif pct_diff < -self.trend_threshold:
            return 'down'
        
        return 'neutral'
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        index: int,
        position_side: str = 'flat',
        entry_price: Optional[float] = None,
        **kwargs
    ) -> str:
        """生成交易信号（带趋势确认）"""
        if index < self.lookback_period:
            return 'hold'
        
        # 获取基础信号
        base_signal = super().generate_signal(data, index, position_side, entry_price, **kwargs)
        
        # 趋势确认
        trend = self._check_trend(data, index)
        
        # 进入信号需要趋势确认
        if position_side == 'flat':
            if base_signal == 'long' and trend in ['up', 'neutral']:
                return 'long'
            if base_signal == 'short' and trend in ['down', 'neutral']:
                return 'short'
            return 'hold'
        
        # 退出信号不需趋势确认
        return base_signal


def create_factor_strategy(**kwargs) -> FactorStrategy:
    """创建因子策略的便捷函数"""
    return FactorStrategy(**kwargs)


def create_factor_strategy_with_trend(**kwargs) -> FactorStrategyWithTrend:
    """创建带趋势确认的因子策略"""
    return FactorStrategyWithTrend(**kwargs)
