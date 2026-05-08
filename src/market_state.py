"""
市场状态识别模块
识别涨跌周期，避免未来函数

市场状态:
- BULL: 牛市（上升趋势）
- BEAR: 熊市（下降趋势）
- SIDEWAYS: 震荡市（无明显趋势）

重要: 所有计算只使用历史数据，不使用未来数据
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """市场状态"""
    BULL = "bull"        # 牛市
    BEAR = "bear"        # 熊市
    SIDEWAYS = "sideways"  # 震荡市


@dataclass
class MarketStateResult:
    """市场状态结果"""
    state: MarketState
    strength: float  # 0-1, 趋势强度
    confidence: float  # 0-1, 置信度
    ma_fast: float
    ma_slow: float
    atr_ratio: float


class MarketStateDetector:
    """
    市场状态检测器
    
    检测方法:
    1. 均线交叉判断趋势方向
    2. ADX判断趋势强度
    3. ATR判断波动状态
    
    重要: 所有指标只使用当前及历史数据，不使用未来数据
    """
    
    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        lookback_period: int = 100,
    ):
        """
        Args:
            fast_period: 快速均线周期
            slow_period: 慢速均线周期
            adx_period: ADX计算周期
            adx_threshold: ADX阈值（>此值认为有趋势）
            lookback_period: 回望周期
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.lookback_period = lookback_period
        
        # 缓存
        self._cache = {}
    
    def detect(self, data: pd.DataFrame, index: int) -> MarketStateResult:
        """
        检测指定时间点的市场状态
        
        重要: 只使用 data[:index+1] 的数据，不使用未来数据
        
        Args:
            data: K线数据
            index: 当前索引
            
        Returns:
            市场状态结果
        """
        if index < self.slow_period + self.adx_period:
            return MarketStateResult(
                state=MarketState.SIDEWAYS,
                strength=0.0,
                confidence=0.0,
                ma_fast=0.0,
                ma_slow=0.0,
                atr_ratio=0.0
            )
        
        # 只使用历史数据（到index为止）
        historical_data = data.iloc[:index+1]
        
        # 计算指标
        close = historical_data['close']
        high = historical_data['high']
        low = historical_data['low']
        
        # 1. 均线（只使用历史数据）
        ma_fast = close.ewm(span=self.fast_period, adjust=False).mean().iloc[-1]
        ma_slow = close.ewm(span=self.slow_period, adjust=False).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        # 2. 趋势方向
        if ma_fast > ma_slow and current_price > ma_fast:
            trend_direction = 1  # 上升
        elif ma_fast < ma_slow and current_price < ma_fast:
            trend_direction = -1  # 下降
        else:
            trend_direction = 0  # 震荡
        
        # 3. ADX（趋势强度）
        adx = self._calculate_adx(high, low, close, self.adx_period)
        
        # 4. ATR比率（波动状态）
        atr = self._calculate_atr(high, low, close, self.adx_period)
        atr_ratio = atr / current_price if current_price > 0 else 0
        
        # 5. 综合判断
        if adx > self.adx_threshold:
            # 有趋势
            if trend_direction > 0:
                state = MarketState.BULL
                strength = min(adx / 50, 1.0)  # 归一化
            elif trend_direction < 0:
                state = MarketState.BEAR
                strength = min(adx / 50, 1.0)
            else:
                state = MarketState.SIDEWAYS
                strength = 0.0
        else:
            # 无明显趋势
            state = MarketState.SIDEWAYS
            strength = 0.0
        
        # 6. 置信度（基于多个指标的一致性）
        confidence = self._calculate_confidence(
            trend_direction, adx, atr_ratio, ma_fast, ma_slow, current_price
        )
        
        return MarketStateResult(
            state=state,
            strength=strength,
            confidence=confidence,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            atr_ratio=atr_ratio
        )
    
    def detect_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        批量检测市场状态
        
        重要: 每个时间点只使用该点及之前的数据
        
        Args:
            data: K线数据
            
        Returns:
            市场状态序列
        """
        states = []
        
        for i in range(len(data)):
            if i < self.slow_period + self.adx_period:
                states.append(MarketState.SIDEWAYS.value)
            else:
                result = self.detect(data, i)
                states.append(result.state.value)
        
        return pd.Series(states, index=data.index, name='market_state')
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, 
                       close: pd.Series, period: int) -> float:
        """
        计算ADX（平均趋向指数）
        
        只使用传入的数据（已经是历史数据）
        """
        if len(high) < period + 1:
            return 0.0
        
        # 计算+DM和-DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where(plus_dm > 0, 0.0)
        minus_dm = minus_dm.where(minus_dm > 0, 0.0)
        
        # 计算TR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        # 使用EMA平滑
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
        
        # 计算DX和ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, 
                       close: pd.Series, period: int) -> float:
        """计算ATR"""
        if len(high) < period + 1:
            return 0.0
        
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
    
    def _calculate_confidence(
        self,
        trend_direction: int,
        adx: float,
        atr_ratio: float,
        ma_fast: float,
        ma_slow: float,
        current_price: float
    ) -> float:
        """
        计算置信度
        
        基于多个指标的一致性
        """
        confidence = 0.0
        
        # 1. 均线方向一致性
        if ma_fast > ma_slow:
            if trend_direction > 0:
                confidence += 0.3
        elif ma_fast < ma_slow:
            if trend_direction < 0:
                confidence += 0.3
        
        # 2. ADX强度
        if adx > self.adx_threshold:
            confidence += 0.3
        
        # 3. 价格位置
        if current_price > ma_fast:
            if trend_direction > 0:
                confidence += 0.2
        elif current_price < ma_fast:
            if trend_direction < 0:
                confidence += 0.2
        
        # 4. 波动率适中（不是极端波动）
        if 0.01 < atr_ratio < 0.1:
            confidence += 0.2
        
        return min(confidence, 1.0)


def get_market_state(data: pd.DataFrame, index: int, **kwargs) -> MarketStateResult:
    """
    获取市场状态的便捷函数
    
    Args:
        data: K线数据
        index: 当前索引
        **kwargs: 其他参数
        
    Returns:
        市场状态结果
    """
    detector = MarketStateDetector(**kwargs)
    return detector.detect(data, index)


def is_bull_market(data: pd.DataFrame, index: int, **kwargs) -> bool:
    """是否为牛市"""
    result = get_market_state(data, index, **kwargs)
    return result.state == MarketState.BULL


def is_bear_market(data: pd.DataFrame, index: int, **kwargs) -> bool:
    """是否为熊市"""
    result = get_market_state(data, index, **kwargs)
    return result.state == MarketState.BEAR


def is_sideways_market(data: pd.DataFrame, index: int, **kwargs) -> bool:
    """是否为震荡市"""
    result = get_market_state(data, index, **kwargs)
    return result.state == MarketState.SIDEWAYS
