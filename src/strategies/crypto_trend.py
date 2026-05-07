"""
加密货币优化趋势策略
针对加密货币市场特点优化：
- 高波动性 -> 放宽止损、自适应参数
- 趋势性强 -> 多时间框架确认
- 24/7交易 -> 考虑不同时间段波动特征
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from .base import BaseStrategy
from ..exceptions import StrategyParameterError, SignalGenerationError

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """趋势方向"""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


@dataclass
class TimeframeSignal:
    """时间框架信号"""
    timeframe: str
    direction: TrendDirection
    strength: float  # 0-1
    ma_fast: float
    ma_slow: float
    atr: float
    volatility: float


@dataclass
class MultiTimeframeResult:
    """多时间框架分析结果"""
    signals: Dict[str, TimeframeSignal]
    consensus: TrendDirection
    consensus_strength: float
    recommended_action: str  # 'long', 'short', 'hold', 'close'


class CryptoIndicators:
    """加密货币技术指标（兼容版，推荐使用 FastIndicators）"""
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """指数移动平均线"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """简单移动平均线"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """平均真实范围"""
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def volatility(close: pd.Series, period: int = 20) -> pd.Series:
        """历史波动率"""
        returns = np.log(close / close.shift(1))
        return returns.rolling(window=period).std() * np.sqrt(365)  # 年化
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """RSI指标"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ADX趋势强度指标"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0.0)
        minus_dm = minus_dm.where(minus_dm > 0, 0.0)
        
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        return dx.ewm(alpha=1/period, min_periods=period).mean()
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """布林带"""
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower
    
    @staticmethod
    def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, 
                   period: int = 10, multiplier: float = 3.0) -> pd.Series:
        """超级趋势指标（向量化版本）"""
        # 使用 FastIndicators 的向量化实现
        from .fast_indicators import FastIndicators
        return FastIndicators.supertrend_vectorized(high, low, close, period, multiplier)


class VolatilityAdaptiveParams:
    """波动率自适应参数"""
    
    # 加密货币波动率分档
    VOLATILITY_REGIMES = {
        'low': {'max_vol': 0.3, 'atr_mult': 2.0, 'position_scale': 1.0},
        'medium': {'max_vol': 0.6, 'atr_mult': 2.5, 'position_scale': 0.7},
        'high': {'max_vol': 1.0, 'atr_mult': 3.0, 'position_scale': 0.5},
        'extreme': {'max_vol': float('inf'), 'atr_mult': 4.0, 'position_scale': 0.3},
    }
    
    # 交易时段波动特征（UTC时间）
    SESSION_VOLATILITY = {
        'asia': {'hours': (0, 8), 'vol_factor': 0.8},      # 亚洲时段
        'europe': {'hours': (8, 16), 'vol_factor': 1.0},    # 欧洲时段
        'us': {'hours': (16, 24), 'vol_factor': 1.2},       # 美国时段
    }
    
    @classmethod
    def get_regime(cls, volatility: float) -> Dict:
        """获取波动率档位"""
        for regime, params in cls.VOLATILITY_REGIMES.items():
            if volatility <= params['max_vol']:
                return {'regime': regime, **params}
        return {'regime': 'extreme', **cls.VOLATILITY_REGIMES['extreme']}
    
    @classmethod
    def get_session_factor(cls, hour: int) -> float:
        """获取时段波动因子"""
        for session, params in cls.SESSION_VOLATILITY.items():
            start, end = params['hours']
            if start <= hour < end:
                return params['vol_factor']
        return 1.0
    
    @classmethod
    def adjust_params(cls, base_params: Dict, volatility: float, hour: int = None) -> Dict:
        """根据波动率调整参数"""
        regime = cls.get_regime(volatility)
        
        adjusted = base_params.copy()
        adjusted['stop_loss_atr_mult'] = regime['atr_mult']
        adjusted['take_profit_atr_mult'] = regime['atr_mult'] * 2
        adjusted['position_scale'] = regime['position_scale']
        
        # 考虑交易时段
        if hour is not None:
            session_factor = cls.get_session_factor(hour)
            adjusted['position_scale'] *= session_factor
        
        return adjusted


class CryptoTrendStrategy(BaseStrategy):
    """加密货币趋势策略"""
    
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        atr_period: int = 14,
        atr_multiplier: float = 2.5,
        use_ema: bool = True,
        use_supertrend: bool = True,
        min_adx: float = 20.0,
        risk_per_trade: float = 0.02,
        max_position_pct: float = 0.3
    ):
        """
        初始化策略
        
        Args:
            fast_period: 快速均线周期
            slow_period: 慢速均线周期
            atr_period: ATR周期
            atr_multiplier: ATR止损倍数
            use_ema: 是否使用EMA（vs SMA）
            use_supertrend: 是否使用超级趋势
            min_adx: 最小ADX阈值（趋势强度）
            risk_per_trade: 每笔交易风险比例
            max_position_pct: 最大仓位比例
        """
        super().__init__("CryptoTrendStrategy")
        
        # 验证参数
        if fast_period <= 0 or slow_period <= 0:
            raise StrategyParameterError("均线周期必须为正数")
        if fast_period >= slow_period:
            raise StrategyParameterError("快速周期必须小于慢速周期")
        
        # 基础参数
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.use_ema = use_ema
        self.use_supertrend = use_supertrend
        self.min_adx = min_adx
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        
        # 指标计算器
        self.indicators = CryptoIndicators()
        self.adaptive_params = VolatilityAdaptiveParams()
        
        # 保存参数
        self.set_params(
            fast_period=fast_period,
            slow_period=slow_period,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            use_ema=use_ema,
            use_supertrend=use_supertrend,
            min_adx=min_adx,
            risk_per_trade=risk_per_trade,
            max_position_pct=max_position_pct
        )
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算所有指标"""
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 均线
        if self.use_ema:
            ma_fast = self.indicators.ema(close, self.fast_period)
            ma_slow = self.indicators.ema(close, self.slow_period)
        else:
            ma_fast = self.indicators.sma(close, self.fast_period)
            ma_slow = self.indicators.sma(close, self.slow_period)
        
        # ATR
        atr = self.indicators.atr(high, low, close, self.atr_period)
        
        # 波动率
        volatility = self.indicators.volatility(close, 20)
        
        # RSI
        rsi = self.indicators.rsi(close, 14)
        
        # ADX
        adx = self.indicators.adx(high, low, close, 14)
        
        # 布林带
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(close, 20, 2.0)
        
        # 超级趋势
        supertrend = self.indicators.supertrend(high, low, close, 10, 3.0)
        
        return {
            'ma_fast': ma_fast,
            'ma_slow': ma_slow,
            'atr': atr,
            'volatility': volatility,
            'rsi': rsi,
            'adx': adx,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'supertrend': supertrend
        }
    
    def _get_trend_direction(self, close: float, ma_fast: float, ma_slow: float, 
                            supertrend: float, adx: float) -> Tuple[TrendDirection, float]:
        """获取趋势方向和强度"""
        # 均线信号
        ma_signal = 0
        if ma_fast > ma_slow:
            ma_signal = 1
        elif ma_fast < ma_slow:
            ma_signal = -1
        
        # 超级趋势信号
        st_signal = 0
        if self.use_supertrend:
            if close > supertrend:
                st_signal = 1
            else:
                st_signal = -1
        
        # 综合信号
        combined = ma_signal
        if self.use_supertrend:
            combined = (ma_signal + st_signal) / 2
        
        # 趋势强度（基于ADX）
        strength = min(adx / 50, 1.0) if adx > self.min_adx else 0
        
        # 确定方向
        if combined >= 0.5 and strength > 0:
            if strength > 0.7:
                direction = TrendDirection.STRONG_UP
            else:
                direction = TrendDirection.UP
        elif combined <= -0.5 and strength > 0:
            if strength > 0.7:
                direction = TrendDirection.STRONG_DOWN
            else:
                direction = TrendDirection.DOWN
        else:
            direction = TrendDirection.NEUTRAL
        
        return direction, strength
    
    def _get_cached_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """获取缓存的指标（避免重复计算）"""
        cache_key = id(data)
        
        if not hasattr(self, '_indicator_cache'):
            self._indicator_cache = {}
        
        if cache_key not in self._indicator_cache:
            self._indicator_cache[cache_key] = self._calculate_indicators(data)
        
        return self._indicator_cache[cache_key]
    
    def generate_signal(self, data: pd.DataFrame, index: int, 
                       position_side: str = 'flat', 
                       entry_price: Optional[float] = None, **kwargs) -> str:
        """
        生成交易信号
        
        Args:
            data: K线数据
            index: 当前索引
            position_side: 当前持仓方向
            entry_price: 入场价格
            
        Returns:
            交易信号: 'long', 'short', 'close', 'hold'
        """
        try:
            # 数据验证
            if index < self.slow_period + 10:
                return 'hold'
            
            # 使用缓存的指标
            indicators = self._get_cached_indicators(data)
            
            close = data['close'].iloc[index]
            ma_fast = indicators['ma_fast'].iloc[index]
            ma_slow = indicators['ma_slow'].iloc[index]
            atr = indicators['atr'].iloc[index]
            volatility = indicators['volatility'].iloc[index]
            adx = indicators['adx'].iloc[index]
            rsi = indicators['rsi'].iloc[index]
            supertrend = indicators['supertrend'].iloc[index]
            
            # 获取趋势方向
            direction, strength = self._get_trend_direction(
                close, ma_fast, ma_slow, supertrend, adx
            )
            
            # 波动率自适应
            vol_regime = self.adaptive_params.get_regime(volatility)
            
            # 止损计算
            stop_distance = atr * vol_regime['atr_mult']
            
            # 信号生成
            if position_side == 'flat':
                # 无持仓，寻找入场机会
                if direction in [TrendDirection.UP, TrendDirection.STRONG_UP]:
                    # RSI过滤超买
                    if rsi < 70:
                        return 'long'
                
                elif direction in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]:
                    # RSI过滤超卖
                    if rsi > 30:
                        return 'short'
            
            elif position_side == 'long':
                # 持多仓，检查出场条件
                if direction in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]:
                    return 'close'
                
                # 止损检查
                if entry_price and close < entry_price - stop_distance:
                    return 'close'
                
                # 趋势减弱
                if direction == TrendDirection.NEUTRAL and strength < 0.3:
                    return 'close'
            
            elif position_side == 'short':
                # 持空仓，检查出场条件
                if direction in [TrendDirection.UP, TrendDirection.STRONG_UP]:
                    return 'close'
                
                # 止损检查
                if entry_price and close > entry_price + stop_distance:
                    return 'close'
                
                # 趋势减弱
                if direction == TrendDirection.NEUTRAL and strength < 0.3:
                    return 'close'
            
            return 'hold'
            
        except Exception as e:
            logger.error(f"信号生成异常: {e}")
            return 'hold'
    
    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss_price: float,
        volatility: float = None
    ) -> float:
        """
        计算仓位大小
        
        Args:
            equity: 当前权益
            entry_price: 入场价格
            stop_loss_price: 止损价格
            volatility: 当前波动率
            
        Returns:
            建议仓位大小
        """
        # 基础风险金额
        risk_amount = equity * self.risk_per_trade
        
        # 止损距离
        stop_distance = abs(entry_price - stop_loss_price)
        
        if stop_distance == 0:
            return 0
        
        # 基础仓位
        position_size = risk_amount / stop_distance
        
        # 波动率调整
        if volatility is not None:
            vol_regime = self.adaptive_params.get_regime(volatility)
            position_size *= vol_regime['position_scale']
        
        # 最大仓位限制
        max_position = equity * self.max_position_pct / entry_price
        
        return min(position_size, max_position)
    
    def get_stop_loss_price(
        self,
        entry_price: float,
        side: str,
        atr: float,
        volatility: float = None
    ) -> float:
        """
        计算止损价格
        
        Args:
            entry_price: 入场价格
            side: 持仓方向
            atr: ATR值
            volatility: 波动率
            
        Returns:
            止损价格
        """
        # 波动率自适应
        if volatility is not None:
            vol_regime = self.adaptive_params.get_regime(volatility)
            multiplier = vol_regime['atr_mult']
        else:
            multiplier = self.atr_multiplier
        
        stop_distance = atr * multiplier
        
        if side == 'long':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance


class MultiTimeframeAnalyzer:
    """多时间框架分析器"""
    
    # 支持的时间框架
    TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']
    
    # 时间框架权重
    TIMEFRAME_WEIGHTS = {
        '5m': 0.1,
        '15m': 0.15,
        '1h': 0.25,
        '4h': 0.3,
        '1d': 0.2
    }
    
    def __init__(self, strategy: CryptoTrendStrategy = None):
        """
        初始化多时间框架分析器
        
        Args:
            strategy: 趋势策略实例
        """
        self.strategy = strategy or CryptoTrendStrategy()
        self.indicators = CryptoIndicators()
    
    def analyze_single_timeframe(self, data: pd.DataFrame, timeframe: str) -> TimeframeSignal:
        """
        分析单个时间框架
        
        Args:
            data: K线数据
            timeframe: 时间框架
            
        Returns:
            时间框架信号
        """
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 计算指标
        if self.strategy.use_ema:
            ma_fast = self.indicators.ema(close, self.strategy.fast_period)
            ma_slow = self.indicators.ema(close, self.strategy.slow_period)
        else:
            ma_fast = self.indicators.sma(close, self.strategy.fast_period)
            ma_slow = self.indicators.sma(close, self.strategy.slow_period)
        
        atr = self.indicators.atr(high, low, close, self.strategy.atr_period)
        volatility = self.indicators.volatility(close, 20)
        adx = self.indicators.adx(high, low, close, 14)
        
        # 超级趋势
        supertrend = self.indicators.supertrend(high, low, close, 10, 3.0)
        
        # 获取当前值
        current_close = close.iloc[-1]
        current_ma_fast = ma_fast.iloc[-1]
        current_ma_slow = ma_slow.iloc[-1]
        current_atr = atr.iloc[-1]
        current_volatility = volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0.3
        current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        current_supertrend = supertrend.iloc[-1]
        
        # 获取趋势方向
        direction, strength = self.strategy._get_trend_direction(
            current_close, current_ma_fast, current_ma_slow, 
            current_supertrend, current_adx
        )
        
        return TimeframeSignal(
            timeframe=timeframe,
            direction=direction,
            strength=strength,
            ma_fast=current_ma_fast,
            ma_slow=current_ma_slow,
            atr=current_atr,
            volatility=current_volatility
        )
    
    def analyze_multiple_timeframes(self, data_dict: Dict[str, pd.DataFrame]) -> MultiTimeframeResult:
        """
        分析多个时间框架
        
        Args:
            data_dict: 不同时间框架的数据字典 {timeframe: DataFrame}
            
        Returns:
            多时间框架分析结果
        """
        signals = {}
        
        # 分析每个时间框架
        for timeframe, data in data_dict.items():
            if len(data) > self.strategy.slow_period + 10:
                signals[timeframe] = self.analyze_single_timeframe(data, timeframe)
        
        if not signals:
            return MultiTimeframeResult(
                signals={},
                consensus=TrendDirection.NEUTRAL,
                consensus_strength=0,
                recommended_action='hold'
            )
        
        # 计算加权共识
        weighted_score = 0
        total_weight = 0
        
        for timeframe, signal in signals.items():
            weight = self.TIMEFRAME_WEIGHTS.get(timeframe, 0.1)
            
            # 方向转换为分数
            direction_score = {
                TrendDirection.STRONG_UP: 2,
                TrendDirection.UP: 1,
                TrendDirection.NEUTRAL: 0,
                TrendDirection.DOWN: -1,
                TrendDirection.STRONG_DOWN: -2
            }.get(signal.direction, 0)
            
            weighted_score += direction_score * weight * signal.strength
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            consensus_score = weighted_score / total_weight
        else:
            consensus_score = 0
        
        # 确定共识方向
        if consensus_score >= 1.0:
            consensus = TrendDirection.STRONG_UP
        elif consensus_score >= 0.3:
            consensus = TrendDirection.UP
        elif consensus_score <= -1.0:
            consensus = TrendDirection.STRONG_DOWN
        elif consensus_score <= -0.3:
            consensus = TrendDirection.DOWN
        else:
            consensus = TrendDirection.NEUTRAL
        
        # 共识强度
        consensus_strength = min(abs(consensus_score) / 2, 1.0)
        
        # 建议操作
        if consensus in [TrendDirection.UP, TrendDirection.STRONG_UP] and consensus_strength > 0.3:
            action = 'long'
        elif consensus in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN] and consensus_strength > 0.3:
            action = 'short'
        else:
            action = 'hold'
        
        return MultiTimeframeResult(
            signals=signals,
            consensus=consensus,
            consensus_strength=consensus_strength,
            recommended_action=action
        )
    
    def resample_to_timeframe(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        重采样数据到指定时间框架
        
        Args:
            data: 原始K线数据（需要有时间索引）
            timeframe: 目标时间框架
            
        Returns:
            重采样后的数据
        """
        # 定义聚合规则
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # 只保留存在的列
        agg_dict = {k: v for k, v in agg_dict.items() if k in data.columns}
        
        # 重采样
        resampled = data.resample(timeframe).agg(agg_dict).dropna()
        
        return resampled


class CryptoTrendWithMTF(CryptoTrendStrategy):
    """带多时间框架分析的加密货币趋势策略"""
    
    def __init__(self, mtf_timeframes: List[str] = None, **kwargs):
        """
        初始化策略
        
        Args:
            mtf_timeframes: 多时间框架列表
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        self.mtf_timeframes = mtf_timeframes or ['5m', '15m', '1h', '4h']
        self.mtf_analyzer = MultiTimeframeAnalyzer(self)
        
        self.set_params(mtf_timeframes=mtf_timeframes)
    
    def generate_signal_with_mtf(
        self,
        data_dict: Dict[str, pd.DataFrame],
        current_index: int,
        position_side: str = 'flat',
        entry_price: Optional[float] = None
    ) -> str:
        """
        使用多时间框架生成信号
        
        Args:
            data_dict: 不同时间框架的数据字典
            current_index: 当前索引
            position_side: 当前持仓方向
            entry_price: 入场价格
            
        Returns:
            交易信号
        """
        # 多时间框架分析
        mtf_result = self.mtf_analyzer.analyze_multiple_timeframes(data_dict)
        
        # 获取主时间框架信号（通常是1h或4h）
        main_tf = '1h' if '1h' in data_dict else list(data_dict.keys())[0]
        main_data = data_dict[main_tf]
        
        # 使用主时间框架计算ATR和波动率
        indicators = self._calculate_indicators(main_data.iloc[:current_index+1])
        atr = indicators['atr'].iloc[-1]
        volatility = indicators['volatility'].iloc[-1]
        
        # 结合MTF结果和止损逻辑
        if position_side == 'flat':
            if mtf_result.recommended_action == 'long' and mtf_result.consensus_strength > 0.3:
                return 'long'
            elif mtf_result.recommended_action == 'short' and mtf_result.consensus_strength > 0.3:
                return 'short'
        
        elif position_side == 'long':
            # 检查止损
            if entry_price and main_data['close'].iloc[current_index] < entry_price - atr * self.atr_multiplier:
                return 'close'
            
            # 趋势反转
            if mtf_result.consensus in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]:
                return 'close'
        
        elif position_side == 'short':
            # 检查止损
            if entry_price and main_data['close'].iloc[current_index] > entry_price + atr * self.atr_multiplier:
                return 'close'
            
            # 趋势反转
            if mtf_result.consensus in [TrendDirection.UP, TrendDirection.STRONG_UP]:
                return 'close'
        
        return 'hold'
    
    def get_analysis_report(self, data_dict: Dict[str, pd.DataFrame]) -> str:
        """
        获取分析报告
        
        Args:
            data_dict: 不同时间框架的数据字典
            
        Returns:
            分析报告文本
        """
        mtf_result = self.mtf_analyzer.analyze_multiple_timeframes(data_dict)
        
        lines = [
            "=== 多时间框架趋势分析报告 ===",
            "",
            "各时间框架信号:",
        ]
        
        for tf, signal in mtf_result.signals.items():
            lines.append(f"  {tf}: {signal.direction.value} (强度: {signal.strength:.2f})")
        
        lines.extend([
            "",
            f"综合趋势: {mtf_result.consensus.value}",
            f"共识强度: {mtf_result.consensus_strength:.2f}",
            f"建议操作: {mtf_result.recommended_action}",
        ])
        
        return "\n".join(lines)