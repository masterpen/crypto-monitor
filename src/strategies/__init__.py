"""
交易策略模块
包含趋势策略、均值回归策略、套利策略
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.params: Dict = {}
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, index: int) -> str:
        """
        生成交易信号
        
        Args:
            data: K线数据
            index: 当前索引
            
        Returns:
            'long', 'short', 'close', 'hold'
        """
        pass
    
    def set_params(self, **params):
        """设置策略参数"""
        self.params.update(params)
    
    def get_params(self) -> Dict:
        """获取策略参数"""
        return self.params.copy()


class TrendStrategy(BaseStrategy):
    """趋势跟踪策略 - 双均线交叉"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__("TrendStrategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.set_params(fast_period=fast_period, slow_period=slow_period)
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> str:
        if index < self.slow_period:
            return 'hold'
        
        # 计算均线
        fast_ma = data['close'].iloc[index - self.fast_period + 1:index + 1].mean()
        slow_ma = data['close'].iloc[index - self.slow_period + 1:index + 1].mean()
        
        # 前一根K线的均线
        prev_fast_ma = data['close'].iloc[index - self.fast_period:index].mean()
        prev_slow_ma = data['close'].iloc[index - self.slow_period:index].mean()
        
        # 金叉：快速均线上穿慢速均线
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
            return 'long'
        
        # 死叉：快速均线下穿慢速均线
        if prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
            return 'short'
        
        return 'hold'


class MACDStrategy(BaseStrategy):
    """MACD趋势策略"""
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ):
        super().__init__("MACDStrategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.set_params(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period
        )
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float, float]:
        """计算MACD"""
        ema_fast = prices.ewm(span=self.fast_period).mean()
        ema_slow = prices.ewm(span=self.slow_period).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal_period).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> str:
        if index < self.slow_period:
            return 'hold'
        
        close_series = data['close'].iloc[:index + 1]
        macd, signal, histogram = self._calculate_macd(close_series)
        
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        
        # MACD柱由负转正 -> 做多
        if prev_hist < 0 and current_hist > 0:
            return 'long'
        
        # MACD柱由正转负 -> 做空
        if prev_hist > 0 and current_hist < 0:
            return 'short'
        
        return 'hold'


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略 - 布林带"""
    
    def __init__(self, period: int = 20, std_multiplier: float = 2.0):
        super().__init__("MeanReversionStrategy")
        self.period = period
        self.std_multiplier = std_multiplier
        self.set_params(period=period, std_multiplier=std_multiplier)
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> str:
        if index < self.period:
            return 'hold'
        
        # 计算布林带
        recent = data['close'].iloc[index - self.period + 1:index + 1]
        middle = recent.mean()
        std = recent.std()
        
        upper_band = middle + self.std_multiplier * std
        lower_band = middle - self.std_multiplier * std
        
        current_price = data['close'].iloc[index]
        
        # 价格触及下轨 -> 做多（均值回归）
        if current_price <= lower_band:
            return 'long'
        
        # 价格触及上轨 -> 做空（均值回归）
        if current_price >= upper_band:
            return 'short'
        
        # 价格回到中轨附近 -> 平仓
        if middle * 0.98 <= current_price <= middle * 1.02:
            return 'close'
        
        return 'hold'


class RSIStrategy(BaseStrategy):
    """RSI均值回归策略"""
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSIStrategy")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.set_params(period=period, oversold=oversold, overbought=overbought)
    
    def _calculate_rsi(self, prices: pd.Series) -> float:
        """计算RSI"""
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.ewm(span=self.period).mean()
        avg_loss = losses.ewm(span=self.period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> str:
        if index < self.period:
            return 'hold'
        
        close_series = data['close'].iloc[:index + 1]
        rsi = self._calculate_rsi(close_series)
        
        # RSI超卖 -> 做多
        if rsi < self.oversold:
            return 'long'
        
        # RSI超买 -> 做空
        if rsi > self.overbought:
            return 'short'
        
        # RSI回归中性区域 -> 平仓
        if self.oversold < rsi < self.overbought:
            return 'close'
        
        return 'hold'


class ArbitrageStrategy(BaseStrategy):
    """三角套利策略（简化版）"""
    
    def __init__(self, spread_threshold: float = 0.01, min_profit: float = 0.001):
        super().__init__("ArbitrageStrategy")
        self.spread_threshold = spread_threshold
        self.min_profit = min_profit
        self.set_params(spread_threshold=spread_threshold, min_profit=min_profit)
        self.last_signal = 'hold'
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> str:
        """
        套利策略需要多个交易对数据，这里简化处理
        实际实现需要监控多个交易对的价差
        """
        if len(data.columns) < 3:
            return 'hold'
        
        # 简化：监控价格波动率
        if index < 20:
            return 'hold'
        
        recent = data['close'].iloc[index - 20:index + 1]
        volatility = recent.std() / recent.mean()
        
        # 波动率低时寻找套利机会
        if volatility < self.spread_threshold:
            return 'hold'  # 低波动，套利空间小
        
        return 'hold'
    
    def calculate_triangular_arb(
        self,
        pair1_price: float,
        pair2_price: float,
        pair3_price: float,
        direction: str = "forward"
    ) -> float:
        """
        计算三角套利收益
        
        Args:
            pair1_price: 交易对1价格 (如 BTC/USDT)
            pair2_price: 交易对2价格 (如 ETH/BTC)
            pair3_price: 交易对3价格 (如 ETH/USDT)
            direction: 套利方向 "forward" 或 "reverse"
        
        Returns:
            理论收益比例
        """
        if direction == "forward":
            # 路径: USDT -> BTC -> ETH -> USDT
            # 假设: 1 USDT -> BTC -> ETH -> ?
            btc_amount = 1 / pair1_price
            eth_amount = btc_amount * pair2_price
            final_usdt = eth_amount * pair3_price
        else:
            # 逆向路径
            eth_amount = 1 / pair3_price
            btc_amount = eth_amount / pair2_price
            final_usdt = btc_amount * pair1_price
        
        return final_usdt - 1  # 收益率


class MomentumStrategy(BaseStrategy):
    """动量策略"""
    
    def __init__(self, lookback_period: int = 20, threshold: float = 0.02):
        super().__init__("MomentumStrategy")
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.set_params(lookback_period=lookback_period, threshold=threshold)
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> str:
        if index < self.lookback_period:
            return 'hold'
        
        current_price = data['close'].iloc[index]
        past_price = data['close'].iloc[index - self.lookback_period]
        
        momentum = (current_price - past_price) / past_price
        
        # 强势上涨 -> 做多
        if momentum > self.threshold:
            return 'long'
        
        # 强势下跌 -> 做空
        if momentum < -self.threshold:
            return 'short'
        
        # 动量减弱 -> 平仓
        if abs(momentum) < self.threshold * 0.5:
            return 'close'
        
        return 'hold'


class StrategyEnsemble:
    """多策略组合"""
    
    def __init__(self):
        self.strategies: list[BaseStrategy] = []
        self.weights: Dict[str, float] = {}
    
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """添加策略"""
        self.strategies.append(strategy)
        self.weights[strategy.name] = weight
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> str:
        """
        基于权重融合多策略信号
        返回信号: 'long', 'short', 'hold'
        """
        if not self.strategies:
            return 'hold'
        
        votes = {'long': 0, 'short': 0, 'hold': 0}
        
        for strategy in self.strategies:
            signal = strategy.generate_signal(data, index)
            weight = self.weights.get(strategy.name, 1.0)
            
            if signal in votes:
                votes[signal] += weight
        
        # 按权重选择信号
        if votes['long'] > votes['short'] and votes['long'] > votes['hold']:
            return 'long'
        if votes['short'] > votes['long'] and votes['short'] > votes['hold']:
            return 'short'
        
        return 'hold'
    
    def get_signal_details(self, data: pd.DataFrame, index: int) -> Dict:
        """获取各策略信号详情"""
        details = {}
        for strategy in self.strategies:
            signal = strategy.generate_signal(data, index)
            details[strategy.name] = {
                'signal': signal,
                'weight': self.weights.get(strategy.name, 1.0),
                'params': strategy.get_params()
            }
        return details


class KDJStrategy(BaseStrategy):
    """KDJ超卖策略 - K值负值买入"""
    
    def __init__(self, period: int = 9, k_period: int = 3, d_period: int = 3):
        super().__init__("KDJStrategy")
        self.period = period
        self.k_period = k_period
        self.d_period = d_period
        self.set_params(period=period, k_period=k_period, d_period=d_period)
    
    def _calculate_kdj(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算KDJ指标"""
        low_min = data['low'].rolling(window=self.period).min()
        high_max = data['high'].rolling(window=self.period).max()
        
        rsv = (data['close'] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)
        
        k = rsv.ewm(com=self.k_period - 1, adjust=False).mean()
        d = k.ewm(com=self.d_period - 1, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return k, d, j
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> str:
        if index < self.period + 5:
            return 'hold'
        
        k, d, j = self._calculate_kdj(data)
        
        current_k = k.iloc[index]
        current_d = d.iloc[index]
        current_j = j.iloc[index]
        
        # 只做多：J值负值 -> 超卖 -> 买入
        if current_j < 0:
            return 'long'
        
        # J值从负转正 -> 卖出
        prev_j = j.iloc[index - 1]
        if prev_j < 0 and current_j >= 0:
            return 'close'
        
        return 'hold'


class KDJCrossStrategy(BaseStrategy):
    """KDJ金叉死叉策略"""
    
    def __init__(self, period: int = 9, overbought: float = 80, oversold: float = 20):
        super().__init__("KDJCrossStrategy")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.set_params(period=period, overbought=overbought, oversold=oversold)
    
    def _calculate_kdj(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        low_min = data['low'].rolling(window=self.period).min()
        high_max = data['high'].rolling(window=self.period).max()
        
        rsv = (data['close'] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)
        
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return k, d, j
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> str:
        if index < self.period + 5:
            return 'hold'
        
        k, d, j = self._calculate_kdj(data)
        
        current_k = k.iloc[index]
        current_d = d.iloc[index]
        prev_k = k.iloc[index - 1]
        prev_d = d.iloc[index - 1]
        
        # 金叉: K上穿D，且在超卖区域
        if prev_k <= prev_d and current_k > current_d and current_d < self.oversold:
            return 'long'
        
        # 死叉: K下穿D，且在超买区域
        if prev_k >= prev_d and current_k < current_d and current_d > self.overbought:
            return 'short'
        
        # K值负值超卖区域 -> 买入
        if current_k < 0:
            return 'long'
        
        # J值负值 -> 买入
        j_val = j.iloc[index]
        if j_val < 0:
            return 'long'
        
        # 超买区域 -> 卖出
        if current_k > 80:
            return 'close'
        
        return 'hold'