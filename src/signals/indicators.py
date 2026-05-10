"""
技术信号模块
KDJ / MACD / RSI / Bollinger / 神奇九转(DeMark Sequential)
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    BUY = 1
    NEUTRAL = 0
    SELL = -1


@dataclass
class SignalResult:
    name: str
    signal: Signal
    strength: float  # 0-1
    value: float     # 原始指标值


# ═══════════════════════════════════════════════════════════
#  KDJ
# ═══════════════════════════════════════════════════════════
class KDJSignal:
    """KDJ信号
    
    买点: K线上穿D线（金叉）且 J < 20
    卖点: K线下穿D线（死叉）且 J > 80
    """
    
    def __init__(self, period: int = 9):
        self.period = period
        self._cache_kdj = {}
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        low_min = data['low'].rolling(self.period).min()
        high_max = data['high'].rolling(self.period).max()
        rsv = ((data['close'] - low_min) / (high_max - low_min).replace(0, 1e-10)) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        return pd.DataFrame({'K': k, 'D': d, 'J': j}, index=data.index)
    
    def _get_kdj(self, data):
        did = id(data)
        if did not in self._cache_kdj:
            self._cache_kdj[did] = self.calculate(data)
        return self._cache_kdj[did]
    
    def signal_at(self, data: pd.DataFrame, idx: int) -> SignalResult:
        if idx < self.period + 3:
            return SignalResult('KDJ', Signal.NEUTRAL, 0, 50)
        
        kdj = self._get_kdj(data)
        k = kdj['K'].iloc[idx]
        d = kdj['D'].iloc[idx]
        j = kdj['J'].iloc[idx]
        k_prev = kdj['K'].iloc[idx-1]
        d_prev = kdj['D'].iloc[idx-1]
        
        # 金叉 + 超卖区
        if k_prev <= d_prev and k > d and j < 30:
            strength = min((30 - max(j, 0)) / 30, 1.0)
            return SignalResult('KDJ', Signal.BUY, strength, j)
        
        # 死叉 + 超买区
        if k_prev >= d_prev and k < d and j > 70:
            strength = min((min(j, 100) - 70) / 30, 1.0)
            return SignalResult('KDJ', Signal.SELL, strength, j)
        
        return SignalResult('KDJ', Signal.NEUTRAL, 0, j)


# ═══════════════════════════════════════════════════════════
#  MACD
# ═══════════════════════════════════════════════════════════
class MACDSignal:
    """MACD信号
    
    买点: DIF上穿DEA（金叉）在零轴上方
    卖点: DIF下穿DEA（死叉）在零轴下方
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self._cache_macd = {}
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        ema_fast = data['close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal_period, adjust=False).mean()
        hist = 2 * (dif - dea)
        return pd.DataFrame({'DIF': dif, 'DEA': dea, 'HIST': hist}, index=data.index)
    
    def _get_macd(self, data):
        did = id(data)
        if did not in self._cache_macd:
            self._cache_macd[did] = self.calculate(data)
        return self._cache_macd[did]
    
    def signal_at(self, data: pd.DataFrame, idx: int) -> SignalResult:
        if idx < self.slow + 1:
            return SignalResult('MACD', Signal.NEUTRAL, 0, 0)
        
        macd = self._get_macd(data)
        dif = macd['DIF'].iloc[idx]
        dea = macd['DEA'].iloc[idx]
        dif_prev = macd['DIF'].iloc[idx-1]
        dea_prev = macd['DEA'].iloc[idx-1]
        
        # 金叉 + 零轴上方（强势）
        if dif_prev <= dea_prev and dif > dea:
            if dif > 0:
                return SignalResult('MACD', Signal.BUY, 0.8, dif)
            else:
                return SignalResult('MACD', Signal.BUY, 0.4, dif)
        
        # 死叉 + 零轴下方（弱势）
        if dif_prev >= dea_prev and dif < dea:
            if dif < 0:
                return SignalResult('MACD', Signal.SELL, 0.8, dif)
            else:
                return SignalResult('MACD', Signal.SELL, 0.4, dif)
        
        return SignalResult('MACD', Signal.NEUTRAL, 0, dif)


# ═══════════════════════════════════════════════════════════
#  RSI
# ═══════════════════════════════════════════════════════════
class RSISignal:
    """RSI信号
    
    买点: RSI < 30 后回升
    卖点: RSI > 70 后回落
    """
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self._cache_rsi = {}
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(span=self.period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=self.period, adjust=False).mean()
        return 100 - (100 / (1 + gain / loss))
    
    def _get_rsi(self, data):
        did = id(data)
        if did not in self._cache_rsi:
            self._cache_rsi[did] = self.calculate(data)
        return self._cache_rsi[did]
    
    def signal_at(self, data: pd.DataFrame, idx: int) -> SignalResult:
        if idx < self.period + 1:
            return SignalResult('RSI', Signal.NEUTRAL, 0, 50)
        
        rsi = self._get_rsi(data)
        current = rsi.iloc[idx]
        prev = rsi.iloc[idx-1]
        
        # 超卖反弹
        if prev <= self.oversold and current > prev:
            strength = (self.oversold - max(prev, 10)) / (self.oversold - 10)
            return SignalResult('RSI', Signal.BUY, max(strength, 0.3), current)
        
        # 超买回落
        if prev >= self.overbought and current < prev:
            strength = (min(prev, 90) - self.overbought) / (90 - self.overbought)
            return SignalResult('RSI', Signal.SELL, max(strength, 0.3), current)
        
        return SignalResult('RSI', Signal.NEUTRAL, 0, current)


# ═══════════════════════════════════════════════════════════
#  布林带
# ═══════════════════════════════════════════════════════════
class BollSignal:
    """布林带信号
    
    买点: 价格触及下轨 + 回升
    卖点: 价格触及上轨 + 回落
    """
    
    def __init__(self, period: int = 20, std: float = 2.0):
        self.period = period
        self.std = std
        self._cache_boll = {}
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        mid = data['close'].rolling(self.period).mean()
        std = data['close'].rolling(self.period).std()
        return pd.DataFrame({
            'UPPER': mid + self.std * std,
            'MID': mid,
            'LOWER': mid - self.std * std,
        }, index=data.index)
    
    def _get_boll(self, data):
        did = id(data)
        if did not in self._cache_boll:
            self._cache_boll[did] = self.calculate(data)
        return self._cache_boll[did]
    
    def signal_at(self, data: pd.DataFrame, idx: int) -> SignalResult:
        if idx < self.period:
            return SignalResult('BOLL', Signal.NEUTRAL, 0, 0)
        
        boll = self._get_boll(data)
        close = data['close'].iloc[idx]
        prev_close = data['close'].iloc[idx-1]
        upper = boll['UPPER'].iloc[idx]
        lower = boll['LOWER'].iloc[idx]
        mid = boll['MID'].iloc[idx]
        
        # 触及下轨反弹
        if prev_close <= lower and close > prev_close:
            return SignalResult('BOLL', Signal.BUY, 0.7, (close - lower) / (upper - lower))
        
        # 触及上轨回落
        if prev_close >= upper and close < prev_close:
            return SignalResult('BOLL', Signal.SELL, 0.7, (close - lower) / (upper - lower))
        
        return SignalResult('BOLL', Signal.NEUTRAL, 0, (close - lower) / (upper - lower))


# ═══════════════════════════════════════════════════════════
#  神奇九转 (DeMark Sequential)
# ═══════════════════════════════════════════════════════════
class DemarkSequential:
    """
    神奇九转 (DeMark Sequential)
    
    买入结构 (Setup Buy):
    - 连续9根K线，每根收盘价 < 4根前的收盘价
    - 出现9时产生买入信号
    
    卖出结构 (Setup Sell):
    - 连续9根K线，每根收盘价 > 4根前的收盘价
    - 出现9时产生卖出信号
    
    Countdown阶段（可选）:
    - Setup完成后进入Countdown
    - 连续13根K线满足条件
    """
    
    def __init__(self, setup_count: int = 9, countdown_count: int = 13):
        self.setup_count = setup_count
        self.countdown_count = countdown_count
        self._cache_seq = {}
    
    def _get_seq(self, data):
        did = id(data)
        if did not in self._cache_seq:
            self._cache_seq[did] = self.calculate(data)
        return self._cache_seq[did]
    
    def signal_at(self, data: pd.DataFrame, idx: int) -> SignalResult:
        if idx < 8:
            return SignalResult('九转', Signal.NEUTRAL, 0, 0)
        
        seq = self._get_seq(data)
        
        # 买入信号：完成9
        if seq['buy_signal'].iloc[idx] > 0:
            count = seq['buy_signal'].iloc[idx]
            # 第9转到第13转有效
            strength = min((count - 8) / 5, 1.0) if count < 14 else 0
            return SignalResult('九转', Signal.BUY, strength, count)
        
        # 卖出信号：完成9
        if seq['sell_signal'].iloc[idx] > 0:
            count = seq['sell_signal'].iloc[idx]
            strength = min((count - 8) / 5, 1.0) if count < 14 else 0
            return SignalResult('九转', Signal.SELL, strength, count)
        
        # 接近完成
        if seq['setup_buy'].iloc[idx] >= 7:
            return SignalResult('九转', Signal.NEUTRAL, 0.3, seq['setup_buy'].iloc[idx])
        
        return SignalResult('九转', Signal.NEUTRAL, 0, seq['setup_buy'].iloc[idx])
