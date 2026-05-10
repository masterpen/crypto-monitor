"""
信号组合策略
多信号投票 + 动态权重
"""
from typing import Dict, List, Optional, Tuple
from .indicators import (
    KDJSignal, MACDSignal, RSISignal, BollSignal, DemarkSequential,
    Signal, SignalResult
)


class SignalComboStrategy:
    """
    信号组合策略
    
    多信号投票机制:
    - 每个指标独立产生信号 (BUY/SELL/NEUTRAL)
    - 加权投票决定最终方向
    - 多信号共振时信号更强
    """
    
    def __init__(
        self,
        # 信号权重
        weights: Dict[str, float] = None,
        # 共振阈值
        min_votes: int = 2,
        # 止损止盈
        stop_loss: float = 0.03,
        take_profit: float = 0.06,
    ):
        self.weights = weights or {
            '九转': 0.30,    # 最重要
            'KDJ': 0.20,
            'MACD': 0.20,
            'RSI': 0.15,
            'BOLL': 0.15,
        }
        self.min_votes = min_votes
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # 信号生成器
        self.signals = {
            'KDJ': KDJSignal(period=9),
            'MACD': MACDSignal(),
            'RSI': RSISignal(),
            'BOLL': BollSignal(),
            '九转': DemarkSequential(),
        }
        
        # 缓存
        self._cache = {}
    
    def _all_signals(self, data, idx) -> List[SignalResult]:
        """获取所有信号"""
        cache_key = id(data)
        if cache_key not in self._cache:
            self._cache[cache_key] = {}
        
        if idx not in self._cache[cache_key]:
            results = []
            for name, sig in self.signals.items():
                try:
                    results.append(sig.signal_at(data, idx))
                except:
                    pass
            self._cache[cache_key][idx] = results
        
        return self._cache[cache_key][idx]
    
    def _vote(self, data, idx) -> Tuple[float, int, int]:
        """投票计算综合得分"""
        results = self._all_signals(data, idx)
        
        buy_score = 0.0
        sell_score = 0.0
        buy_count = 0
        sell_count = 0
        
        for r in results:
            w = self.weights.get(r.name, 0.1)
            
            if r.signal == Signal.BUY:
                buy_score += w * r.strength
                buy_count += 1
            elif r.signal == Signal.SELL:
                sell_score += w * r.strength
                sell_count += 1
        
        # 归一化到 [-1, 1]
        total_weight = sum(self.weights.get(r.name, 0.1) for r in results)
        net_score = (buy_score - sell_score) / total_weight if total_weight > 0 else 0
        
        return net_score, buy_count, sell_count
    
    def generate_signal(self, data, idx, position_side='flat', entry_price=None, **kw):
        """生成交易信号"""
        if idx < 30:
            return 'hold'
        
        score, buy_count, sell_count = self._vote(data, idx)
        current_price = data['close'].iloc[idx]
        
        if position_side == 'flat':
            # 做多：>=2个买入信号 + 无卖出信号
            if score > 0.2 and buy_count >= self.min_votes and sell_count == 0:
                return 'long'
            
            # 做空：>=2个卖出信号 + 无买入信号
            if score < -0.2 and sell_count >= self.min_votes and buy_count == 0:
                return 'short'
        
        elif position_side == 'long':
            if entry_price:
                if current_price < entry_price * (1 - self.stop_loss):
                    return 'close'
                if current_price > entry_price * (1 + self.take_profit):
                    return 'close'
            # 信号反转退出
            if sell_count >= 2 and buy_count == 0:
                return 'close'
        
        elif position_side == 'short':
            if entry_price:
                if current_price > entry_price * (1 + self.stop_loss):
                    return 'close'
                if current_price < entry_price * (1 - self.take_profit):
                    return 'close'
            if buy_count >= 2 and sell_count == 0:
                return 'close'
        
        return 'hold'
    
    def get_detail(self, data, idx) -> Dict:
        """获取详细信号"""
        results = self._all_signals(data, idx)
        return {
            r.name: {
                'signal': r.signal.name,
                'strength': f"{r.strength:.2f}",
                'value': f"{r.value:.4f}",
            }
            for r in results
        }
