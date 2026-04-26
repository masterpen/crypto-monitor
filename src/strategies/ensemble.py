import pandas as pd
from typing import Dict, List
from .base import BaseStrategy


class StrategyEnsemble:
    """多策略组合"""

    def __init__(self):
        self.strategies: List[BaseStrategy] = []
        self.weights: Dict[str, float] = {}

    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        self.strategies.append(strategy)
        self.weights[strategy.name] = weight

    def generate_signal(self, data: pd.DataFrame, index: int) -> str:
        if not self.strategies:
            return 'hold'

        votes = {'long': 0, 'short': 0, 'hold': 0, 'close': 0}

        for strategy in self.strategies:
            signal = strategy.generate_signal(data, index)
            weight = self.weights.get(strategy.name, 1.0)
            if signal in votes:
                votes[signal] += weight

        best_signal = 'hold'
        best_weight = votes['hold']
        for sig in ('long', 'short', 'close'):
            if votes[sig] > best_weight:
                best_weight = votes[sig]
                best_signal = sig

        return best_signal

    def get_signal_details(self, data: pd.DataFrame, index: int) -> Dict:
        details = {}
        for strategy in self.strategies:
            signal = strategy.generate_signal(data, index)
            details[strategy.name] = {
                'signal': signal,
                'weight': self.weights.get(strategy.name, 1.0),
                'params': strategy.get_params()
            }
        return details