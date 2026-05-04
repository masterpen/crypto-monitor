"""
回测模块
"""
from .engine import BacktestEngine, BacktestResult
from .vectorized_engine import VectorizedBacktestEngine, VectorizedBacktestResult, run_vectorized_backtest

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'VectorizedBacktestEngine',
    'VectorizedBacktestResult',
    'run_vectorized_backtest',
]