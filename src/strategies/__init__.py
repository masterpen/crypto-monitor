"""
交易策略模块
"""
from .base import BaseStrategy
from .trend import TrendStrategy
from .macd import MACDStrategy
from .mean_reversion import MeanReversionStrategy
from .rsi import RSIStrategy
from .momentum import MomentumStrategy
from .arbitrage import ArbitrageStrategy
from .kdj import KDJStrategy
from .kdj_cross import KDJCrossStrategy
from .kdj_exit import KDJExitAtK50, KDJExitOverbought, KDJExitWithSLTP
from .volume_surge import VolumeSurgeStrategy
from .ensemble import StrategyEnsemble
from .optimizer import StrategyOptimizer, OptimizationResult, optimize_strategy
from .comparator import StrategyComparator, StrategyComparisonResult, compare_strategies
from .crypto_trend import (
    CryptoTrendStrategy,
    CryptoTrendWithMTF,
    MultiTimeframeAnalyzer,
    CryptoIndicators,
    VolatilityAdaptiveParams,
    TrendDirection,
    TimeframeSignal,
    MultiTimeframeResult
)
from .fast_indicators import (
    FastIndicators,
    VectorizedCryptoIndicators,
    ParallelBacktester,
    BacktestResult,
    optimize_strategy_parallel
)
from .factor_strategy import (
    FactorStrategy,
    FactorStrategyWithTrend,
    create_factor_strategy,
    create_factor_strategy_with_trend
)

__all__ = [
    'BaseStrategy',
    'TrendStrategy',
    'MACDStrategy',
    'MeanReversionStrategy',
    'RSIStrategy',
    'MomentumStrategy',
    'ArbitrageStrategy',
    'KDJStrategy',
    'KDJCrossStrategy',
    'KDJExitAtK50',
    'KDJExitOverbought',
    'KDJExitWithSLTP',
    'VolumeSurgeStrategy',
    'StrategyEnsemble',
    'StrategyOptimizer',
    'OptimizationResult',
    'optimize_strategy',
    'StrategyComparator',
    'StrategyComparisonResult',
    'compare_strategies',
    'CryptoTrendStrategy',
    'CryptoTrendWithMTF',
    'MultiTimeframeAnalyzer',
    'CryptoIndicators',
    'VolatilityAdaptiveParams',
    'TrendDirection',
    'TimeframeSignal',
    'MultiTimeframeResult',
    'FastIndicators',
    'VectorizedCryptoIndicators',
    'ParallelBacktester',
    'BacktestResult',
    'optimize_strategy_parallel',
]