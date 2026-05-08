"""
默认回测配置
避免每次都要指定参数
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BacktestConfig:
    """回测配置"""
    # 基础配置
    initial_capital: float = 10000
    commission: float = 0.001
    slippage: float = 0.0005
    days: int = 90
    
    # 默认币种
    default_symbols: List[str] = None
    
    def __post_init__(self):
        if self.default_symbols is None:
            self.default_symbols = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
                'XRPUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT'
            ]


@dataclass
class FactorStrategyConfig:
    """因子策略配置"""
    # 因子权重（基于IC）
    atr_ratio_ic: float = 0.10
    volatility_ic: float = -0.12
    
    # 进入/退出阈值
    entry_threshold: float = 0.5
    exit_threshold: float = 0.2
    
    # 止损止盈
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    
    # 仓位管理
    max_position_pct: float = 0.3
    risk_per_trade: float = 0.02
    
    # 因子参数
    atr_period: int = 14
    volatility_period: int = 20
    lookback_period: int = 50
    
    # 市场状态过滤
    use_market_state: bool = True
    bull_only: bool = False  # 只在牛市交易
    bear_only: bool = False  # 只在熊市交易


# 默认配置实例
DEFAULT_BACKTEST_CONFIG = BacktestConfig()
DEFAULT_FACTOR_CONFIG = FactorStrategyConfig()
