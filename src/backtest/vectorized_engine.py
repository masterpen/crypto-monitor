"""
向量化回测引擎模块
使用向量化操作提高回测性能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging

from ..exceptions import BacktestError, DataNotFoundError, DataValidationError

logger = logging.getLogger(__name__)


@dataclass
class VectorizedBacktestResult:
    """向量化回测结果"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_trade_pnl: float
    profit_factor: float
    equity_curve: pd.Series
    daily_returns: pd.Series
    signals: pd.Series
    positions: pd.Series
    
    def summary(self) -> str:
        """生成回测报告摘要"""
        return f"""
=== 向量化回测结果报告 ===
总交易次数: {self.total_trades}
盈利交易: {self.winning_trades} ({self.win_rate:.2%})
亏损交易: {self.losing_trades}
总盈亏: {self.total_pnl:.2f}
最大回撤: {self.max_drawdown:.2f} ({self.max_drawdown_pct:.2%})
夏普比率: {self.sharpe_ratio:.2f}
索提诺比率: {self.sortino_ratio:.2f}
卡尔玛比率: {self.calmar_ratio:.2f}
平均交易盈亏: {self.avg_trade_pnl:.2f}
盈亏比: {self.profit_factor:.2f}
"""


class VectorizedBacktestEngine:
    """向量化回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        初始化向量化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点率
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame], pd.Series],
        position_size: float = 0.5
    ) -> VectorizedBacktestResult:
        """
        运行向量化回测
        
        Args:
            data: K线数据
            signal_func: 信号生成函数，接收DataFrame，返回信号Series
            position_size: 仓位比例
            
        Returns:
            回测结果
        """
        # 验证输入数据
        if data is None or data.empty:
            raise DataNotFoundError("回测数据为空")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise DataValidationError(
                f"回测数据缺少必需的列: {missing_columns}",
                validation_errors=[f"缺少列: {col}" for col in missing_columns]
            )
        
        if len(data) < 2:
            raise DataValidationError(
                f"回测数据至少需要2条记录，当前: {len(data)}",
                validation_errors=["数据量不足"]
            )
        
        try:
            # 生成信号
            signals = signal_func(data)
            
            # 验证信号
            if signals is None or len(signals) != len(data):
                raise BacktestError("信号长度与数据长度不匹配")
            
            # 转换信号为仓位
            positions = self._signals_to_positions(signals, position_size)
            
            # 计算收益
            returns = self._calculate_returns(data, positions)
            
            # 计算权益曲线
            equity_curve = self._calculate_equity_curve(returns)
            
            # 计算指标
            metrics = self._calculate_metrics(returns, equity_curve, positions)
            
            return VectorizedBacktestResult(
                total_trades=metrics['total_trades'],
                winning_trades=metrics['winning_trades'],
                losing_trades=metrics['losing_trades'],
                win_rate=metrics['win_rate'],
                total_pnl=metrics['total_pnl'],
                max_drawdown=metrics['max_drawdown'],
                max_drawdown_pct=metrics['max_drawdown_pct'],
                sharpe_ratio=metrics['sharpe_ratio'],
                sortino_ratio=metrics['sortino_ratio'],
                calmar_ratio=metrics['calmar_ratio'],
                avg_trade_pnl=metrics['avg_trade_pnl'],
                profit_factor=metrics['profit_factor'],
                equity_curve=equity_curve,
                daily_returns=returns,
                signals=signals,
                positions=positions
            )
        except Exception as e:
            if isinstance(e, (DataNotFoundError, DataValidationError, BacktestError)):
                raise
            logger.error(f"向量化回测执行异常: {e}")
            raise BacktestError(
                f"向量化回测执行异常: {str(e)}",
                details={"data_length": len(data)}
            )
    
    def _signals_to_positions(self, signals: pd.Series, position_size: float) -> pd.Series:
        """
        将信号转换为仓位
        
        Args:
            signals: 信号序列
            position_size: 仓位比例
            
        Returns:
            仓位序列
        """
        # 初始化仓位
        positions = pd.Series(0.0, index=signals.index)
        
        # 当前仓位状态
        current_position = 0.0
        
        for i in range(len(signals)):
            signal = signals.iloc[i]
            
            if signal == 'long':
                current_position = position_size
            elif signal == 'short':
                current_position = -position_size
            elif signal == 'close':
                current_position = 0.0
            # 'hold' 信号保持当前仓位
            
            positions.iloc[i] = current_position
        
        return positions
    
    def _calculate_returns(self, data: pd.DataFrame, positions: pd.Series) -> pd.Series:
        """
        计算收益
        
        Args:
            data: K线数据
            positions: 仓位序列
            
        Returns:
            收益序列
        """
        # 计算价格变化率
        price_returns = data['close'].pct_change()
        
        # 计算策略收益（考虑仓位）
        strategy_returns = positions.shift(1) * price_returns
        
        # 考虑手续费和滑点
        # 计算仓位变化
        position_changes = positions.diff().abs()
        
        # 计算交易成本
        transaction_costs = position_changes * (self.commission + self.slippage)
        
        # 减去交易成本
        strategy_returns = strategy_returns - transaction_costs
        
        # 填充NaN值
        strategy_returns = strategy_returns.fillna(0)
        
        return strategy_returns
    
    def _calculate_equity_curve(self, returns: pd.Series) -> pd.Series:
        """
        计算权益曲线
        
        Args:
            returns: 收益序列
            
        Returns:
            权益曲线
        """
        # 计算累积收益
        cumulative_returns = (1 + returns).cumprod()
        
        # 计算权益曲线
        equity_curve = self.initial_capital * cumulative_returns
        
        return equity_curve
    
    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        positions: pd.Series
    ) -> Dict[str, Any]:
        """
        计算回测指标
        
        Args:
            returns: 收益序列
            equity_curve: 权益曲线
            positions: 仓位序列
            
        Returns:
            指标字典
        """
        # 计算交易次数
        position_changes = positions.diff().abs()
        total_trades = int((position_changes > 0).sum())
        
        # 计算盈利和亏损交易
        winning_trades = int((returns > 0).sum())
        losing_trades = int((returns < 0).sum())
        
        # 计算胜率
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 计算总盈亏
        total_pnl = equity_curve.iloc[-1] - self.initial_capital
        
        # 计算最大回撤
        cummax = equity_curve.cummax()
        drawdown = equity_curve - cummax
        max_drawdown = drawdown.min()
        max_drawdown_pct = abs(max_drawdown / cummax[drawdown.idxmin()]) if cummax[drawdown.idxmin()] > 0 else 0
        
        # 计算日收益率
        daily_returns = returns
        
        # 计算夏普比率（年化，假设252交易日）
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 计算索提诺比率
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # 计算卡尔玛比率
        annual_return = daily_returns.mean() * 252
        calmar_ratio = annual_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        # 计算平均交易盈亏
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        # 计算盈亏比
        total_winning = returns[returns > 0].sum()
        total_losing = abs(returns[returns < 0].sum())
        profit_factor = total_winning / total_losing if total_losing > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'avg_trade_pnl': avg_trade_pnl,
            'profit_factor': profit_factor
        }


def run_vectorized_backtest(
    data: pd.DataFrame,
    signal_func: Callable[[pd.DataFrame], pd.Series],
    initial_capital: float = 10000,
    commission: float = 0.001,
    slippage: float = 0.0005,
    position_size: float = 0.5
) -> VectorizedBacktestResult:
    """
    运行向量化回测的便捷函数
    
    Args:
        data: K线数据
        signal_func: 信号生成函数
        initial_capital: 初始资金
        commission: 手续费率
        slippage: 滑点率
        position_size: 仓位比例
        
    Returns:
        回测结果
    """
    engine = VectorizedBacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage
    )
    
    return engine.run_backtest(data, signal_func, position_size)