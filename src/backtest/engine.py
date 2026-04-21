"""
回测引擎模块
支持多策略批量回测、参数优化、性能统计
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Trade:
    """交易记录"""
    timestamp: pd.Timestamp
    side: PositionSide
    price: float
    quantity: float
    commission: float
    slippage: float
    signal_reason: str = ""


@dataclass
class Position:
    """持仓信息"""
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None


@dataclass
class BacktestResult:
    """回测结果"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_winning: float = 0.0
    avg_losing: float = 0.0
    profit_factor: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    
    def summary(self) -> str:
        """生成回测报告摘要"""
        return f"""
=== 回测结果报告 ===
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


class BacktestEngine:
    """回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.commission = commission  # 手续费比例
        self.slippage = slippage      # 滑点比例
        self.position = Position()
        self.capital = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.daily_pnl: List[float] = []
        
    def reset(self):
        """重置回测状态"""
        self.position = Position()
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.daily_pnl = []
    
    def can_open_position(self, price: float, quantity: float) -> bool:
        """检查是否可以开仓"""
        required_capital = price * quantity * (1 + self.commission + self.slippage)
        return self.capital >= required_capital
    
    def open_long(self, timestamp: pd.Timestamp, price: float, quantity: float, reason: str = ""):
        """开多仓"""
        if not self.can_open_position(price, quantity):
            logger.warning(f"资金不足无法开多: 需要 {price * quantity * (1 + self.commission):.2f}, 现有 {self.capital:.2f}")
            return False
        
        # 扣除资金（含手续费和滑点）
        cost = price * quantity * (1 + self.commission + self.slippage)
        self.capital -= cost
        
        # 更新持仓
        if self.position.side == PositionSide.SHORT:
            # 平空开多
            self._close_short(timestamp, price)
        
        self.position = Position(
            side=PositionSide.LONG,
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp
        )
        
        self.trades.append(Trade(
            timestamp=timestamp,
            side=PositionSide.LONG,
            price=price * (1 + self.slippage),
            quantity=quantity,
            commission=price * quantity * self.commission,
            slippage=price * quantity * self.slippage,
            signal_reason=reason
        ))
        
        logger.debug(f"开多: 价格={price}, 数量={quantity}, 剩余资金={self.capital:.2f}")
        return True
    
    def open_short(self, timestamp: pd.Timestamp, price: float, quantity: float, reason: str = ""):
        """开空仓"""
        if not self.can_open_position(price, quantity):
            logger.warning(f"资金不足无法开空")
            return False
        
        cost = price * quantity * (1 + self.commission + self.slippage)
        self.capital -= cost
        
        if self.position.side == PositionSide.LONG:
            # 平多开空
            self._close_long(timestamp, price)
        
        self.position = Position(
            side=PositionSide.SHORT,
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp
        )
        
        self.trades.append(Trade(
            timestamp=timestamp,
            side=PositionSide.SHORT,
            price=price * (1 - self.slippage),
            quantity=quantity,
            commission=price * quantity * self.commission,
            slippage=price * quantity * self.slippage,
            signal_reason=reason
        ))
        
        logger.debug(f"开空: 价格={price}, 数量={quantity}, 剩余资金={self.capital:.2f}")
        return True
    
    def _close_long(self, timestamp: pd.Timestamp, price: float):
        """平多仓"""
        if self.position.side != PositionSide.LONG:
            return
        
        # 收入 = 数量 * 价格 * (1 - 手续费 - 滑点)
        revenue = self.position.quantity * price * (1 - self.commission - self.slippage)
        self.capital += revenue
        self.position = Position()
    
    def _close_short(self, timestamp: pd.Timestamp, price: float):
        """平空仓"""
        if self.position.side != PositionSide.SHORT:
            return
        
        revenue = self.position.quantity * price * (1 - self.commission - self.slippage)
        self.capital += revenue
        self.position = Position()
    
    def close_position(self, timestamp: pd.Timestamp, price: float, reason: str = ""):
        """平仓"""
        if self.position.side == PositionSide.FLAT:
            return
        
        trade_price = price * (1 + self.slippage) if self.position.side == PositionSide.LONG else price * (1 - self.slippage)
        
        if self.position.side == PositionSide.LONG:
            self._close_long(timestamp, price)
        else:
            self._close_short(timestamp, price)
        
        self.trades.append(Trade(
            timestamp=timestamp,
            side=PositionSide.FLAT,
            price=trade_price,
            quantity=self.position.quantity if self.position.quantity > 0 else 0,
            commission=price * self.position.quantity * self.commission,
            slippage=price * self.position.quantity * self.slippage,
            signal_reason=reason
        ))
        
        logger.debug(f"平仓: 价格={price}, 剩余资金={self.capital:.2f}")
    
    def update_equity(self):
        """更新权益曲线"""
        # 计算当前市值
        market_value = self.get_market_value()
        total_equity = self.capital + market_value
        self.equity_curve.append(total_equity)
    
    def get_market_value(self) -> float:
        """获取持仓市值（简化计算）"""
        return 0.0  # 实时价格需要传入
    
    def calculate_metrics(self) -> BacktestResult:
        """计算回测指标"""
        equity = pd.Series(self.equity_curve)
        
        # 交易统计
        trades_df = pd.DataFrame([{
            'timestamp': t.timestamp,
            'side': t.side.value,
            'price': t.price,
            'quantity': t.quantity,
            'commission': t.commission
        } for t in self.trades])
        
        # 筛选出完整的买卖交易对来计算盈亏
        closed_trades = self._analyze_trades()
        
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        
        total_pnl = sum(t['pnl'] for t in closed_trades) if closed_trades else 0
        total_winning = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        total_losing = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        
        # 最大回撤
        cummax = equity.cummax()
        drawdown = equity - cummax
        max_drawdown = drawdown.min()
        max_drawdown_pct = abs(max_drawdown / cummax[drawdown.idxmin()]) if cummax[drawdown.idxmin()] > 0 else 0
        
        # 日收益率
        daily_returns = equity.pct_change().dropna()
        
        # 夏普比率（年化，假设252交易日）
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 索提诺比率
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # 卡尔玛比率
        annual_return = daily_returns.mean() * 252
        calmar_ratio = annual_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        # 盈利/亏损交易统计
        n_trades = len(closed_trades)
        n_winning = len(winning_trades)
        n_losing = len(losing_trades)
        
        return BacktestResult(
            total_trades=n_trades,
            winning_trades=n_winning,
            losing_trades=n_losing,
            win_rate=n_winning / n_trades if n_trades > 0 else 0,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            avg_trade_pnl=total_pnl / n_trades if n_trades > 0 else 0,
            avg_winning=total_winning / n_winning if n_winning > 0 else 0,
            avg_losing=total_losing / n_losing if n_losing > 0 else 0,
            profit_factor=total_winning / total_losing if total_losing > 0 else 0,
            trades=self.trades,
            equity_curve=equity,
            daily_returns=daily_returns
        )
    
    def _analyze_trades(self) -> List[Dict]:
        """分析交易对，计算每笔完整交易的盈亏"""
        # 简化实现：跟踪持仓成本计算盈亏
        closed_trades = []
        entry_price = None
        entry_quantity = None
        entry_side = None
        
        for trade in self.trades:
            if trade.side in [PositionSide.LONG, PositionSide.SHORT]:
                entry_price = trade.price
                entry_quantity = trade.quantity
                entry_side = trade.side
            elif trade.side == PositionSide.FLAT and entry_price is not None:
                if entry_side == PositionSide.LONG:
                    pnl = (trade.price - entry_price) * entry_quantity - trade.commission
                else:
                    pnl = (entry_price - trade.price) * entry_quantity - trade.commission
                
                closed_trades.append({
                    'entry_time': None,  # 可以记录完整时间戳
                    'exit_time': trade.timestamp,
                    'entry_price': entry_price,
                    'exit_price': trade.price,
                    'quantity': entry_quantity,
                    'pnl': pnl,
                    'side': entry_side.value
                })
                
                entry_price = None
                entry_quantity = None
                entry_side = None
        
        return closed_trades

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func: callable,
        **strategy_params
    ) -> BacktestResult:
        """运行回测的便捷函数"""
        engine = BacktestEngine(self.initial_capital, self.commission, self.slippage)
        
        for i in range(len(data)):
            row = data.iloc[i]
            current_price = row['close']
            
            signal = strategy_func(data, i, **strategy_params)
            
            if signal == 'long' and engine.position.side != PositionSide.LONG:
                quantity = (engine.capital * 0.5) / current_price
                if quantity > 0:
                    engine.open_long(row.name, current_price, quantity, reason=f"策略信号:{signal}")
            
            elif signal == 'short' and engine.position.side != PositionSide.SHORT:
                quantity = (engine.capital * 0.5) / current_price
                if quantity > 0:
                    engine.open_short(row.name, current_price, quantity, reason=f"策略信号:{signal}")
            
            elif signal == 'close' and engine.position.side != PositionSide.FLAT:
                engine.close_position(row.name, current_price, reason=f"策略信号:{signal}")
            
            engine.update_equity()
        
        if engine.position.side != PositionSide.FLAT:
            engine.close_position(data.iloc[-1].name, data.iloc[-1]['close'], reason="回测结束")
        
        return engine.calculate_metrics()