"""
回测引擎模块
支持多策略批量回测、参数优化、性能统计
"""
import math
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
        slippage: float = 0.0005,
        quantity_precision: int = 6  # 数量小数位数，加密货币默认6位
    ):
        self.initial_capital = initial_capital
        self.commission = commission  # 手续费比例
        self.slippage = slippage      # 滑点比例
        self.quantity_precision = quantity_precision
        self.position = Position()
        self.capital = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.daily_pnl: List[float] = []
        self._short_margin: float = 0.0  # 做空冻结的保证金
        self.bankrupted: bool = False     # 是否已爆仓
        
    def reset(self):
        """重置回测状态"""
        self.position = Position()
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.daily_pnl = []
        self._short_margin = 0.0
        self.bankrupted = False
    
    def can_open_position(self, price: float, quantity: float) -> bool:
        """检查是否可以开仓（含滑点和手续费）"""
        fill_price = price * (1 + self.slippage)  # 最不利成交价
        total_cost = fill_price * quantity * (1 + self.commission)
        return self.capital >= total_cost
    
    def open_long(self, timestamp: pd.Timestamp, price: float, quantity: float, reason: str = ""):
        """开多仓"""
        # 滑点先改变成交价，手续费基于实际成交价计算
        fill_price = price * (1 + self.slippage)  # 买入滑点不利
        commission_cost = fill_price * quantity * self.commission
        total_cost = fill_price * quantity + commission_cost

        if not self.can_open_position(price, quantity):
            logger.warning(f"资金不足无法开多: 需要 {total_cost:.2f}, 现有 {self.capital:.2f}")
            return False

        # 先平掉反向持仓（平空会释放保证金）
        if self.position.side == PositionSide.SHORT:
            self._close_short(timestamp, price)

        # 重新检查资金（平仓后可用资金可能变化）
        if self.capital < total_cost:
            logger.warning(f"平仓后资金仍不足无法开多: 需要 {total_cost:.2f}, 现有 {self.capital:.2f}")
            return False

        self.capital -= total_cost

        self.position = Position(
            side=PositionSide.LONG,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=timestamp
        )

        self.trades.append(Trade(
            timestamp=timestamp,
            side=PositionSide.LONG,
            price=fill_price,
            quantity=quantity,
            commission=commission_cost,
            slippage=price * quantity * self.slippage,
            signal_reason=reason
        ))

        logger.debug(f"开多: 成交价={fill_price:.4f}, 数量={quantity}, 剩余资金={self.capital:.2f}")
        return True
    
    def open_short(self, timestamp: pd.Timestamp, price: float, quantity: float, reason: str = ""):
        """
        开空仓（保证金模型）
        做空逻辑：借币卖出，冻结保证金
        - 冻结保证金 = 开仓价值（简化为1:1全额保证金）
        - 卖出收入存入保证金账户
        """
        # 先平掉反向持仓（平多会释放资金）
        if self.position.side == PositionSide.LONG:
            self._close_long(timestamp, price)

        # 滑点先改变成交价，手续费基于实际成交价计算
        fill_price = price * (1 - self.slippage)  # 卖出滑点不利
        commission_cost = fill_price * quantity * self.commission
        margin = fill_price * quantity  # 全额保证金（基于实际成交价）
        total_deduct = margin + commission_cost

        if self.capital < total_deduct:
            logger.warning(f"资金不足无法开空: 需要 {total_deduct:.2f}, 现有 {self.capital:.2f}")
            return False

        # 扣除保证金和手续费
        self.capital -= total_deduct
        self._short_margin = margin

        self.position = Position(
            side=PositionSide.SHORT,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=timestamp
        )

        self.trades.append(Trade(
            timestamp=timestamp,
            side=PositionSide.SHORT,
            price=fill_price,
            quantity=quantity,
            commission=commission_cost,
            slippage=price * quantity * self.slippage,
            signal_reason=reason
        ))

        logger.debug(f"开空: 成交价={fill_price:.4f}, 数量={quantity}, 冻结保证金={margin:.2f}, 剩余资金={self.capital:.2f}")
        return True
    
    def _close_long(self, timestamp: pd.Timestamp, price: float):
        """平多仓"""
        if self.position.side != PositionSide.LONG:
            return
        
        # 滑点先改变成交价，手续费基于实际成交价计算
        fill_price = price * (1 - self.slippage)  # 卖出滑点不利
        commission_cost = fill_price * self.position.quantity * self.commission
        revenue = fill_price * self.position.quantity - commission_cost
        self.capital += revenue
        self.position = Position()
    
    def _close_short(self, timestamp: pd.Timestamp, price: float):
        """
        平空仓（买回归还借币，释放保证金）
        盈亏 = 卖出收入 - 买回成本 = quantity * (entry_price - current_price)
        归还资金 = 保证金 + 盈亏 - 平仓费用
        """
        if self.position.side != PositionSide.SHORT:
            return

        # 滑点先改变成交价，手续费基于实际成交价计算
        fill_price = price * (1 + self.slippage)  # 买入滑点不利
        commission_cost = fill_price * self.position.quantity * self.commission

        # 做空盈亏：价格下跌赚，价格上涨亏
        pnl = self.position.quantity * (self.position.entry_price - fill_price)
        # 归还 = 保证金 + 盈亏 - 手续费
        released = self._short_margin + pnl - commission_cost
        self.capital += released

        self._short_margin = 0.0
        self.position = Position()
    
    def close_position(self, timestamp: pd.Timestamp, price: float, reason: str = ""):
        """平仓"""
        if self.position.side == PositionSide.FLAT:
            return

        # 在平仓前保存持仓信息，因为 _close_long/_close_short 会重置 position
        closing_side = self.position.side
        closing_quantity = self.position.quantity

        # 计算实际成交价和手续费（与 _close_long/_close_short 一致）
        if closing_side == PositionSide.LONG:
            fill_price = price * (1 - self.slippage)  # 卖出滑点不利
        else:
            fill_price = price * (1 + self.slippage)  # 买入滑点不利
        trade_commission = fill_price * closing_quantity * self.commission
        trade_slippage = price * closing_quantity * self.slippage

        if closing_side == PositionSide.LONG:
            self._close_long(timestamp, price)
        else:
            self._close_short(timestamp, price)

        self.trades.append(Trade(
            timestamp=timestamp,
            side=PositionSide.FLAT,
            price=fill_price,
            quantity=closing_quantity,
            commission=trade_commission,
            slippage=trade_slippage,
            signal_reason=reason
        ))

        logger.debug(f"平仓: 成交价={fill_price:.4f}, 剩余资金={self.capital:.2f}")
    
    def update_equity(self, current_price: float = 0.0):
        """更新权益曲线，返回是否爆仓"""
        market_value = self._calc_market_value(current_price)
        total_equity = self.capital + market_value

        # 爆仓检查：总权益 <= 0 或可用资金为负
        if total_equity <= 0 or self.capital < 0:
            self.bankrupted = True
            total_equity = max(total_equity, 0.0)  # 权益不低于0
            logger.warning(f"爆仓! 总权益={total_equity:.2f}, 可用资金={self.capital:.2f}")

        self.equity_curve.append(total_equity)
        return not self.bankrupted
    
    def _calc_market_value(self, current_price: float) -> float:
        """计算持仓市值"""
        if self.position.side == PositionSide.FLAT or current_price <= 0:
            return 0.0
        if self.position.side == PositionSide.LONG:
            return self.position.quantity * current_price
        else:  # SHORT
            # 空仓市值 = 入场卖出金额 - 回购成本
            return self.position.quantity * (2 * self.position.entry_price - current_price)
    
    def get_market_value(self) -> float:
        """获取持仓市值（兼容旧接口）"""
        return 0.0
    
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
        """
        运行回测
        
        防未来函数逻辑：
        - 第 i 根K线收盘时，基于 close 产生信号
        - 信号在第 i+1 根K线的开盘价（open）执行
        - 因此第一根K线（i=0）只产生信号，不执行交易
        
        策略状态同步：
        - strategy_func 签名: (data, index, position_side='flat', **params)
        - position_side 由引擎传入实际持仓状态，避免策略内部状态与引擎脱节
        """
        engine = BacktestEngine(
            self.initial_capital, self.commission, self.slippage,
            self.quantity_precision
        )

        pending_signal = None  # 上一根K线产生的待执行信号

        for i in range(len(data)):
            row = data.iloc[i]
            current_close = row['close']
            current_open = row['open']

            # 先用当前K线收盘价更新权益（反映浮动盈亏）
            if not engine.update_equity(current_close):
                # 爆仓，强制平仓并终止
                engine.close_position(row.name, current_close, reason="爆仓强平")
                break

            # 基于当前K线数据产生信号，传入引擎实际持仓状态
            current_position_side = engine.position.side.value  # 'long', 'short', 'flat'
            entry_price = engine.position.entry_price if engine.position.side != PositionSide.FLAT else None
            signal = strategy_func(data, i, position_side=current_position_side,
                                   entry_price=entry_price, **strategy_params)

            # 执行上一根K线产生的待执行信号（在当前K线开盘价成交）
            if pending_signal is not None and i > 0:
                exec_price = current_open  # 下一根K线开盘价成交
                total_equity = engine.capital + engine._calc_market_value(exec_price)

                if pending_signal == 'long' and engine.position.side != PositionSide.LONG:
                    quantity = (total_equity * 0.5) / exec_price
                    quantity = self._round_quantity(quantity)
                    if quantity > 0:
                        engine.open_long(row.name, exec_price, quantity, reason=f"策略信号:{pending_signal}")

                elif pending_signal == 'short' and engine.position.side != PositionSide.SHORT:
                    quantity = (total_equity * 0.5) / exec_price
                    quantity = self._round_quantity(quantity)
                    if quantity > 0:
                        engine.open_short(row.name, exec_price, quantity, reason=f"策略信号:{pending_signal}")

                elif pending_signal == 'close' and engine.position.side != PositionSide.FLAT:
                    engine.close_position(row.name, exec_price, reason=f"策略信号:{pending_signal}")

                pending_signal = None

            # 记录本根K线产生的信号，等下一根K线开盘执行
            if signal in ('long', 'short', 'close'):
                pending_signal = signal

        # 回测结束时，如果还有待执行信号或持仓，用最后一根K线收盘价处理
        if engine.position.side != PositionSide.FLAT:
            last_price = data.iloc[-1]['close']
            engine.close_position(data.iloc[-1].name, last_price, reason="回测结束")

        return engine.calculate_metrics()

    def _round_quantity(self, quantity: float) -> float:
        """根据交易品种规则截断数量精度（向下取整到指定小数位）"""
        factor = 10 ** self.quantity_precision
        return math.floor(quantity * factor) / factor