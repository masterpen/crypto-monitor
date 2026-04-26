"""
风控模块
包含仓位管理、止损、风控规则
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """风控配置"""
    max_position_pct: float = 0.2       # 单品种最大仓位比例
    max_total_position_pct: float = 0.6  # 总仓位上限
    max_daily_loss_pct: float = 0.05     # 日度最大亏损比例
    max_drawdown_pct: float = 0.15       # 最大回撤限制
    stop_loss_pct: float = 0.02          # 止损比例
    take_profit_pct: float = 0.05        # 止盈比例
    max_trades_per_day: int = 10         # 日度最大交易次数
    min_trade_interval_sec: int = 60     # 最小交易间隔（秒）


@dataclass
class RiskMetrics:
    """风险指标"""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    last_trade_time: Optional[datetime] = None


class RiskManager:
    """风控管理器"""
    
    def __init__(self, config: RiskConfig = None, initial_capital: float = 10000):
        self.config = config or RiskConfig()
        self.initial_capital = initial_capital
        self.metrics = RiskMetrics(peak_equity=initial_capital, current_equity=initial_capital)
        
        # 持仓信息
        self.positions: Dict[str, Dict] = {}
        
        # 日度统计
        self.daily_stats: Dict[str, Dict] = defaultdict(lambda: {
            'pnl': 0.0,
            'trades': 0,
            'start_equity': 0.0
        })
        
        # 交易记录
        self.trade_log: List[Dict] = []
        
        # 风控规则启用状态
        self.risk_rules_enabled = {
            'position_limit': True,
            'daily_loss_limit': True,
            'drawdown_limit': True,
            'stop_loss': True,
            'trade_frequency': True
        }
    
    def update_equity(self, equity: float):
        """更新权益"""
        self.metrics.current_equity = equity
        
        # 更新峰值
        if equity > self.metrics.peak_equity:
            self.metrics.peak_equity = equity
        
        # 更新回撤
        if self.metrics.peak_equity > 0:
            self.metrics.current_drawdown = self.metrics.peak_equity - equity
            self.metrics.current_drawdown_pct = self.metrics.current_drawdown / self.metrics.peak_equity
    
    def update_position(self, symbol: str, side: str, quantity: float, entry_price: float):
        """更新持仓"""
        if quantity == 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = {
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_time': datetime.now()
            }
    
    def check_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict:
        """
        检查订单是否允许执行
        
        Returns:
            {'allowed': bool, 'reason': str}
        """
        # 检查交易频率
        if self.risk_rules_enabled['trade_frequency']:
            if not self._check_trade_frequency():
                return {'allowed': False, 'reason': '交易频率限制'}
        
        # 检查日度亏损
        if self.risk_rules_enabled['daily_loss_limit']:
            if not self._check_daily_loss_limit():
                return {'allowed': False, 'reason': '日度亏损超限'}
        
        # 检查最大回撤
        if self.risk_rules_enabled['drawdown_limit']:
            if not self._check_drawdown_limit():
                return {'allowed': False, 'reason': '最大回撤超限'}
        
        # 检查仓位限制
        if self.risk_rules_enabled['position_limit']:
            check_result = self._check_position_limit(symbol, quantity, price)
            if not check_result['allowed']:
                return check_result
        
        return {'allowed': True, 'reason': ''}
    
    def _check_trade_frequency(self) -> bool:
        """检查交易频率"""
        if self.metrics.last_trade_time:
            elapsed = (datetime.now() - self.metrics.last_trade_time).total_seconds()
            if elapsed < self.config.min_trade_interval_sec:
                return False
        return True
    
    def _check_daily_loss_limit(self) -> bool:
        """检查日度亏损限制"""
        today = datetime.now().date().isoformat()
        daily_loss = self.daily_stats[today]['pnl']
        daily_loss_pct = daily_loss / self.initial_capital if self.initial_capital > 0 else 0
        
        if daily_loss_pct <= -self.config.max_daily_loss_pct:
            logger.warning(f"日度亏损限制触发: {daily_loss_pct:.2%}")
            return False
        return True
    
    def _check_drawdown_limit(self) -> bool:
        """检查最大回撤限制"""
        if self.metrics.current_drawdown_pct >= self.config.max_drawdown_pct:
            logger.warning(f"最大回撤限制触发: {self.metrics.current_drawdown_pct:.2%}")
            return False
        return True
    
    def _check_position_limit(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict:
        """检查仓位限制"""
        if price is None:
            return {'allowed': True, 'reason': ''}
        
        order_value = quantity * price
        
        # 单品种仓位限制
        max_position_value = self.initial_capital * self.config.max_position_pct
        if order_value > max_position_value:
            return {
                'allowed': False,
                'reason': f'单品种仓位超限: {order_value:.2f} > {max_position_value:.2f}'
            }
        
        # 总仓位限制
        current_total_value = sum(
            pos['quantity'] * pos.get('current_price', pos['entry_price'])
            for pos in self.positions.values()
        )
        new_total_value = current_total_value + order_value
        
        max_total_value = self.initial_capital * self.config.max_total_position_pct
        if new_total_value > max_total_value:
            return {
                'allowed': False,
                'reason': f'总仓位超限: {new_total_value:.2f} > {max_total_value:.2f}'
            }
        
        return {'allowed': True, 'reason': ''}
    
    def check_stop_loss(self, symbol: str, current_price: float) -> Tuple[bool, str]:
        """
        检查是否触发止损
        
        Returns:
            (是否触发, 原因)
        """
        if not self.risk_rules_enabled['stop_loss']:
            return False, ''
        
        position = self.positions.get(symbol)
        if not position or position['side'] == 'flat':
            return False, ''
        
        entry_price = position['entry_price']
        pnl_pct = (current_price - entry_price) / entry_price
        
        if position['side'] == 'long' and pnl_pct <= -self.config.stop_loss_pct:
            return True, f'多头止损: {pnl_pct:.2%}'
        
        if position['side'] == 'short' and pnl_pct >= self.config.stop_loss_pct:
            return True, f'空头止损: {pnl_pct:.2%}'
        
        return False, ''
    
    def check_take_profit(self, symbol: str, current_price: float) -> Tuple[bool, str]:
        """
        检查是否触发止盈
        
        Returns:
            (是否触发, 原因)
        """
        position = self.positions.get(symbol)
        if not position or position['side'] == 'flat':
            return False, ''
        
        entry_price = position['entry_price']
        pnl_pct = (current_price - entry_price) / entry_price
        
        if position['side'] == 'long' and pnl_pct >= self.config.take_profit_pct:
            return True, f'多头止盈: {pnl_pct:.2%}'
        
        if position['side'] == 'short' and pnl_pct <= -self.config.take_profit_pct:
            return True, f'空头止盈: {pnl_pct:.2%}'
        
        return False, ''
    
    def record_trade(self, trade: Dict):
        """记录交易"""
        self.trade_log.append({
            **trade,
            'timestamp': datetime.now()
        })
        
        # 更新日度统计
        today = datetime.now().date().isoformat()
        self.daily_stats[today]['trades'] += 1
        if 'pnl' in trade:
            self.daily_stats[today]['pnl'] += trade['pnl']
        
        self.metrics.daily_trades = self.daily_stats[today]['trades']
        self.metrics.daily_pnl = self.daily_stats[today]['pnl']
        self.metrics.last_trade_time = datetime.now()
    
    def get_risk_report(self) -> Dict:
        """获取风控报告"""
        today = datetime.now().date().isoformat()
        
        return {
            'metrics': {
                'daily_pnl': self.metrics.daily_pnl,
                'daily_trades': self.metrics.daily_trades,
                'current_drawdown': self.metrics.current_drawdown,
                'current_drawdown_pct': self.metrics.current_drawdown_pct,
                'peak_equity': self.metrics.peak_equity,
                'current_equity': self.metrics.current_equity
            },
            'limits': {
                'max_daily_loss_pct': self.config.max_daily_loss_pct,
                'max_drawdown_pct': self.config.max_drawdown_pct,
                'max_position_pct': self.config.max_position_pct,
                'max_total_position_pct': self.config.max_total_position_pct,
                'stop_loss_pct': self.config.stop_loss_pct,
                'take_profit_pct': self.config.take_profit_pct
            },
            'status': {
                'daily_loss_limit': '正常' if self._check_daily_loss_limit() else '触发',
                'drawdown_limit': '正常' if self._check_drawdown_limit() else '触发',
                'trade_frequency': '正常' if self._check_trade_frequency() else '限制中'
            },
            'today_stats': self.daily_stats.get(today, {'pnl': 0, 'trades': 0})
        }


class PositionSizer:
    """仓位计算器"""
    
    def __init__(self, initial_capital: float = 10000, risk_per_trade: float = 0.01):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade  # 每笔交易风险比例
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_amount: Optional[float] = None
    ) -> float:
        """
        计算仓位大小（基于风险）
        
        Args:
            entry_price: 入场价格
            stop_loss_price: 止损价格
            risk_amount: 风险金额（可选，默认使用风险比例）
        
        Returns:
            建议的开仓数量
        """
        if risk_amount is None:
            risk_amount = self.initial_capital * self.risk_per_trade
        
        # 价格差（风险距离）
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
        
        # 数量 = 风险金额 / 每单位价格风险
        quantity = risk_amount / price_risk
        
        return quantity
    
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        计算凯利公式仓位比例
        
        Args:
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
        
        Returns:
            建议的仓位比例
        """
        if avg_loss == 0:
            return 0

        b = avg_win / avg_loss
        if b == 0:
            return 0

        q = 1 - win_rate
        p = win_rate
        
        # f* = (bp - q) / b
        kelly = (b * p - q) / b
        
        # 通常使用半凯利降低风险
        return max(0, kelly / 2)
    
    def fixed_fraction(
        self,
        fraction: float = 0.1
    ) -> float:
        """
        固定比例仓位
        
        Args:
            fraction: 仓位比例
        
        Returns:
            仓位大小（金额）
        """
        return self.initial_capital * fraction


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alerts: List[Dict] = []
        self.alert_callbacks: List[callable] = []
    
    def add_callback(self, callback: callable):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        data: Optional[Dict] = None
    ):
        """
        发送告警
        
        Args:
            level: 'info', 'warning', 'error', 'critical'
            title: 告警标题
            message: 告警内容
            data: 附加数据
        """
        alert = {
            'level': level,
            'title': title,
            'message': message,
            'data': data or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.alerts.append(alert)
        logger.warning(f"[{level.upper()}] {title}: {message}")
        
        # 触发回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调失败: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """获取最近的告警"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff
        ]
    
    def clear_alerts(self):
        """清除告警历史"""
        self.alerts = []