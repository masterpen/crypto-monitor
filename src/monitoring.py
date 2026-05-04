"""
监控和报告模块
提供实时监控、性能报告和告警功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

from .exceptions import QuantTradingError

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    total_pnl: float
    daily_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    avg_trade_pnl: float
    current_equity: float
    peak_equity: float


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    condition: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    threshold: float
    metric: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    enabled: bool = True
    message_template: str = ""


@dataclass
class Alert:
    """告警"""
    alert_id: str
    rule_id: str
    timestamp: datetime
    severity: str
    title: str
    message: str
    metric_value: float
    threshold: float
    acknowledged: bool = False


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self, initial_capital: float = 10000):
        """
        初始化性能跟踪器
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.equity_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_stats: Dict[str, Dict[str, float]] = {}
        
        # 当前状态
        self.current_equity: float = initial_capital
        self.peak_equity: float = initial_capital
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_pnl: float = 0.0
        self.daily_pnl: float = 0.0
    
    def update_equity(self, equity: float, timestamp: Optional[datetime] = None) -> None:
        """
        更新权益
        
        Args:
            equity: 当前权益
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.current_equity = equity
        
        # 更新峰值
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # 记录权益历史
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': equity,
            'peak_equity': self.peak_equity,
            'drawdown': self.peak_equity - equity,
            'drawdown_pct': (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        })
        
        # 更新日度统计
        date_str = timestamp.date().isoformat()
        if date_str not in self.daily_stats:
            self.daily_stats[date_str] = {
                'start_equity': equity,
                'end_equity': equity,
                'high_equity': equity,
                'low_equity': equity,
                'pnl': 0.0,
                'trades': 0
            }
        
        daily = self.daily_stats[date_str]
        daily['end_equity'] = equity
        daily['high_equity'] = max(daily['high_equity'], equity)
        daily['low_equity'] = min(daily['low_equity'], equity)
        daily['pnl'] = equity - daily['start_equity']
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        commission: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        记录交易
        
        Args:
            symbol: 交易对
            side: 交易方向
            quantity: 数量
            entry_price: 入场价格
            exit_price: 出场价格
            pnl: 盈亏
            commission: 手续费
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 记录交易历史
        self.trade_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'commission': commission,
            'net_pnl': pnl - commission
        })
        
        # 更新统计
        self.total_trades += 1
        self.total_pnl += pnl - commission
        
        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1
        
        # 更新日度统计
        date_str = timestamp.date().isoformat()
        if date_str in self.daily_stats:
            self.daily_stats[date_str]['trades'] += 1
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """
        获取当前性能指标
        
        Returns:
            性能指标
        """
        # 计算胜率
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # 计算最大回撤
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        
        if self.equity_history:
            equity_series = pd.Series([h['equity'] for h in self.equity_history])
            cummax = equity_series.cummax()
            drawdown = equity_series - cummax
            max_drawdown = drawdown.min()
            max_drawdown_pct = abs(max_drawdown / cummax[drawdown.idxmin()]) if cummax[drawdown.idxmin()] > 0 else 0
        
        # 计算夏普比率（简化计算）
        if self.equity_history and len(self.equity_history) > 1:
            equity_series = pd.Series([h['equity'] for h in self.equity_history])
            returns = equity_series.pct_change().dropna()
            if returns.std() > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # 计算盈亏比
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        losing_trades = [t for t in self.trade_history if t['pnl'] < 0]
        
        total_winning = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        total_losing = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        
        profit_factor = total_winning / total_losing if total_losing > 0 else 0
        
        # 计算平均交易盈亏
        avg_trade_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            total_pnl=self.total_pnl,
            daily_pnl=self.daily_pnl,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=self.total_trades,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            current_equity=self.current_equity,
            peak_equity=self.peak_equity
        )
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        获取权益曲线
        
        Returns:
            权益曲线DataFrame
        """
        if not self.equity_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.equity_history).set_index('timestamp')
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        获取交易历史
        
        Returns:
            交易历史DataFrame
        """
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history).set_index('timestamp')
    
    def get_daily_stats(self) -> pd.DataFrame:
        """
        获取日度统计
        
        Returns:
            日度统计DataFrame
        """
        if not self.daily_stats:
            return pd.DataFrame()
        
        return pd.DataFrame(self.daily_stats).T


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self.callbacks: List[Callable] = []
        self.alert_counter: int = 0
    
    def add_rule(self, rule: AlertRule) -> None:
        """
        添加告警规则
        
        Args:
            rule: 告警规则
        """
        self.rules[rule.rule_id] = rule
        logger.info(f"添加告警规则: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        移除告警规则
        
        Args:
            rule_id: 规则ID
            
        Returns:
            是否成功
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"移除告警规则: {rule_id}")
            return True
        return False
    
    def add_callback(self, callback: Callable) -> None:
        """
        添加告警回调
        
        Args:
            callback: 回调函数
        """
        self.callbacks.append(callback)
    
    def check_alerts(self, metrics: PerformanceMetrics) -> List[Alert]:
        """
        检查告警
        
        Args:
            metrics: 性能指标
            
        Returns:
            触发的告警列表
        """
        triggered_alerts = []
        
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # 获取指标值
            metric_value = self._get_metric_value(metrics, rule.metric)
            
            if metric_value is None:
                continue
            
            # 检查条件
            triggered = False
            
            if rule.condition == 'gt':
                triggered = metric_value > rule.threshold
            elif rule.condition == 'lt':
                triggered = metric_value < rule.threshold
            elif rule.condition == 'eq':
                triggered = metric_value == rule.threshold
            elif rule.condition == 'gte':
                triggered = metric_value >= rule.threshold
            elif rule.condition == 'lte':
                triggered = metric_value <= rule.threshold
            
            if triggered:
                # 创建告警
                self.alert_counter += 1
                alert_id = f"ALERT_{self.alert_counter:06d}"
                
                # 生成消息
                if rule.message_template:
                    message = rule.message_template.format(
                        metric=rule.metric,
                        value=metric_value,
                        threshold=rule.threshold
                    )
                else:
                    message = f"{rule.metric} = {metric_value:.4f} (阈值: {rule.threshold:.4f})"
                
                alert = Alert(
                    alert_id=alert_id,
                    rule_id=rule_id,
                    timestamp=datetime.now(),
                    severity=rule.severity,
                    title=rule.name,
                    message=message,
                    metric_value=metric_value,
                    threshold=rule.threshold
                )
                
                self.alerts.append(alert)
                triggered_alerts.append(alert)
                
                # 触发回调
                for callback in self.callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"告警回调失败: {e}")
                
                logger.warning(f"告警触发: {rule.name} - {message}")
        
        return triggered_alerts
    
    def _get_metric_value(self, metrics: PerformanceMetrics, metric_name: str) -> Optional[float]:
        """
        获取指标值
        
        Args:
            metrics: 性能指标
            metric_name: 指标名称
            
        Returns:
            指标值
        """
        metric_map = {
            'total_pnl': metrics.total_pnl,
            'daily_pnl': metrics.daily_pnl,
            'win_rate': metrics.win_rate,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'max_drawdown_pct': metrics.max_drawdown_pct,
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'profit_factor': metrics.profit_factor,
            'avg_trade_pnl': metrics.avg_trade_pnl,
            'current_equity': metrics.current_equity,
            'peak_equity': metrics.peak_equity
        }
        
        return metric_map.get(metric_name)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        确认告警
        
        Args:
            alert_id: 告警ID
            
        Returns:
            是否成功
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """
        获取活跃告警
        
        Returns:
            活跃告警列表
        """
        return [alert for alert in self.alerts if not alert.acknowledged]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """
        获取告警历史
        
        Args:
            hours: 时间范围（小时）
            
        Returns:
            告警历史
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff]


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, performance_tracker: PerformanceTracker):
        """
        初始化报告生成器
        
        Args:
            performance_tracker: 性能跟踪器
        """
        self.performance_tracker = performance_tracker
    
    def generate_daily_report(self, date: Optional[str] = None) -> str:
        """
        生成日度报告
        
        Args:
            date: 日期（格式：YYYY-MM-DD）
            
        Returns:
            报告内容
        """
        if date is None:
            date = datetime.now().date().isoformat()
        
        # 获取日度统计
        daily_stats = self.performance_tracker.get_daily_stats()
        
        if date not in daily_stats.index:
            return f"没有 {date} 的数据"
        
        stats = daily_stats.loc[date]
        
        # 获取当前指标
        metrics = self.performance_tracker.get_current_metrics()
        
        lines = [
            f"# 量化交易系统日度报告 - {date}",
            "",
            "## 基本信息",
            f"- 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 初始资金: {self.performance_tracker.initial_capital:,.2f}",
            "",
            "## 日度表现",
            f"- 日度盈亏: {stats['pnl']:,.2f}",
            f"- 交易次数: {int(stats['trades'])}",
            f"- 最高权益: {stats['high_equity']:,.2f}",
            f"- 最低权益: {stats['low_equity']:,.2f}",
            "",
            "## 累计表现",
            f"- 总盈亏: {metrics.total_pnl:,.2f}",
            f"- 总交易次数: {metrics.total_trades}",
            f"- 胜率: {metrics.win_rate:.2%}",
            f"- 盈亏比: {metrics.profit_factor:.2f}",
            f"- 夏普比率: {metrics.sharpe_ratio:.2f}",
            "",
            "## 风险指标",
            f"- 最大回撤: {metrics.max_drawdown:,.2f}",
            f"- 最大回撤比例: {metrics.max_drawdown_pct:.2%}",
            f"- 当前权益: {metrics.current_equity:,.2f}",
            f"- 峰值权益: {metrics.peak_equity:,.2f}",
        ]
        
        return "\n".join(lines)
    
    def generate_weekly_report(self) -> str:
        """
        生成周度报告
        
        Returns:
            报告内容
        """
        # 获取当前指标
        metrics = self.performance_tracker.get_current_metrics()
        
        # 获取交易历史
        trade_history = self.performance_tracker.get_trade_history()
        
        # 计算周度统计
        if not trade_history.empty:
            # 获取最近7天的交易
            week_ago = datetime.now() - timedelta(days=7)
            weekly_trades = trade_history[trade_history.index >= week_ago]
            
            weekly_pnl = weekly_trades['net_pnl'].sum() if not weekly_trades.empty else 0
            weekly_trades_count = len(weekly_trades)
            weekly_win_rate = len(weekly_trades[weekly_trades['pnl'] > 0]) / weekly_trades_count if weekly_trades_count > 0 else 0
        else:
            weekly_pnl = 0
            weekly_trades_count = 0
            weekly_win_rate = 0
        
        lines = [
            "# 量化交易系统周度报告",
            "",
            "## 基本信息",
            f"- 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 报告周期: {datetime.now().date() - timedelta(days=7)} 至 {datetime.now().date()}",
            "",
            "## 周度表现",
            f"- 周度盈亏: {weekly_pnl:,.2f}",
            f"- 周度交易次数: {weekly_trades_count}",
            f"- 周度胜率: {weekly_win_rate:.2%}",
            "",
            "## 累计表现",
            f"- 总盈亏: {metrics.total_pnl:,.2f}",
            f"- 总交易次数: {metrics.total_trades}",
            f"- 胜率: {metrics.win_rate:.2%}",
            f"- 盈亏比: {metrics.profit_factor:.2f}",
            f"- 夏普比率: {metrics.sharpe_ratio:.2f}",
            "",
            "## 风险指标",
            f"- 最大回撤: {metrics.max_drawdown:,.2f}",
            f"- 最大回撤比例: {metrics.max_drawdown_pct:.2%}",
            f"- 当前权益: {metrics.current_equity:,.2f}",
            f"- 峰值权益: {metrics.peak_equity:,.2f}",
        ]
        
        return "\n".join(lines)
    
    def generate_performance_report(self) -> str:
        """
        生成性能报告
        
        Returns:
            报告内容
        """
        # 获取当前指标
        metrics = self.performance_tracker.get_current_metrics()
        
        # 获取权益曲线
        equity_curve = self.performance_tracker.get_equity_curve()
        
        # 计算额外指标
        if not equity_curve.empty:
            # 计算月度收益
            equity_series = equity_curve['equity']
            monthly_returns = equity_series.resample('M').last().pct_change().dropna()
            
            # 计算年度收益
            annual_returns = equity_series.resample('Y').last().pct_change().dropna()
            
            # 计算最大回撤持续时间
            drawdown_series = equity_curve['drawdown']
            in_drawdown = drawdown_series > 0
            drawdown_groups = (~in_drawdown).cumsum()
            drawdown_durations = in_drawdown.groupby(drawdown_groups).sum()
            max_drawdown_duration = drawdown_durations.max() if not drawdown_durations.empty else 0
        else:
            monthly_returns = pd.Series()
            annual_returns = pd.Series()
            max_drawdown_duration = 0
        
        lines = [
            "# 量化交易系统性能报告",
            "",
            "## 基本信息",
            f"- 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 初始资金: {self.performance_tracker.initial_capital:,.2f}",
            f"- 当前权益: {metrics.current_equity:,.2f}",
            "",
            "## 收益指标",
            f"- 总盈亏: {metrics.total_pnl:,.2f}",
            f"- 总收益率: {metrics.total_pnl / self.performance_tracker.initial_capital:.2%}",
            f"- 年化收益率: {metrics.sharpe_ratio:.2%} (基于夏普比率)",
            f"- 平均交易盈亏: {metrics.avg_trade_pnl:,.2f}",
            "",
            "## 风险指标",
            f"- 最大回撤: {metrics.max_drawdown:,.2f}",
            f"- 最大回撤比例: {metrics.max_drawdown_pct:.2%}",
            f"- 最大回撤持续时间: {max_drawdown_duration} 个周期",
            f"- 夏普比率: {metrics.sharpe_ratio:.2f}",
            f"- 盈亏比: {metrics.profit_factor:.2f}",
            "",
            "## 交易统计",
            f"- 总交易次数: {metrics.total_trades}",
            f"- 盈利交易: {metrics.winning_trades}",
            f"- 亏损交易: {metrics.losing_trades}",
            f"- 胜率: {metrics.win_rate:.2%}",
            "",
            "## 月度收益",
        ]
        
        if not monthly_returns.empty:
            for date, ret in monthly_returns.items():
                lines.append(f"- {date.strftime('%Y-%m')}: {ret:.2%}")
        else:
            lines.append("- 暂无数据")
        
        lines.append("")
        lines.append("## 年度收益")
        
        if not annual_returns.empty:
            for date, ret in annual_returns.items():
                lines.append(f"- {date.year}: {ret:.2%}")
        else:
            lines.append("- 暂无数据")
        
        return "\n".join(lines)
    
    def save_report(self, report: str, filename: str) -> bool:
        """
        保存报告
        
        Args:
            report: 报告内容
            filename: 文件名
            
        Returns:
            是否成功
        """
        try:
            # 创建报告目录
            report_dir = Path("reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存报告
            filepath = report_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"报告已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
            return False


class MonitoringSystem:
    """监控系统"""
    
    def __init__(self, initial_capital: float = 10000):
        """
        初始化监控系统
        
        Args:
            initial_capital: 初始资金
        """
        self.performance_tracker = PerformanceTracker(initial_capital)
        self.alert_manager = AlertManager()
        self.report_generator = ReportGenerator(self.performance_tracker)
        
        # 设置默认告警规则
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self) -> None:
        """设置默认告警规则"""
        # 日度亏损告警
        self.alert_manager.add_rule(AlertRule(
            rule_id="daily_loss_warning",
            name="日度亏损警告",
            condition="lt",
            threshold=-0.02,
            metric="daily_pnl",
            severity="warning",
            message_template="日度亏损警告: {metric} = {value:.2%} (阈值: {threshold:.2%})"
        ))
        
        # 最大回撤告警
        self.alert_manager.add_rule(AlertRule(
            rule_id="max_drawdown_warning",
            name="最大回撤警告",
            condition="gt",
            threshold=0.1,
            metric="max_drawdown_pct",
            severity="warning",
            message_template="最大回撤警告: {metric} = {value:.2%} (阈值: {threshold:.2%})"
        ))
        
        # 胜率告警
        self.alert_manager.add_rule(AlertRule(
            rule_id="win_rate_warning",
            name="胜率警告",
            condition="lt",
            threshold=0.4,
            metric="win_rate",
            severity="info",
            message_template="胜率警告: {metric} = {value:.2%} (阈值: {threshold:.2%})"
        ))
    
    def update_equity(self, equity: float, timestamp: Optional[datetime] = None) -> None:
        """
        更新权益
        
        Args:
            equity: 当前权益
            timestamp: 时间戳
        """
        self.performance_tracker.update_equity(equity, timestamp)
        
        # 检查告警
        metrics = self.performance_tracker.get_current_metrics()
        self.alert_manager.check_alerts(metrics)
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        commission: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        记录交易
        
        Args:
            symbol: 交易对
            side: 交易方向
            quantity: 数量
            entry_price: 入场价格
            exit_price: 出场价格
            pnl: 盈亏
            commission: 手续费
            timestamp: 时间戳
        """
        self.performance_tracker.record_trade(
            symbol, side, quantity, entry_price, exit_price, pnl, commission, timestamp
        )
        
        # 检查告警
        metrics = self.performance_tracker.get_current_metrics()
        self.alert_manager.check_alerts(metrics)
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            系统状态
        """
        metrics = self.performance_tracker.get_current_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            'metrics': {
                'total_pnl': metrics.total_pnl,
                'daily_pnl': metrics.daily_pnl,
                'win_rate': metrics.win_rate,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'max_drawdown_pct': metrics.max_drawdown_pct,
                'total_trades': metrics.total_trades,
                'profit_factor': metrics.profit_factor,
                'current_equity': metrics.current_equity,
                'peak_equity': metrics.peak_equity
            },
            'alerts': {
                'active_count': len(active_alerts),
                'alerts': [
                    {
                        'id': alert.alert_id,
                        'title': alert.title,
                        'severity': alert.severity,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in active_alerts
                ]
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_daily_report(self, date: Optional[str] = None) -> str:
        """
        生成日度报告
        
        Args:
            date: 日期
            
        Returns:
            报告内容
        """
        return self.report_generator.generate_daily_report(date)
    
    def generate_weekly_report(self) -> str:
        """
        生成周度报告
        
        Returns:
            报告内容
        """
        return self.report_generator.generate_weekly_report()
    
    def generate_performance_report(self) -> str:
        """
        生成性能报告
        
        Returns:
            报告内容
        """
        return self.report_generator.generate_performance_report()


# 全局监控系统实例
_monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system() -> MonitoringSystem:
    """获取全局监控系统"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem()
    return _monitoring_system


def set_monitoring_system(system: MonitoringSystem) -> None:
    """设置全局监控系统"""
    global _monitoring_system
    _monitoring_system = system