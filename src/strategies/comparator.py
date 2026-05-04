"""
策略比较模块
支持多策略回测比较和性能分析
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from ..backtest.engine import BacktestEngine, BacktestResult
from ..exceptions import StrategyError

logger = logging.getLogger(__name__)


@dataclass
class StrategyComparisonResult:
    """策略比较结果"""
    strategy_names: List[str]
    results: Dict[str, BacktestResult]
    rankings: Dict[str, Dict[str, int]]
    best_strategy: str
    comparison_metrics: Dict[str, Dict[str, float]]
    
    def summary(self) -> str:
        """生成比较结果摘要"""
        lines = ["=== 策略比较结果 ===", ""]
        
        # 添加排名
        lines.append("排名:")
        for metric, ranking in self.rankings.items():
            lines.append(f"  {metric}:")
            for rank, (strategy, score) in enumerate(sorted(ranking.items(), key=lambda x: x[1], reverse=True), 1):
                lines.append(f"    {rank}. {strategy}: {score:.4f}")
        
        lines.append("")
        lines.append(f"综合最佳策略: {self.best_strategy}")
        
        return "\n".join(lines)


class StrategyComparator:
    """策略比较器"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        初始化比较器
        
        Args:
            data: 回测数据
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点率
        """
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # 验证数据
        if data is None or data.empty:
            raise ValueError("回测数据不能为空")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"数据缺少必需的列: {missing_columns}")
    
    def compare_strategies(
        self,
        strategies: Dict[str, Any],
        metrics: Optional[List[str]] = None
    ) -> StrategyComparisonResult:
        """
        比较多个策略
        
        Args:
            strategies: 策略字典，格式: {'strategy_name': strategy_instance}
            metrics: 比较指标列表
            
        Returns:
            比较结果
        """
        if not strategies:
            raise ValueError("策略字典不能为空")
        
        if metrics is None:
            metrics = ['sharpe_ratio', 'total_pnl', 'win_rate', 'profit_factor', 'max_drawdown_pct']
        
        results: Dict[str, BacktestResult] = {}
        comparison_metrics: Dict[str, Dict[str, float]] = {}
        
        for name, strategy in strategies.items():
            try:
                logger.info(f"回测策略: {name}")
                
                # 创建回测引擎
                engine = BacktestEngine(
                    initial_capital=self.initial_capital,
                    commission=self.commission,
                    slippage=self.slippage
                )
                
                # 运行回测
                result = engine.run_backtest(
                    self.data,
                    strategy.generate_signal
                )
                
                results[name] = result
                comparison_metrics[name] = {
                    'sharpe_ratio': result.sharpe_ratio,
                    'total_pnl': result.total_pnl,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'max_drawdown_pct': result.max_drawdown_pct,
                    'calmar_ratio': result.calmar_ratio,
                    'sortino_ratio': result.sortino_ratio,
                    'total_trades': result.total_trades,
                    'avg_trade_pnl': result.avg_trade_pnl
                }
                
            except Exception as e:
                logger.error(f"策略 {name} 回测失败: {e}")
                continue
        
        if not results:
            raise StrategyError("所有策略回测都失败了")
        
        # 计算排名
        rankings: Dict[str, Dict[str, int]] = {}
        for metric in metrics:
            if metric == 'max_drawdown_pct':
                # 最大回撤越小越好
                sorted_strategies = sorted(
                    results.keys(),
                    key=lambda x: getattr(results[x], metric, 0)
                )
            else:
                # 其他指标越大越好
                sorted_strategies = sorted(
                    results.keys(),
                    key=lambda x: getattr(results[x], metric, 0),
                    reverse=True
                )
            
            rankings[metric] = {name: rank + 1 for rank, name in enumerate(sorted_strategies)}
        
        # 计算综合得分
        total_scores: Dict[str, float] = {name: 0 for name in results.keys()}
        for metric, ranking in rankings.items():
            for name, rank in ranking.items():
                total_scores[name] += rank
        
        # 选择最佳策略
        best_strategy = min(total_scores.keys(), key=lambda x: total_scores[x])
        
        return StrategyComparisonResult(
            strategy_names=list(results.keys()),
            results=results,
            rankings=rankings,
            best_strategy=best_strategy,
            comparison_metrics=comparison_metrics
        )
    
    def compare_parameters(
        self,
        strategy_class: type,
        param_sets: Dict[str, Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> StrategyComparisonResult:
        """
        比较同一策略的不同参数
        
        Args:
            strategy_class: 策略类
            param_sets: 参数集合，格式: {'param_name': {'param1': value1, ...}}
            metrics: 比较指标列表
            
        Returns:
            比较结果
        """
        strategies: Dict[str, Any] = {}
        for name, params in param_sets.items():
            try:
                strategy = strategy_class(**params)
                strategies[name] = strategy
            except Exception as e:
                logger.error(f"创建策略 {name} 失败: {e}")
                continue
        
        return self.compare_strategies(strategies, metrics)
    
    def generate_report(
        self,
        comparison_result: StrategyComparisonResult,
        output_file: Optional[str] = None
    ) -> str:
        """
        生成比较报告
        
        Args:
            comparison_result: 比较结果
            output_file: 输出文件路径
            
        Returns:
            报告内容
        """
        lines = ["# 策略比较报告", ""]
        
        # 添加基本信息
        lines.append("## 基本信息")
        lines.append(f"- 数据长度: {len(self.data)} 条")
        lines.append(f"- 初始资金: {self.initial_capital}")
        lines.append(f"- 手续费率: {self.commission}")
        lines.append(f"- 滑点率: {self.slippage}")
        lines.append("")
        
        # 添加策略列表
        lines.append("## 策略列表")
        for name in comparison_result.strategy_names:
            lines.append(f"- {name}")
        lines.append("")
        
        # 添加详细结果
        lines.append("## 详细结果")
        for name, result in comparison_result.results.items():
            lines.append(f"### {name}")
            lines.append(f"- 总交易次数: {result.total_trades}")
            lines.append(f"- 盈利交易: {result.winning_trades} ({result.win_rate:.2%})")
            lines.append(f"- 总盈亏: {result.total_pnl:.2f}")
            lines.append(f"- 最大回撤: {result.max_drawdown:.2f} ({result.max_drawdown_pct:.2%})")
            lines.append(f"- 夏普比率: {result.sharpe_ratio:.2f}")
            lines.append(f"- 盈亏比: {result.profit_factor:.2f}")
            lines.append("")
        
        # 添加排名
        lines.append("## 排名")
        for metric, ranking in comparison_result.rankings.items():
            lines.append(f"### {metric}")
            for rank, (strategy, score) in enumerate(sorted(ranking.items(), key=lambda x: x[1], reverse=True), 1):
                lines.append(f"{rank}. {strategy}: {score:.4f}")
            lines.append("")
        
        # 添加最佳策略
        lines.append("## 最佳策略")
        lines.append(f"综合最佳策略: {comparison_result.best_strategy}")
        
        report = "\n".join(lines)
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"报告已保存到: {output_file}")
        
        return report


def compare_strategies(
    data: pd.DataFrame,
    strategies: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    **kwargs
) -> StrategyComparisonResult:
    """
    策略比较的便捷函数
    
    Args:
        data: 回测数据
        strategies: 策略字典
        metrics: 比较指标列表
        **kwargs: 其他参数
        
    Returns:
        比较结果
    """
    comparator = StrategyComparator(data, **kwargs)
    return comparator.compare_strategies(strategies, metrics)