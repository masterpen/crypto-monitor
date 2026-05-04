"""
策略参数优化模块
支持网格搜索、随机搜索等参数优化方法
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from itertools import product
import logging
import time

from ..backtest.engine import BacktestEngine, BacktestResult
from ..exceptions import StrategyParameterError, BacktestError

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    best_result: BacktestResult
    all_results: List[Dict[str, Any]]
    optimization_time: float
    
    def summary(self) -> str:
        """生成优化结果摘要"""
        return f"""
=== 策略参数优化结果 ===
最佳参数: {self.best_params}
最佳得分: {self.best_score:.4f}
优化耗时: {self.optimization_time:.2f}秒
总测试组合: {len(self.all_results)}

最佳回测结果:
{self.best_result.summary()}
"""


class StrategyOptimizer:
    """策略参数优化器"""
    
    def __init__(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        metric: str = 'sharpe_ratio',
        initial_capital: float = 10000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        初始化优化器
        
        Args:
            strategy_class: 策略类
            data: 回测数据
            metric: 优化指标 ('sharpe_ratio', 'total_pnl', 'win_rate', 'profit_factor', 'calmar_ratio')
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点率
        """
        self.strategy_class = strategy_class
        self.data = data
        self.metric = metric
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
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Tuple[BacktestResult, float]:
        """
        评估参数组合
        
        Args:
            params: 参数组合
            
        Returns:
            回测结果和得分
        """
        try:
            # 创建策略实例
            strategy = self.strategy_class(**params)
            
            # 创建回测引擎
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission=self.commission,
                slippage=self.slippage
            )
            
            # 运行回测
            result = engine.run_backtest(
                self.data,
                strategy.generate_signal,
                **params
            )
            
            # 计算得分
            score = self._calculate_score(result)
            
            return result, score
        except Exception as e:
            logger.warning(f"参数评估失败: {params}, 错误: {e}")
            # 返回默认结果
            default_result = BacktestResult()
            return default_result, -np.inf
    
    def _calculate_score(self, result: BacktestResult) -> float:
        """
        计算得分
        
        Args:
            result: 回测结果
            
        Returns:
            得分
        """
        if self.metric == 'sharpe_ratio':
            return result.sharpe_ratio
        elif self.metric == 'total_pnl':
            return result.total_pnl
        elif self.metric == 'win_rate':
            return result.win_rate
        elif self.metric == 'profit_factor':
            return result.profit_factor
        elif self.metric == 'calmar_ratio':
            return result.calmar_ratio
        else:
            raise ValueError(f"不支持的优化指标: {self.metric}")
    
    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        verbose: bool = True
    ) -> OptimizationResult:
        """
        网格搜索优化
        
        Args:
            param_grid: 参数网格，格式: {'param_name': [value1, value2, ...]}
            verbose: 是否显示进度
            
        Returns:
            优化结果
        """
        if not param_grid:
            raise ValueError("参数网格不能为空")
        
        # 生成参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        total_combinations = len(param_combinations)
        logger.info(f"开始网格搜索，共 {total_combinations} 种参数组合")
        
        all_results = []
        best_score = -np.inf
        best_params = None
        best_result = None
        
        start_time = time.time()
        
        for i, values in enumerate(param_combinations):
            params = dict(zip(param_names, values))
            
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"进度: {i + 1}/{total_combinations}")
            
            result, score = self._evaluate_params(params)
            
            all_results.append({
                'params': params,
                'score': score,
                'result': result
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                best_result = result
        
        optimization_time = time.time() - start_time
        
        logger.info(f"网格搜索完成，耗时: {optimization_time:.2f}秒")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=best_result,
            all_results=all_results,
            optimization_time=optimization_time
        )
    
    def random_search(
        self,
        param_distributions: Dict[str, Any],
        n_iter: int = 100,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        随机搜索优化
        
        Args:
            param_distributions: 参数分布，格式: {'param_name': distribution}
                distribution 可以是:
                - 列表: 随机选择其中一个值
                - 元组 (min, max): 均匀分布
                - 元组 (min, max, step): 按步长随机选择
            n_iter: 迭代次数
            verbose: 是否显示进度
            
        Returns:
            优化结果
        """
        if not param_distributions:
            raise ValueError("参数分布不能为空")
        
        logger.info(f"开始随机搜索，共 {n_iter} 次迭代")
        
        all_results = []
        best_score = -np.inf
        best_params = None
        best_result = None
        
        start_time = time.time()
        
        for i in range(n_iter):
            # 生成随机参数
            params = {}
            for param_name, distribution in param_distributions.items():
                if isinstance(distribution, list):
                    # 从列表中随机选择
                    params[param_name] = np.random.choice(distribution)
                elif isinstance(distribution, tuple):
                    if len(distribution) == 2:
                        # 均匀分布 (min, max)
                        min_val, max_val = distribution
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            params[param_name] = np.random.randint(min_val, max_val + 1)
                        else:
                            params[param_name] = np.random.uniform(min_val, max_val)
                    elif len(distribution) == 3:
                        # 按步长随机选择 (min, max, step)
                        min_val, max_val, step = distribution
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            params[param_name] = np.random.choice(
                                range(min_val, max_val + 1, step)
                            )
                        else:
                            params[param_name] = np.random.choice(
                                np.arange(min_val, max_val + step, step)
                            )
                    else:
                        raise ValueError(f"无效的分布格式: {distribution}")
                else:
                    raise ValueError(f"无效的分布类型: {type(distribution)}")
            
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"进度: {i + 1}/{n_iter}")
            
            result, score = self._evaluate_params(params)
            
            all_results.append({
                'params': params,
                'score': score,
                'result': result
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                best_result = result
        
        optimization_time = time.time() - start_time
        
        logger.info(f"随机搜索完成，耗时: {optimization_time:.2f}秒")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=best_result,
            all_results=all_results,
            optimization_time=optimization_time
        )
    
    def walk_forward_optimization(
        self,
        param_grid: Dict[str, List[Any]],
        train_ratio: float = 0.7,
        n_splits: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        滚动前向优化
        
        Args:
            param_grid: 参数网格
            train_ratio: 训练集比例
            n_splits: 分割数量
            verbose: 是否显示进度
            
        Returns:
            优化结果
        """
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio 必须在 0 和 1 之间")
        
        total_len = len(self.data)
        split_size = total_len // n_splits
        
        all_results = []
        best_params_list = []
        
        logger.info(f"开始滚动前向优化，共 {n_splits} 个分割")
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, total_len)
            
            # 分割训练集和测试集
            split_data = self.data.iloc[start_idx:end_idx]
            train_size = int(len(split_data) * train_ratio)
            
            train_data = split_data.iloc[:train_size]
            test_data = split_data.iloc[train_size:]
            
            if verbose:
                logger.info(f"分割 {i + 1}/{n_splits}: 训练集 {len(train_data)} 条, 测试集 {len(test_data)} 条")
            
            # 在训练集上优化
            train_optimizer = StrategyOptimizer(
                self.strategy_class,
                train_data,
                self.metric,
                self.initial_capital,
                self.commission,
                self.slippage
            )
            
            train_result = train_optimizer.grid_search(param_grid, verbose=False)
            best_params_list.append(train_result.best_params)
            
            # 在测试集上验证
            test_optimizer = StrategyOptimizer(
                self.strategy_class,
                test_data,
                self.metric,
                self.initial_capital,
                self.commission,
                self.slippage
            )
            
            test_result, test_score = test_optimizer._evaluate_params(train_result.best_params)
            
            all_results.append({
                'split': i + 1,
                'train_params': train_result.best_params,
                'train_score': train_result.best_score,
                'test_score': test_score,
                'test_result': test_result
            })
            
            if verbose:
                logger.info(f"  训练集得分: {train_result.best_score:.4f}, 测试集得分: {test_score:.4f}")
        
        # 计算平均最佳参数
        avg_params = self._average_params(best_params_list)
        
        return {
            'splits': all_results,
            'average_params': avg_params,
            'best_params_list': best_params_list
        }
    
    def _average_params(self, params_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算参数平均值
        
        Args:
            params_list: 参数列表
            
        Returns:
            平均参数
        """
        if not params_list:
            return {}
        
        avg_params = {}
        for key in params_list[0].keys():
            values = [p[key] for p in params_list]
            
            # 尝试计算平均值
            try:
                if all(isinstance(v, (int, float)) for v in values):
                    avg_params[key] = np.mean(values)
                    # 如果原值是整数，平均值也取整
                    if all(isinstance(v, int) for v in values):
                        avg_params[key] = int(round(avg_params[key]))
                else:
                    # 对于非数值类型，取众数
                    from collections import Counter
                    counter = Counter(values)
                    avg_params[key] = counter.most_common(1)[0][0]
            except:
                # 如果计算失败，取第一个值
                avg_params[key] = values[0]
        
        return avg_params


def optimize_strategy(
    strategy_class: type,
    data: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    metric: str = 'sharpe_ratio',
    method: str = 'grid_search',
    n_iter: int = 100,
    **kwargs
) -> OptimizationResult:
    """
    策略参数优化的便捷函数
    
    Args:
        strategy_class: 策略类
        data: 回测数据
        param_grid: 参数网格
        metric: 优化指标
        method: 优化方法 ('grid_search', 'random_search')
        n_iter: 随机搜索迭代次数
        **kwargs: 其他参数
        
    Returns:
        优化结果
    """
    optimizer = StrategyOptimizer(
        strategy_class,
        data,
        metric,
        **kwargs
    )
    
    if method == 'grid_search':
        return optimizer.grid_search(param_grid)
    elif method == 'random_search':
        return optimizer.random_search(param_grid, n_iter)
    else:
        raise ValueError(f"不支持的优化方法: {method}")