"""
执行算法模块
支持TWAP、VWAP等执行算法
"""
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..exceptions import ExecutionError, OrderError

logger = logging.getLogger(__name__)


class ExecutionAlgorithm(Enum):
    """执行算法类型"""
    TWAP = "twap"  # 时间加权平均价格
    VWAP = "vwap"  # 成交量加权平均价格
    ICEBERG = "iceberg"  # 冰山订单
    SNIPER = "sniper"  # 狙击手算法
    IS = "is"  # 实施缺口算法


@dataclass
class AlgorithmConfig:
    """算法配置"""
    algorithm: ExecutionAlgorithm
    duration_minutes: int = 60  # 执行持续时间（分钟）
    num_slices: int = 10  # 切片数量
    max_participation_rate: float = 0.1  # 最大参与率
    price_limit: Optional[float] = None  # 价格限制
    urgency: str = "medium"  # 紧急程度 ('low', 'medium', 'high')
    randomize_slices: bool = True  # 是否随机化切片
    min_slice_size: float = 0.001  # 最小切片大小
    max_slice_size: float = 0.1  # 最大切片大小


@dataclass
class ExecutionSlice:
    """执行切片"""
    slice_id: int
    quantity: float
    scheduled_time: datetime
    actual_time: Optional[datetime] = None
    price: Optional[float] = None
    status: str = "pending"  # 'pending', 'executing', 'completed', 'failed'
    error_message: Optional[str] = None


@dataclass
class ExecutionPlan:
    """执行计划"""
    plan_id: str
    symbol: str
    side: str
    total_quantity: float
    algorithm: ExecutionAlgorithm
    config: AlgorithmConfig
    slices: List[ExecutionSlice]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "created"  # 'created', 'executing', 'completed', 'failed', 'cancelled'
    executed_quantity: float = 0.0
    average_price: float = 0.0
    total_commission: float = 0.0


class TWAPAlgorithm:
    """时间加权平均价格算法"""
    
    def __init__(self, config: AlgorithmConfig):
        """
        初始化TWAP算法
        
        Args:
            config: 算法配置
        """
        self.config = config
    
    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        start_time: Optional[datetime] = None
    ) -> ExecutionPlan:
        """
        创建执行计划
        
        Args:
            symbol: 交易对
            side: 交易方向
            total_quantity: 总数量
            start_time: 开始时间
            
        Returns:
            执行计划
        """
        if start_time is None:
            start_time = datetime.now()
        
        # 计算切片数量
        num_slices = self.config.num_slices
        
        # 计算每个切片的基础数量
        base_quantity = total_quantity / num_slices
        
        # 创建切片
        slices = []
        current_time = start_time
        
        for i in range(num_slices):
            # 计算切片数量（可选随机化）
            if self.config.randomize_slices:
                # 随机调整切片数量（±20%）
                random_factor = np.random.uniform(0.8, 1.2)
                slice_quantity = base_quantity * random_factor
                
                # 限制在最小和最大切片大小之间
                min_quantity = total_quantity * self.config.min_slice_size
                max_quantity = total_quantity * self.config.max_slice_size
                slice_quantity = max(min_quantity, min(slice_quantity, max_quantity))
            else:
                slice_quantity = base_quantity
            
            # 确保不超过总数量
            if i == num_slices - 1:
                # 最后一个切片使用剩余数量
                executed_quantity = sum(s.quantity for s in slices)
                slice_quantity = total_quantity - executed_quantity
            
            # 创建执行切片
            execution_slice = ExecutionSlice(
                slice_id=i,
                quantity=slice_quantity,
                scheduled_time=current_time
            )
            
            slices.append(execution_slice)
            
            # 计算下一个切片时间
            time_interval = timedelta(minutes=self.config.duration_minutes / num_slices)
            current_time += time_interval
        
        # 创建执行计划
        plan_id = f"TWAP_{symbol}_{side}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return ExecutionPlan(
            plan_id=plan_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            algorithm=ExecutionAlgorithm.TWAP,
            config=self.config,
            slices=slices,
            created_at=datetime.now()
        )


class VWAPAlgorithm:
    """成交量加权平均价格算法"""
    
    def __init__(self, config: AlgorithmConfig, volume_profile: Optional[pd.Series] = None):
        """
        初始化VWAP算法
        
        Args:
            config: 算法配置
            volume_profile: 成交量分布（24小时）
        """
        self.config = config
        self.volume_profile = volume_profile
        
        # 如果没有提供成交量分布，使用默认分布
        if self.volume_profile is None:
            self.volume_profile = self._create_default_volume_profile()
    
    def _create_default_volume_profile(self) -> pd.Series:
        """创建默认的成交量分布"""
        # 创建24小时的成交量分布
        hours = pd.date_range(start='00:00', end='23:59', freq='H').hour
        
        # 模拟典型的成交量分布（亚洲、欧洲、美国交易时段）
        volume_weights = []
        for hour in hours:
            if 0 <= hour < 8:  # 亚洲时段
                weight = 0.3
            elif 8 <= hour < 16:  # 欧洲时段
                weight = 0.5
            else:  # 美国时段
                weight = 0.7
            
            # 添加一些随机性
            weight *= np.random.uniform(0.8, 1.2)
            volume_weights.append(weight)
        
        # 归一化
        total = sum(volume_weights)
        volume_weights = [w / total for w in volume_weights]
        
        return pd.Series(volume_weights, index=hours)
    
    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        start_time: Optional[datetime] = None
    ) -> ExecutionPlan:
        """
        创建执行计划
        
        Args:
            symbol: 交易对
            side: 交易方向
            total_quantity: 总数量
            start_time: 开始时间
            
        Returns:
            执行计划
        """
        if start_time is None:
            start_time = datetime.now()
        
        # 计算切片数量
        num_slices = self.config.num_slices
        
        # 获取当前小时的成交量权重
        current_hour = start_time.hour
        current_weight = self.volume_profile.get(current_hour, 0.5)
        
        # 计算每个切片的基础数量
        base_quantity = total_quantity / num_slices
        
        # 创建切片
        slices = []
        current_time = start_time
        
        for i in range(num_slices):
            # 获取当前小时的成交量权重
            hour = current_time.hour
            weight = self.volume_profile.get(hour, 0.5)
            
            # 根据成交量权重调整切片数量
            volume_factor = weight / current_weight
            
            # 计算切片数量
            slice_quantity = base_quantity * volume_factor
            
            # 限制在最小和最大切片大小之间
            min_quantity = total_quantity * self.config.min_slice_size
            max_quantity = total_quantity * self.config.max_slice_size
            slice_quantity = max(min_quantity, min(slice_quantity, max_quantity))
            
            # 确保不超过总数量
            if i == num_slices - 1:
                # 最后一个切片使用剩余数量
                executed_quantity = sum(s.quantity for s in slices)
                slice_quantity = total_quantity - executed_quantity
            
            # 创建执行切片
            execution_slice = ExecutionSlice(
                slice_id=i,
                quantity=slice_quantity,
                scheduled_time=current_time
            )
            
            slices.append(execution_slice)
            
            # 计算下一个切片时间
            time_interval = timedelta(minutes=self.config.duration_minutes / num_slices)
            current_time += time_interval
        
        # 创建执行计划
        plan_id = f"VWAP_{symbol}_{side}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return ExecutionPlan(
            plan_id=plan_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            algorithm=ExecutionAlgorithm.VWAP,
            config=self.config,
            slices=slices,
            created_at=datetime.now()
        )


class IcebergAlgorithm:
    """冰山订单算法"""
    
    def __init__(self, config: AlgorithmConfig):
        """
        初始化冰山订单算法
        
        Args:
            config: 算法配置
        """
        self.config = config
    
    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        visible_quantity: float,
        start_time: Optional[datetime] = None
    ) -> ExecutionPlan:
        """
        创建执行计划
        
        Args:
            symbol: 交易对
            side: 交易方向
            total_quantity: 总数量
            visible_quantity: 可见数量
            start_time: 开始时间
            
        Returns:
            执行计划
        """
        if start_time is None:
            start_time = datetime.now()
        
        # 计算切片数量
        num_slices = int(np.ceil(total_quantity / visible_quantity))
        
        # 创建切片
        slices = []
        current_time = start_time
        
        for i in range(num_slices):
            # 计算切片数量
            if i == num_slices - 1:
                # 最后一个切片使用剩余数量
                executed_quantity = sum(s.quantity for s in slices)
                slice_quantity = total_quantity - executed_quantity
            else:
                slice_quantity = visible_quantity
            
            # 创建执行切片
            execution_slice = ExecutionSlice(
                slice_id=i,
                quantity=slice_quantity,
                scheduled_time=current_time
            )
            
            slices.append(execution_slice)
            
            # 计算下一个切片时间
            time_interval = timedelta(minutes=self.config.duration_minutes / num_slices)
            current_time += time_interval
        
        # 创建执行计划
        plan_id = f"ICEBERG_{symbol}_{side}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return ExecutionPlan(
            plan_id=plan_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            algorithm=ExecutionAlgorithm.ICEBERG,
            config=self.config,
            slices=slices,
            created_at=datetime.now()
        )


class ExecutionEngine:
    """执行引擎"""
    
    def __init__(self, order_executor: Optional[Callable] = None):
        """
        初始化执行引擎
        
        Args:
            order_executor: 订单执行函数
        """
        self.order_executor = order_executor
        self.execution_plans: Dict[str, ExecutionPlan] = {}
        self.is_running = False
        
        # 统计信息
        self.stats = {
            'total_plans': 0,
            'completed_plans': 0,
            'failed_plans': 0,
            'total_executed_quantity': 0.0,
            'total_commission': 0.0
        }
    
    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        algorithm: ExecutionAlgorithm,
        config: AlgorithmConfig,
        **kwargs
    ) -> ExecutionPlan:
        """
        创建执行计划
        
        Args:
            symbol: 交易对
            side: 交易方向
            total_quantity: 总数量
            algorithm: 执行算法
            config: 算法配置
            **kwargs: 其他参数
            
        Returns:
            执行计划
        """
        if algorithm == ExecutionAlgorithm.TWAP:
            twap = TWAPAlgorithm(config)
            plan = twap.create_execution_plan(symbol, side, total_quantity, **kwargs)
        elif algorithm == ExecutionAlgorithm.VWAP:
            vwap = VWAPAlgorithm(config)
            plan = vwap.create_execution_plan(symbol, side, total_quantity, **kwargs)
        elif algorithm == ExecutionAlgorithm.ICEBERG:
            visible_quantity = kwargs.get('visible_quantity', total_quantity * 0.1)
            iceberg = IcebergAlgorithm(config)
            plan = iceberg.create_execution_plan(symbol, side, total_quantity, visible_quantity, **kwargs)
        else:
            raise ExecutionError(f"不支持的执行算法: {algorithm}")
        
        # 保存执行计划
        self.execution_plans[plan.plan_id] = plan
        self.stats['total_plans'] += 1
        
        logger.info(f"创建执行计划: {plan.plan_id}, 算法: {algorithm.value}")
        
        return plan
    
    async def execute_plan(
        self,
        plan_id: str,
        price_func: Optional[Callable] = None
    ) -> bool:
        """
        执行计划
        
        Args:
            plan_id: 计划ID
            price_func: 价格获取函数
            
        Returns:
            是否成功
        """
        if plan_id not in self.execution_plans:
            raise ExecutionError(f"执行计划不存在: {plan_id}")
        
        plan = self.execution_plans[plan_id]
        
        if plan.status != "created":
            raise ExecutionError(f"执行计划状态错误: {plan.status}")
        
        plan.status = "executing"
        plan.started_at = datetime.now()
        
        logger.info(f"开始执行计划: {plan_id}")
        
        try:
            for execution_slice in plan.slices:
                # 等待到计划执行时间
                now = datetime.now()
                if execution_slice.scheduled_time > now:
                    wait_seconds = (execution_slice.scheduled_time - now).total_seconds()
                    await asyncio.sleep(wait_seconds)
                
                # 执行切片
                success = await self._execute_slice(plan, execution_slice, price_func)
                
                if not success:
                    plan.status = "failed"
                    plan.completed_at = datetime.now()
                    self.stats['failed_plans'] += 1
                    logger.error(f"执行计划失败: {plan_id}, 切片: {execution_slice.slice_id}")
                    return False
            
            # 执行完成
            plan.status = "completed"
            plan.completed_at = datetime.now()
            self.stats['completed_plans'] += 1
            
            logger.info(f"执行计划完成: {plan_id}")
            
            return True
        except Exception as e:
            plan.status = "failed"
            plan.completed_at = datetime.now()
            self.stats['failed_plans'] += 1
            logger.error(f"执行计划异常: {plan_id}, 错误: {e}")
            return False
    
    async def _execute_slice(
        self,
        plan: ExecutionPlan,
        execution_slice: ExecutionSlice,
        price_func: Optional[Callable] = None
    ) -> bool:
        """
        执行切片
        
        Args:
            plan: 执行计划
            execution_slice: 执行切片
            price_func: 价格获取函数
            
        Returns:
            是否成功
        """
        execution_slice.status = "executing"
        execution_slice.actual_time = datetime.now()
        
        try:
            # 获取价格
            if price_func:
                price = await price_func(plan.symbol)
            else:
                # 使用计划价格或市场价格
                price = plan.config.price_limit or 0.0
            
            # 执行订单
            if self.order_executor:
                success = await self.order_executor(
                    symbol=plan.symbol,
                    side=plan.side,
                    quantity=execution_slice.quantity,
                    price=price
                )
                
                if success:
                    execution_slice.status = "completed"
                    execution_slice.price = price
                    
                    # 更新计划统计
                    plan.executed_quantity += execution_slice.quantity
                    plan.average_price = (
                        (plan.average_price * (plan.executed_quantity - execution_slice.quantity) + 
                         price * execution_slice.quantity) / plan.executed_quantity
                    )
                    
                    # 更新全局统计
                    self.stats['total_executed_quantity'] += execution_slice.quantity
                    
                    logger.info(f"执行切片成功: {plan.plan_id}, 切片: {execution_slice.slice_id}")
                    
                    return True
                else:
                    execution_slice.status = "failed"
                    execution_slice.error_message = "订单执行失败"
                    
                    return False
            else:
                # 模拟执行
                execution_slice.status = "completed"
                execution_slice.price = price
                
                # 更新计划统计
                plan.executed_quantity += execution_slice.quantity
                
                logger.info(f"模拟执行切片: {plan.plan_id}, 切片: {execution_slice.slice_id}")
                
                return True
        except Exception as e:
            execution_slice.status = "failed"
            execution_slice.error_message = str(e)
            
            logger.error(f"执行切片异常: {plan.plan_id}, 切片: {execution_slice.slice_id}, 错误: {e}")
            
            return False
    
    def cancel_plan(self, plan_id: str) -> bool:
        """
        取消执行计划
        
        Args:
            plan_id: 计划ID
            
        Returns:
            是否成功
        """
        if plan_id not in self.execution_plans:
            return False
        
        plan = self.execution_plans[plan_id]
        
        if plan.status in ["completed", "failed", "cancelled"]:
            return False
        
        plan.status = "cancelled"
        plan.completed_at = datetime.now()
        
        logger.info(f"取消执行计划: {plan_id}")
        
        return True
    
    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        获取计划状态
        
        Args:
            plan_id: 计划ID
            
        Returns:
            计划状态
        """
        if plan_id not in self.execution_plans:
            return None
        
        plan = self.execution_plans[plan_id]
        
        return {
            'plan_id': plan.plan_id,
            'symbol': plan.symbol,
            'side': plan.side,
            'total_quantity': plan.total_quantity,
            'executed_quantity': plan.executed_quantity,
            'average_price': plan.average_price,
            'status': plan.status,
            'algorithm': plan.algorithm.value,
            'created_at': plan.created_at.isoformat(),
            'started_at': plan.started_at.isoformat() if plan.started_at else None,
            'completed_at': plan.completed_at.isoformat() if plan.completed_at else None,
            'slices_total': len(plan.slices),
            'slices_completed': len([s for s in plan.slices if s.status == "completed"]),
            'slices_failed': len([s for s in plan.slices if s.status == "failed"])
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息
        """
        return {
            'total_plans': self.stats['total_plans'],
            'completed_plans': self.stats['completed_plans'],
            'failed_plans': self.stats['failed_plans'],
            'success_rate': self.stats['completed_plans'] / self.stats['total_plans'] if self.stats['total_plans'] > 0 else 0,
            'total_executed_quantity': self.stats['total_executed_quantity'],
            'total_commission': self.stats['total_commission']
        }


def create_twap_plan(
    symbol: str,
    side: str,
    total_quantity: float,
    duration_minutes: int = 60,
    num_slices: int = 10,
    **kwargs
) -> ExecutionPlan:
    """
    创建TWAP执行计划的便捷函数
    
    Args:
        symbol: 交易对
        side: 交易方向
        total_quantity: 总数量
        duration_minutes: 执行持续时间（分钟）
        num_slices: 切片数量
        **kwargs: 其他参数
        
    Returns:
        执行计划
    """
    config = AlgorithmConfig(
        algorithm=ExecutionAlgorithm.TWAP,
        duration_minutes=duration_minutes,
        num_slices=num_slices,
        **kwargs
    )
    
    engine = ExecutionEngine()
    return engine.create_execution_plan(symbol, side, total_quantity, ExecutionAlgorithm.TWAP, config)


def create_vwap_plan(
    symbol: str,
    side: str,
    total_quantity: float,
    duration_minutes: int = 60,
    num_slices: int = 10,
    volume_profile: Optional[pd.Series] = None,
    **kwargs
) -> ExecutionPlan:
    """
    创建VWAP执行计划的便捷函数
    
    Args:
        symbol: 交易对
        side: 交易方向
        total_quantity: 总数量
        duration_minutes: 执行持续时间（分钟）
        num_slices: 切片数量
        volume_profile: 成交量分布
        **kwargs: 其他参数
        
    Returns:
        执行计划
    """
    config = AlgorithmConfig(
        algorithm=ExecutionAlgorithm.VWAP,
        duration_minutes=duration_minutes,
        num_slices=num_slices,
        **kwargs
    )
    
    engine = ExecutionEngine()
    return engine.create_execution_plan(symbol, side, total_quantity, ExecutionAlgorithm.VWAP, config)