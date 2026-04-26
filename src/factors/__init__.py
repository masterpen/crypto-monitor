"""
因子挖掘框架 (Factor Mining Framework)

提供标准化的因子定义、计算、评估流程，支持：
- 自定义因子注册与发现
- 因子计算（单币种/批量）
- 因子评估（IC/IR/分层回测）
- 因子组合（多因子合成信号）

使用方式：
  from src.factors import FactorBase, FactorRegistry, FactorEvaluator

  # 1. 定义因子
  class MyFactor(FactorBase):
      def calculate(self, data: pd.DataFrame) -> pd.Series:
          return data['close'].rolling(20).mean() / data['close']

  # 2. 注册
  FactorRegistry.register('my_factor', MyFactor)

  # 3. 计算
  factor_values = FactorRegistry.get('my_factor').calculate(data)

  # 4. 评估
  ev = FactorEvaluator(data)
  ic, ir = ev.evaluate('my_factor', forward_period=24)
"""

from src.factors.base import FactorBase
from src.factors.registry import FactorRegistry
from src.factors.evaluator import FactorEvaluator
from src.factors.composite import CompositeFactor

__all__ = ['FactorBase', 'FactorRegistry', 'FactorEvaluator', 'CompositeFactor']
