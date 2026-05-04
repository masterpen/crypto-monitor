"""
动态风控模块
支持基于波动率的仓位管理和动态止损
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from ..exceptions import RiskError, RiskLimitExceededError

logger = logging.getLogger(__name__)


@dataclass
class VolatilityConfig:
    """波动率配置"""
    lookback_period: int = 20  # 回看周期
    atr_period: int = 14  # ATR周期
    volatility_multiplier: float = 2.0  # 波动率乘数
    max_volatility_pct: float = 0.05  # 最大波动率限制
    min_volatility_pct: float = 0.005  # 最小波动率限制


@dataclass
class DynamicRiskConfig:
    """动态风控配置"""
    base_stop_loss_pct: float = 0.02  # 基础止损比例
    volatility_adjusted_stop_loss: bool = True  # 是否使用波动率调整止损
    trailing_stop: bool = True  # 是否使用追踪止损
    trailing_stop_pct: float = 0.03  # 追踪止损比例
    max_position_volatility: float = 0.1  # 最大仓位波动率
    risk_parity: bool = True  # 是否使用风险平价
    correlation_threshold: float = 0.7  # 相关性阈值


class VolatilityCalculator:
    """波动率计算器"""
    
    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        计算平均真实范围 (ATR)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期
            
        Returns:
            ATR序列
        """
        # 计算真实范围
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_historical_volatility(
        prices: pd.Series,
        period: int = 20,
        annualize: bool = True
    ) -> pd.Series:
        """
        计算历史波动率
        
        Args:
            prices: 价格序列
            period: 计算周期
            annualize: 是否年化
            
        Returns:
            波动率序列
        """
        # 计算对数收益率
        returns = np.log(prices / prices.shift(1))
        
        # 计算滚动标准差
        volatility = returns.rolling(window=period).std()
        
        # 年化
        if annualize:
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    @staticmethod
    def calculate_parkinson_volatility(
        high: pd.Series,
        low: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        计算Parkinson波动率
        
        Args:
            high: 最高价序列
            low: 最低价序列
            period: 计算周期
            
        Returns:
            波动率序列
        """
        # 计算对数高低价比率
        log_hl = np.log(high / low)
        
        # 计算Parkinson波动率
        parkinson_vol = np.sqrt((log_hl ** 2).rolling(window=period).mean() / (4 * np.log(2)))
        
        return parkinson_vol


class DynamicPositionSizer:
    """动态仓位计算器"""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        risk_per_trade: float = 0.01,
        volatility_config: Optional[VolatilityConfig] = None
    ):
        """
        初始化动态仓位计算器
        
        Args:
            initial_capital: 初始资金
            risk_per_trade: 每笔交易风险比例
            volatility_config: 波动率配置
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.volatility_config = volatility_config or VolatilityConfig()
        self.volatility_calculator = VolatilityCalculator()
    
    def calculate_volatility_adjusted_position(
        self,
        price: float,
        volatility: float,
        max_position_pct: float = 0.2
    ) -> float:
        """
        计算波动率调整后的仓位
        
        Args:
            price: 当前价格
            volatility: 波动率
            max_position_pct: 最大仓位比例
            
        Returns:
            建议的仓位大小
        """
        if volatility <= 0:
            return 0
        
        # 基于波动率的仓位调整
        # 波动率越高，仓位越小
        volatility_factor = 1.0 / (volatility * self.volatility_config.volatility_multiplier)
        
        # 限制波动率因子范围
        volatility_factor = max(0.1, min(volatility_factor, 2.0))
        
        # 计算基础仓位
        base_position = self.initial_capital * self.risk_per_trade
        
        # 调整后的仓位
        adjusted_position = base_position * volatility_factor
        
        # 限制最大仓位
        max_position = self.initial_capital * max_position_pct
        adjusted_position = min(adjusted_position, max_position)
        
        # 计算数量
        quantity = adjusted_position / price
        
        return quantity
    
    def calculate_risk_parity_position(
        self,
        price: float,
        volatility: float,
        portfolio_volatility: float,
        target_risk: float = 0.01
    ) -> float:
        """
        计算风险平价仓位
        
        Args:
            price: 当前价格
            volatility: 资产波动率
            portfolio_volatility: 组合波动率
            target_risk: 目标风险
            
        Returns:
            建议的仓位大小
        """
        if volatility <= 0 or portfolio_volatility <= 0:
            return 0
        
        # 风险平价公式：仓位 = 目标风险 / (资产波动率 * 组合波动率)
        risk_contribution = target_risk / (volatility * portfolio_volatility)
        
        # 计算仓位金额
        position_value = self.initial_capital * risk_contribution
        
        # 计算数量
        quantity = position_value / price
        
        return quantity
    
    def calculate_atr_based_position(
        self,
        price: float,
        atr: float,
        risk_multiplier: float = 2.0
    ) -> float:
        """
        计算基于ATR的仓位
        
        Args:
            price: 当前价格
            atr: 平均真实范围
            risk_multiplier: 风险乘数
            
        Returns:
            建议的仓位大小
        """
        if atr <= 0:
            return 0
        
        # 风险金额
        risk_amount = self.initial_capital * self.risk_per_trade
        
        # 基于ATR的止损距离
        stop_distance = atr * risk_multiplier
        
        # 计算数量
        quantity = risk_amount / stop_distance
        
        return quantity


class DynamicRiskManager:
    """动态风控管理器"""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        dynamic_config: Optional[DynamicRiskConfig] = None,
        volatility_config: Optional[VolatilityConfig] = None
    ):
        """
        初始化动态风控管理器
        
        Args:
            initial_capital: 初始资金
            dynamic_config: 动态风控配置
            volatility_config: 波动率配置
        """
        self.initial_capital = initial_capital
        self.dynamic_config = dynamic_config or DynamicRiskConfig()
        self.volatility_config = volatility_config or VolatilityConfig()
        
        self.volatility_calculator = VolatilityCalculator()
        self.position_sizer = DynamicPositionSizer(
            initial_capital,
            volatility_config=self.volatility_config
        )
        
        # 持仓信息
        self.positions: Dict[str, Dict] = {}
        
        # 价格历史
        self.price_history: Dict[str, pd.Series] = {}
        
        # 波动率历史
        self.volatility_history: Dict[str, pd.Series] = {}
        
        # 追踪止损
        self.trailing_stops: Dict[str, float] = {}
        
        # 风险指标
        self.risk_metrics: Dict[str, Any] = {
            'portfolio_volatility': 0.0,
            'var_95': 0.0,
            'var_99': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def update_price(self, symbol: str, price: float, high: float, low: float) -> None:
        """
        更新价格数据
        
        Args:
            symbol: 交易对
            price: 当前价格
            high: 最高价
            low: 最低价
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = pd.Series(dtype=float)
        
        # 添加新价格
        new_data = pd.Series([price], index=[datetime.now()])
        self.price_history[symbol] = pd.concat([self.price_history[symbol], new_data])
        
        # 保留最近的数据
        max_history = max(self.volatility_config.lookback_period * 2, 100)
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol].iloc[-max_history:]
        
        # 更新波动率
        self._update_volatility(symbol)
        
        # 更新追踪止损
        if self.dynamic_config.trailing_stop:
            self._update_trailing_stop(symbol, price)
    
    def _update_volatility(self, symbol: str) -> None:
        """更新波动率"""
        if symbol not in self.price_history:
            return
        
        prices = self.price_history[symbol]
        if len(prices) < self.volatility_config.lookback_period:
            return
        
        # 计算历史波动率
        volatility = self.volatility_calculator.calculate_historical_volatility(
            prices,
            self.volatility_config.lookback_period
        )
        
        self.volatility_history[symbol] = volatility
    
    def _update_trailing_stop(self, symbol: str, price: float) -> None:
        """更新追踪止损"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        if position['side'] == 'flat':
            return
        
        entry_price = position['entry_price']
        
        # 计算当前盈亏
        if position['side'] == 'long':
            pnl_pct = (price - entry_price) / entry_price
            
            # 更新追踪止损
            if pnl_pct > 0:
                # 盈利时，追踪止损跟随
                trailing_stop_price = price * (1 - self.dynamic_config.trailing_stop_pct)
                
                if symbol not in self.trailing_stops:
                    self.trailing_stops[symbol] = trailing_stop_price
                else:
                    # 只能上移，不能下移
                    self.trailing_stops[symbol] = max(
                        self.trailing_stops[symbol],
                        trailing_stop_price
                    )
        elif position['side'] == 'short':
            pnl_pct = (entry_price - price) / entry_price
            
            # 更新追踪止损
            if pnl_pct > 0:
                # 盈利时，追踪止损跟随
                trailing_stop_price = price * (1 + self.dynamic_config.trailing_stop_pct)
                
                if symbol not in self.trailing_stops:
                    self.trailing_stops[symbol] = trailing_stop_price
                else:
                    # 只能下移，不能上移
                    self.trailing_stops[symbol] = min(
                        self.trailing_stops[symbol],
                        trailing_stop_price
                    )
    
    def calculate_dynamic_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str
    ) -> float:
        """
        计算动态止损价格
        
        Args:
            symbol: 交易对
            entry_price: 入场价格
            side: 持仓方向 ('long' 或 'short')
            
        Returns:
            止损价格
        """
        base_stop_loss = self.dynamic_config.base_stop_loss_pct
        
        # 如果启用波动率调整止损
        if self.dynamic_config.volatility_adjusted_stop_loss:
            if symbol in self.volatility_history:
                volatility = self.volatility_history[symbol].iloc[-1]
                
                if not np.isnan(volatility):
                    # 基于波动率调整止损
                    volatility_factor = volatility * self.volatility_config.volatility_multiplier
                    
                    # 限制波动率因子范围
                    volatility_factor = max(
                        self.volatility_config.min_volatility_pct,
                        min(volatility_factor, self.volatility_config.max_volatility_pct)
                    )
                    
                    # 使用较大的止损
                    base_stop_loss = max(base_stop_loss, volatility_factor)
        
        # 计算止损价格
        if side == 'long':
            stop_loss_price = entry_price * (1 - base_stop_loss)
        else:
            stop_loss_price = entry_price * (1 + base_stop_loss)
        
        return stop_loss_price
    
    def check_stop_loss(
        self,
        symbol: str,
        current_price: float
    ) -> Tuple[bool, str]:
        """
        检查是否触发止损
        
        Args:
            symbol: 交易对
            current_price: 当前价格
            
        Returns:
            (是否触发, 原因)
        """
        if symbol not in self.positions:
            return False, ''
        
        position = self.positions[symbol]
        if position['side'] == 'flat':
            return False, ''
        
        entry_price = position['entry_price']
        
        # 检查固定止损
        if position['side'] == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct <= -self.dynamic_config.base_stop_loss_pct:
                return True, f'多头止损: {pnl_pct:.2%}'
        elif position['side'] == 'short':
            pnl_pct = (entry_price - current_price) / entry_price
            if pnl_pct <= -self.dynamic_config.base_stop_loss_pct:
                return True, f'空头止损: {pnl_pct:.2%}'
        
        # 检查追踪止损
        if self.dynamic_config.trailing_stop and symbol in self.trailing_stops:
            trailing_stop = self.trailing_stops[symbol]
            
            if position['side'] == 'long' and current_price <= trailing_stop:
                return True, f'追踪止损: {current_price:.2f} <= {trailing_stop:.2f}'
            elif position['side'] == 'short' and current_price >= trailing_stop:
                return True, f'追踪止损: {current_price:.2f} >= {trailing_stop:.2f}'
        
        return False, ''
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        side: str,
        max_position_pct: float = 0.2
    ) -> float:
        """
        计算仓位大小
        
        Args:
            symbol: 交易对
            price: 当前价格
            side: 持仓方向
            max_position_pct: 最大仓位比例
            
        Returns:
            建议的仓位大小
        """
        # 获取波动率
        volatility = 0.0
        if symbol in self.volatility_history:
            volatility = self.volatility_history[symbol].iloc[-1]
        
        if volatility <= 0:
            # 使用默认波动率
            volatility = 0.02
        
        # 计算波动率调整后的仓位
        quantity = self.position_sizer.calculate_volatility_adjusted_position(
            price,
            volatility,
            max_position_pct
        )
        
        return quantity
    
    def calculate_portfolio_risk(self) -> Dict[str, float]:
        """
        计算组合风险
        
        Returns:
            风险指标字典
        """
        if not self.positions:
            return {
                'portfolio_volatility': 0.0,
                'var_95': 0.0,
                'var_99': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # 计算组合波动率
        portfolio_volatility = 0.0
        for symbol, position in self.positions.items():
            if symbol in self.volatility_history:
                volatility = self.volatility_history[symbol].iloc[-1]
                if not np.isnan(volatility):
                    # 简化计算：假设等权重
                    portfolio_volatility += volatility ** 2
        
        portfolio_volatility = np.sqrt(portfolio_volatility / len(self.positions))
        
        # 计算VaR（简化计算）
        # 假设正态分布
        var_95 = portfolio_volatility * 1.645  # 95% VaR
        var_99 = portfolio_volatility * 2.326  # 99% VaR
        
        self.risk_metrics = {
            'portfolio_volatility': portfolio_volatility,
            'var_95': var_95,
            'var_99': var_99,
            'max_drawdown': 0.0,  # 需要从权益曲线计算
            'sharpe_ratio': 0.0  # 需要从收益计算
        }
        
        return self.risk_metrics
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        获取风控报告
        
        Returns:
            风控报告字典
        """
        # 计算组合风险
        portfolio_risk = self.calculate_portfolio_risk()
        
        return {
            'positions': self.positions,
            'trailing_stops': self.trailing_stops,
            'portfolio_risk': portfolio_risk,
            'volatility_history': {
                symbol: vol.iloc[-1] if len(vol) > 0 else 0.0
                for symbol, vol in self.volatility_history.items()
            },
            'config': {
                'base_stop_loss_pct': self.dynamic_config.base_stop_loss_pct,
                'trailing_stop': self.dynamic_config.trailing_stop,
                'trailing_stop_pct': self.dynamic_config.trailing_stop_pct,
                'volatility_adjusted_stop_loss': self.dynamic_config.volatility_adjusted_stop_loss
            }
        }