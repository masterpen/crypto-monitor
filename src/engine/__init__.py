"""
交易执行引擎
支持订单管理、持仓跟踪、与交易所API对接
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: float
    filled_quantity: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    client_order_id: str = ""


@dataclass
class TradeExecution:
    """交易执行记录"""
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    commission: float
    timestamp: datetime
    order_type: OrderType


class OrderManager:
    """订单管理器"""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.pending_orders: List[Order] = []
        self.executed_trades: List[TradeExecution] = []
        self.order_counter = 0
    
    def create_order_id(self) -> str:
        """生成唯一订单ID"""
        self.order_counter += 1
        return f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self.order_counter}"
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        price: float,
        quantity: float
    ) -> Order:
        """创建订单"""
        order_id = self.create_order_id()
        client_order_id = f"cl_{datetime.now().timestamp()}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            client_order_id=client_order_id
        )
        
        self.orders[order_id] = order
        self.pending_orders.append(order)
        
        logger.info(f"创建订单: {order}")
        return order
    
    def update_order_status(self, order_id: str, status: OrderStatus, filled_qty: float = 0):
        """更新订单状态"""
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = status
            order.filled_quantity = filled_qty
            order.updated_at = datetime.now()
            
            if status == OrderStatus.FILLED:
                self.pending_orders = [o for o in self.pending_orders if o.order_id != order_id]
            
            logger.info(f"订单状态更新: {order_id} -> {status.value}")
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self.orders.get(order_id)
    
    def get_pending_orders(self) -> List[Order]:
        """获取待成交订单"""
        return self.pending_orders.copy()
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        order = self.orders.get(order_id)
        if order and order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
            self.update_order_status(order_id, OrderStatus.CANCELLED)
            return True
        return False


class TradingEngine:
    """交易执行引擎"""
    
    def __init__(self, api_client, risk_manager=None):
        self.api_client = api_client
        self.risk_manager = risk_manager
        self.order_manager = OrderManager()
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.callbacks: Dict[str, List[Callable]] = {
            'on_order_update': [],
            'on_trade': [],
            'on_position_update': [],
            'on_error': []
        }
    
    def register_callback(self, event: str, callback: Callable):
        """注册回调"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """触发回调"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"回调执行失败 {event}: {e}")
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None
    ) -> Optional[Order]:
        """
        下单
        
        Args:
            symbol: 交易对
            side: 买卖方向
            order_type: 订单类型
            quantity: 数量
            price: 价格（限价单必需）
        """
        # 风控检查
        if self.risk_manager:
            check_result = self.risk_manager.check_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price
            )
            if not check_result['allowed']:
                logger.warning(f"风控拒绝下单: {check_result['reason']}")
                self._trigger_callback('on_error', {
                    'type': 'risk_rejection',
                    'symbol': symbol,
                    'reason': check_result['reason']
                })
                return None
        
        # 创建订单
        order = self.order_manager.create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price or 0,
            quantity=quantity
        )
        
        try:
            # 提交到交易所
            if order_type == OrderType.MARKET:
                response = await self.api_client.place_market_order(symbol, side.value, quantity)
            else:
                response = await self.api_client.place_limit_order(symbol, side.value, quantity, price)
            
            # 更新订单状态
            self.order_manager.update_order_status(order.order_id, OrderStatus.SUBMITTED)
            
            logger.info(f"订单已提交: {order.order_id}")
            return order
            
        except Exception as e:
            logger.error(f"下单失败: {e}")
            self.order_manager.update_order_status(order.order_id, OrderStatus.REJECTED)
            self._trigger_callback('on_error', {
                'type': 'order_rejection',
                'symbol': symbol,
                'error': str(e)
            })
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        order = self.order_manager.get_order(order_id)
        if not order:
            return False
        
        try:
            await self.api_client.cancel_order(order.symbol, order.client_order_id)
            return self.order_manager.cancel_order(order_id)
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None):
        """取消所有订单"""
        pending = self.order_manager.get_pending_orders()
        for order in pending:
            if symbol is None or order.symbol == symbol:
                await self.cancel_order(order.order_id)
    
    def get_position(self, symbol: str) -> Dict:
        """获取持仓"""
        return self.positions.get(symbol, {
            'symbol': symbol,
            'side': 'flat',
            'quantity': 0,
            'entry_price': 0,
            'unrealized_pnl': 0
        })
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """获取所有持仓"""
        return self.positions.copy()
    
    def update_position(self, symbol: str, trade: TradeExecution):
        """更新持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'symbol': symbol,
                'side': 'flat',
                'quantity': 0,
                'entry_price': 0,
                'unrealized_pnl': 0
            }
        
        pos = self.positions[symbol]
        
        if trade.side == OrderSide.BUY:
            if pos['side'] == 'short':
                # 平空
                pos['quantity'] -= trade.quantity
                if pos['quantity'] <= 0:
                    pos['side'] = 'flat'
                    pos['quantity'] = 0
            else:
                # 开多或加仓
                if pos['quantity'] > 0:
                    # 加仓，计算新入场价
                    total_cost = pos['quantity'] * pos['entry_price'] + trade.quantity * trade.price
                    pos['quantity'] += trade.quantity
                    pos['entry_price'] = total_cost / pos['quantity']
                else:
                    pos['side'] = 'long'
                    pos['quantity'] = trade.quantity
                    pos['entry_price'] = trade.price
        
        elif trade.side == OrderSide.SELL:
            if pos['side'] == 'long':
                # 平多
                pos['quantity'] -= trade.quantity
                if pos['quantity'] <= 0:
                    pos['side'] = 'flat'
                    pos['quantity'] = 0
            else:
                # 开空或加仓
                if pos['quantity'] > 0:
                    total_cost = pos['quantity'] * pos['entry_price'] + trade.quantity * trade.price
                    pos['quantity'] += trade.quantity
                    pos['entry_price'] = total_cost / pos['quantity']
                else:
                    pos['side'] = 'short'
                    pos['quantity'] = trade.quantity
                    pos['entry_price'] = trade.price
        
        self._trigger_callback('on_position_update', pos)
    
    async def sync_positions(self):
        """同步持仓（从交易所）"""
        try:
            account = await self.api_client.get_account()
            for balance in account.get('balances', []):
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                # 这里需要根据交易对来更新持仓
                # 简化实现
                logger.debug(f"资产 {asset}: 可用={free}, 锁定={locked}")
                
        except Exception as e:
            logger.error(f"同步持仓失败: {e}")


class SignalExecutor:
    """信号执行器 - 将策略信号转换为实际交易"""
    
    def __init__(self, trading_engine: TradingEngine):
        self.engine = trading_engine
        self.position_size_pct = 0.1  # 每次下单使用10%仓位
    
    async def execute_signal(
        self,
        symbol: str,
        signal: str,
        current_price: float,
        available_capital: float
    ):
        """
        执行交易信号
        
        Args:
            symbol: 交易对
            signal: 'long', 'short', 'close', 'hold'
            current_price: 当前价格
            available_capital: 可用资金
        """
        if signal == 'hold':
            return
        
        current_position = self.engine.get_position(symbol)
        
        # 计算下单数量
        quantity = (available_capital * self.position_size_pct) / current_price
        
        if signal == 'long':
            if current_position['side'] != 'long':
                # 平仓现有持仓
                if current_position['side'] != 'flat':
                    await self._close_position(symbol, current_position, current_price)
                # 开多
                await self.engine.place_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity
                )
        
        elif signal == 'short':
            if current_position['side'] != 'short':
                if current_position['side'] != 'flat':
                    await self._close_position(symbol, current_position, current_price)
                await self.engine.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=quantity
                )
        
        elif signal == 'close':
            if current_position['side'] != 'flat':
                await self._close_position(symbol, current_position, current_price)
    
    async def _close_position(
        self,
        symbol: str,
        position: Dict,
        current_price: float
    ):
        """平仓"""
        if position['side'] == 'long':
            await self.engine.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position['quantity']
            )
        elif position['side'] == 'short':
            await self.engine.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=position['quantity']
            )