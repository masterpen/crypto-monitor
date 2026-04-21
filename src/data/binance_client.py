"""
Binance API 数据采集模块
支持历史K线下载和实时WebSocket行情
"""
import asyncio
import aiohttp
import pandas as pd
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BinanceClient:
    """币安交易所客户端"""

    BASE_URL = "https://api.binance.com/api/v3"
    TESTNET_URL = "https://testnet.binance.vision/api/v3"

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    def _sign(self, params: str) -> str:
        """生成签名"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def signed_get(self, endpoint: str, params: dict = None) -> dict:
        """ signed GET请求 """
        if params is None:
            params = {}
        params['timestamp'] = int(time.time() * 1000)
        query_string = '&'.join(f"{k}={v}" for k, v in params.items())
        params['signature'] = self._sign(query_string)
        
        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.base_url}{endpoint}"
        
        async with self.session.get(url, params=params, headers=headers) as resp:
            data = await resp.json()
            if resp.status != 200:
                print(f"Error: {data}")
            return data

    async def get_account(self) -> dict:
        """获取账户信息"""
        return await self.signed_get('/api/v3/account')

    def get_historical_klines_sync(
        self,
        symbol: str,
        interval: str,
        start_date,
        end_date
    ) -> pd.DataFrame:
        """同步获取历史K线数据"""
        async def _fetch():
            async with self:
                return await self.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )
        return asyncio.run(_fetch())
    
    async def get_klines(
        self, 
        symbol: str, 
        interval: str = "1h", 
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            symbol: 交易对，如 'BTCUSDT'
            interval: K线周期，如 '1m', '5m', '1h', '4h', '1d'
            limit: 数据条数，最大1000
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        url = f"{self.base_url}/klines"
        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"API请求失败: {response.status}")
            data = await response.json()
        
        # 转换为DataFrame
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # 类型转换
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume"]]
    
    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取历史K线数据（自动分页）
        """
        start = int(pd.Timestamp(start_date).timestamp() * 1000)
        end = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        all_klines = []
        current_start = start
        
        while current_start < end:
            df = await self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start,
                end_time=end
            )
            
            if df.empty:
                break
                
            all_klines.append(df)
            current_start = int(df["open_time"].max().timestamp() * 1000) + 1
            
            # 避免请求过快
            await asyncio.sleep(0.2)
        
        if all_klines:
            return pd.concat(all_klines, ignore_index=True)
        return pd.DataFrame()
    
    async def get_symbol_info(self, symbol: str) -> Dict:
        """获取交易对信息"""
        url = f"{self.base_url}/exchangeInfo"
        async with self.session.get(url) as response:
            data = await response.json()
        
        for s in data.get("symbols", []):
            if s["symbol"] == symbol:
                return s
        return {}
    
    async def get_ticker(self, symbol: str) -> Dict:
        """获取最新行情"""
        url = f"{self.base_url}/ticker/24hr"
        params = {"symbol": symbol}
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """获取订单簿"""
        url = f"{self.base_url}/depth"
        params = {"symbol": symbol, "limit": limit}
        async with self.session.get(url, params=params) as response:
            return await response.json()


class WebSocketClient:
    """WebSocket实时行情客户端"""
    
    STREAM_URL = "wss://stream.binance.com:9443/ws"
    TESTNET_WS_URL = "wss://testnet.binance.vision/ws"
    
    def __init__(self, testnet: bool = True):
        self.ws_url = self.TESTNET_WS_URL if testnet else self.STREAM_URL
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.subscriptions: List[str] = []
        self.callbacks: Dict[str, callable] = {}
    
    async def connect(self):
        """建立WebSocket连接"""
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(self.ws_url)
        logger.info(f"WebSocket连接成功: {self.ws_url}")
    
    async def subscribe(self, streams: List[str], callback: callable):
        """
        订阅行情流
        
        Args:
            streams: 订阅的流列表，如 ['btcusdt@kline_1m', 'btcusdt@depth@100ms']
            callback: 回调函数，接收数据作为参数
        """
        for stream in streams:
            self.subscriptions.append(stream)
            self.callbacks[stream] = callback
        
        # 发送订阅消息
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(datetime.now().timestamp())
        }
        await self.ws.send_json(subscribe_msg)
        logger.info(f"已订阅: {streams}")
    
    async def unsubscribe(self, streams: List[str]):
        """取消订阅"""
        for stream in streams:
            self.subscriptions.remove(stream)
            self.callbacks.pop(stream, None)
        
        unsubscribe_msg = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": int(datetime.now().timestamp())
        }
        await self.ws.send_json(unsubscribe_msg)
    
    async def listen(self):
        """监听消息"""
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = msg.json()
                
                # 处理心跳
                if data.get("result") is None and "stream" in data:
                    stream = data["stream"]
                    if stream in self.callbacks:
                        self.callbacks[stream](data["data"])
            
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket错误: {msg.data}")
                break
    
    async def close(self):
        """关闭连接"""
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
        logger.info("WebSocket连接已关闭")