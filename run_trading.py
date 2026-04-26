"""
实盘交易入口脚本
"""
import argparse
import asyncio
import logging
import os
import yaml
from datetime import datetime

from src.data.binance_client import BinanceClient, WebSocketClient
from src.strategies import TrendStrategy, MACDStrategy, MeanReversionStrategy, RSIStrategy
from src.engine import TradingEngine, OrderSide, OrderType
from src.risk import RiskManager, RiskConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"配置加载失败: {e}")
        config = {}

    # 环境变量优先于配置文件
    binance_cfg = config.setdefault('binance', {})
    if os.environ.get('BINANCE_API_KEY'):
        binance_cfg['api_key'] = os.environ['BINANCE_API_KEY']
    if os.environ.get('BINANCE_API_SECRET'):
        binance_cfg['api_secret'] = os.environ['BINANCE_API_SECRET']

    return config


class TradingBot:
    def __init__(self, symbol, strategy_name, config):
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.config = config
        self.running = False

        self.strategy = self._create_strategy()
        self.risk_manager = RiskManager(
            config=RiskConfig(
                max_position_pct=config.get('risk', {}).get('max_position_pct', 0.2),
                max_daily_loss_pct=config.get('risk', {}).get('max_daily_loss', 0.05),
                stop_loss_pct=config.get('risk', {}).get('stop_loss_pct', 0.02)
            )
        )

    def _create_strategy(self):
        strategies = {
            'TrendStrategy': TrendStrategy,
            'MACDStrategy': MACDStrategy,
            'MeanReversionStrategy': MeanReversionStrategy,
            'RSIStrategy': RSIStrategy,
        }
        return strategies.get(self.strategy_name, TrendStrategy)()

    async def start(self):
        logger.info(f"启动交易机器人: {self.symbol} - {self.strategy_name}")
        self.running = True

        client = BinanceClient(
            api_key=self.config.get('binance', {}).get('api_key', ''),
            api_secret=self.config.get('binance', {}).get('api_secret', ''),
            testnet=self.config.get('binance', {}).get('testnet', True)
        )

        async with client:
            data = await client.get_klines(
                symbol=self.symbol,
                interval='1h',
                limit=100
            )

            if data is None or len(data) == 0:
                logger.error("无法获取市场数据")
                return

            logger.info(f"获取到 {len(data)} 条K线数据")

            current_price = data['close'].iloc[-1]
            signal = self.strategy.generate_signal(data, len(data) - 1)

            logger.info(f"当前价格: {current_price}, 信号: {signal}")

            self.running = False

    async def monitor_market(self):
        logger.info("开始市场监控...")
        ws = WebSocketClient(testnet=self.config.get('binance', {}).get('testnet', True))

        try:
            await ws.connect()
            await ws.subscribe(
                streams=[f"{self.symbol.lower()}@kline_1h"],
                callback=self.on_price_update
            )
            await ws.listen()
        except Exception as e:
            logger.error(f"WebSocket错误: {e}")
        finally:
            await ws.close()

    def on_price_update(self, data):
        kline = data.get('k', {})
        price = float(kline.get('c', 0))

        logger.debug(f"价格更新: {price}")


async def run_trading(symbol, strategy_name, config=None):
    logger.info(f"开始实盘交易: {symbol} - {strategy_name}")

    if config is None:
        config = load_config()
    bot = TradingBot(symbol, strategy_name, config)

    await bot.start()


def main():
    parser = argparse.ArgumentParser(description='量化交易实盘')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='交易对')
    parser.add_argument('--strategy', type=str, default='TrendStrategy',
                        choices=['TrendStrategy', 'MACDStrategy', 'MeanReversionStrategy', 'RSIStrategy'],
                        help='策略名称')
    parser.add_argument('--live', action='store_true', help='使用实盘（非测试网）')

    args = parser.parse_args()

    config = load_config()
    if not args.live:
        config.setdefault('binance', {})['testnet'] = True

    asyncio.run(run_trading(args.symbol, args.strategy, config))


if __name__ == "__main__":
    main()
