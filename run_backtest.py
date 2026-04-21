"""
回测入口脚本
"""
import argparse
import yaml
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.backtest.engine import BacktestEngine
from src.strategies import TrendStrategy, MACDStrategy, MeanReversionStrategy, KDJStrategy, KDJCrossStrategy
from src.data.binance_client import BinanceClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """加载配置"""
    try:
        with open('config/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {}


STRATEGIES = {
    'TrendStrategy': TrendStrategy,
    'MACDStrategy': MACDStrategy,
    'MeanReversionStrategy': MeanReversionStrategy,
    'KDJStrategy': KDJStrategy,
    'KDJCrossStrategy': KDJCrossStrategy,
}


def run_backtest(symbol: str, strategy_name: str, days: int, initial_capital: float = 10000):
    """运行回测"""
    logger.info(f"开始回测: {symbol} - {strategy_name}")

    config = load_config()

    # 获取数据
    try:
        client = BinanceClient(testnet=True)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"获取数据: {start_date} - {end_date}")
        data = client.get_historical_klines_sync(
            symbol=symbol,
            interval='1h',
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        logger.warning(f"Binance API获取失败，使用模拟数据: {e}")
        # 生成模拟数据用于测试
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1h')
        data = pd.DataFrame({
            'open': 40000 + np.random.randn(100).cumsum() * 100,
            'high': 40100 + np.random.randn(100).cumsum() * 100,
            'low': 39900 + np.random.randn(100).cumsum() * 100,
            'close': 40000 + np.random.randn(100).cumsum() * 100,
            'volume': np.random.rand(100) * 1000
        }, index=dates)

    if data is None or len(data) == 0:
        logger.error("无法获取数据")
        return

    logger.info(f"获取到 {len(data)} 条K线数据")

    # 创建策略
    strategy_class = STRATEGIES.get(strategy_name)
    if strategy_class is None:
        logger.error(f"未知策略: {strategy_name}")
        return

    strategy = strategy_class()

    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=0.001,  # 0.1% 手续费
        slippage=0.0005    # 0.05% 滑点
    )

    # 定义策略函数供回测引擎使用
    def strategy_func(data, index, **params):
        return strategy.generate_signal(data, index)

    # 运行回测
    results = engine.run_backtest(data, strategy_func)

    # 计算总收益率
    total_return = (results.total_pnl / initial_capital) if initial_capital > 0 else 0

    # 输出结果
    print("\n" + "="*60)
    print(f"回测结果: {symbol} - {strategy_name}")
    print("="*60)
    print(f"总收益率: {total_return:.2%}")
    print(f"夏普比率: {results.sharpe_ratio:.2f}")
    print(f"最大回撤: {results.max_drawdown_pct:.2%}")
    print(f"胜率: {results.win_rate:.2%}")
    print(f"盈亏比: {results.profit_factor:.2f}")
    print(f"总交易次数: {results.total_trades}")
    print("="*60)

    return results


def main():
    parser = argparse.ArgumentParser(description='量化交易回测')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='交易对')
    parser.add_argument('--strategy', type=str, default='TrendStrategy',
                        choices=list(STRATEGIES.keys()), help='策略名称')
    parser.add_argument('--days', type=int, default=90, help='回测天数')
    parser.add_argument('--capital', type=float, default=10000, help='初始资金')

    args = parser.parse_args()

    run_backtest(args.symbol, args.strategy, args.days, args.capital)


if __name__ == "__main__":
    main()