"""
回测入口脚本
"""
import argparse
import logging

from src.backtest.engine import BacktestEngine
from src.data import fetch_klines
from src.strategies import (
    TrendStrategy, MACDStrategy, MeanReversionStrategy,
    KDJStrategy, KDJCrossStrategy,
    KDJExitAtK50, KDJExitOverbought, KDJExitWithSLTP
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


STRATEGIES = {
    'TrendStrategy': TrendStrategy,
    'MACDStrategy': MACDStrategy,
    'MeanReversionStrategy': MeanReversionStrategy,
    'KDJStrategy': KDJStrategy,
    'KDJCrossStrategy': KDJCrossStrategy,
    'KDJExitAtK50': KDJExitAtK50,
    'KDJExitOverbought': KDJExitOverbought,
    'KDJExitWithSLTP': KDJExitWithSLTP,
}


def run_backtest(symbol: str, strategy_name: str, days: int, initial_capital: float = 10000):
    logger.info(f"开始回测: {symbol} - {strategy_name}")

    data = fetch_klines(symbol, '1h', days)

    if data is None or len(data) == 0:
        logger.error("无法获取数据")
        return

    logger.info(f"获取到 {len(data)} 条K线数据")

    strategy_class = STRATEGIES.get(strategy_name)
    if strategy_class is None:
        logger.error(f"未知策略: {strategy_name}")
        return

    strategy = strategy_class()

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=0.001,
        slippage=0.0005
    )

    def strategy_func(data, index, position_side='flat', entry_price=None, **params):
        return strategy.generate_signal(data, index, position_side=position_side,
                                        entry_price=entry_price)

    results = engine.run_backtest(data, strategy_func)
    total_return = (results.total_pnl / initial_capital) if initial_capital > 0 else 0

    print("\n" + "=" * 60)
    print(f"回测结果: {symbol} - {strategy_name}")
    print("=" * 60)
    print(f"总收益率: {total_return:.2%}")
    print(f"夏普比率: {results.sharpe_ratio:.2f}")
    print(f"最大回撤: {results.max_drawdown_pct:.2%}")
    print(f"胜率: {results.win_rate:.2%}")
    print(f"盈亏比: {results.profit_factor:.2f}")
    print(f"总交易次数: {results.total_trades}")
    print("=" * 60)

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