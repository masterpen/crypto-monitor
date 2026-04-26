"""
对比4种出场方案的 KDJ 回测 (BTCUSDT, 1h, 180天)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from run_backtest import fetch_real_klines, BacktestEngine
from src.strategies import KDJStrategy
from src.strategies import KDJExitAtK50, KDJExitOverbought, KDJExitWithSLTP

SYMBOL = 'BTCUSDT'
DAYS = 180
CAPITAL = 10000

strategies = [
    ("基准版 (J转正出场)",    KDJStrategy()),
    ("方案A (K>50出场)",      KDJExitAtK50()),
    ("方案B (J>80超买出场)",  KDJExitOverbought()),
    ("方案C (J转正+止盈止损)", KDJExitWithSLTP()),
]

print(f"正在获取 {SYMBOL} 真实历史数据...")
data = fetch_real_klines(SYMBOL, '1h', DAYS)
print(f"数据加载完成: {len(data)} 根K线\n")

header = f"{'策略':<22} {'总收益':>8} {'胜率':>8} {'盈亏比':>8} {'夏普':>7} {'最大回撤':>9} {'交易次数':>8}"
print("=" * len(header))
print(header)
print("=" * len(header))

for name, strategy in strategies:
    engine = BacktestEngine(initial_capital=CAPITAL, commission=0.001, slippage=0.0005)
    def make_func(s):
        def f(data, index, **kw):
            return s.generate_signal(data, index)
        return f
    result = engine.run_backtest(data, make_func(strategy))
    TR = result.total_pnl / CAPITAL
    print(f"{name:<22} {TR:>7.2%} {result.win_rate:>8.2%} {result.profit_factor:>8.3f} "
          f"{result.sharpe_ratio:>7.2f} {result.max_drawdown_pct:>8.2%} {result.total_trades:>8}")

print("=" * len(header))
