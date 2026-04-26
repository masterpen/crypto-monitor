import json
import sys
from run_backtest import fetch_real_klines, BacktestEngine
from src.strategies import KDJStrategy, KDJExitAtK50, KDJExitOverbought, KDJExitWithSLTP

SYMBOL = 'BTCUSDT'
DAYS = 180
CAPITAL = 10000

strategies = [
    ("Baseline (J>0 Exit)",    KDJStrategy()),
    ("Option A (K>50 Exit)",   KDJExitAtK50()),
    ("Option B (J>80 Exit)",   KDJExitOverbought()),
    ("Option C (J>0+SL/TP)",   KDJExitWithSLTP()),
]

data = fetch_real_klines(SYMBOL, '1h', DAYS)
results_list = []

for name, strategy in strategies:
    engine = BacktestEngine(initial_capital=CAPITAL, commission=0.001, slippage=0.0005)
    def make_func(s):
        def f(data, index, **kw):
            return s.generate_signal(data, index)
        return f
    result = engine.run_backtest(data, make_func(strategy))
    TR = result.total_pnl / CAPITAL
    
    res_dict = {
        "name": name,
        "total_return": TR,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown_pct,
        "total_trades": result.total_trades
    }
    results_list.append(res_dict)

with open('compare_results.json', 'w') as f:
    json.dump(results_list, f, indent=4)

print("SUCCESS: JSON saved.")
