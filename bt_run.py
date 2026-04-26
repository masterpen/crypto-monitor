import sys
sys.stdout.reconfigure(encoding='utf-8')
from run_backtest import run_backtest
res = run_backtest('BTCUSDT', 'KDJStrategy', 180, 10000)
TR = res.total_pnl / 10000
print(f'=== REAL DATA BACKTEST ===')
print(f'Total Return  : {TR:.4f} ({TR*100:.2f}%)')
print(f'Sharpe Ratio  : {res.sharpe_ratio:.3f}')
print(f'Max Drawdown  : {res.max_drawdown_pct:.4f} ({res.max_drawdown_pct*100:.2f}%)')
print(f'Win Rate      : {res.win_rate:.4f} ({res.win_rate*100:.2f}%)')
print(f'Profit Factor : {res.profit_factor:.3f}')
print(f'Total Trades  : {res.total_trades}')
