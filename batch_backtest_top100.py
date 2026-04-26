"""
批量回测 Binance 前100币种的所有策略表现，支持多策略对比
"""
import argparse
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import pandas as pd
import numpy as np

from src.data import fetch_klines, get_top_symbols_by_volume
from src.backtest.engine import BacktestEngine, PositionSide
from src.strategies import (
    TrendStrategy, MACDStrategy, MeanReversionStrategy,
    RSIStrategy, MomentumStrategy,
    KDJStrategy, KDJCrossStrategy,
    KDJExitAtK50, KDJExitOverbought, KDJExitWithSLTP,
    VolumeSurgeStrategy
)

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

STRATEGY_CHOICES = {
    '1': ('TrendStrategy', TrendStrategy),
    '2': ('MACDStrategy', MACDStrategy),
    '3': ('MeanReversionStrategy', MeanReversionStrategy),
    '4': ('RSIStrategy', RSIStrategy),
    '5': ('MomentumStrategy', MomentumStrategy),
    '6': ('KDJStrategy', KDJStrategy),
    '7': ('KDJCrossStrategy', KDJCrossStrategy),
    '8': ('KDJExitAtK50', KDJExitAtK50),
    '9': ('KDJExitOverbought', KDJExitOverbought),
    '10': ('KDJExitWithSLTP', KDJExitWithSLTP),
    '11': ('VolumeSurgeStrategy', VolumeSurgeStrategy),
    'all': ('ALL', None),
}


def _backtest_single(args):
    """全局函数（ProcessPoolExecutor 需要可 pickle）"""
    symbol, strategy_kwargs, days = args
    return backtest_symbol(symbol, strategy_kwargs, days)


def backtest_symbol(symbol: str, strategy_kwargs: dict, days: int = 90) -> dict:
    try:
        data = fetch_klines(symbol, '1h', days)
        if data is None or len(data) < 50:
            return {'symbol': symbol, 'error': '数据不足'}

        sname = strategy_kwargs['name']
        sclass = strategy_kwargs['class']
        strategy = sclass()

        bt_engine = BacktestEngine(initial_capital=10000, commission=0.001, slippage=0.0005)

        use_limit_order = (hasattr(strategy, 'entry_mode') and strategy.entry_mode == 'limit'
                           and hasattr(strategy, 'calc_limit_price'))

        if not use_limit_order:
            def strategy_func(data, index, position_side='flat', entry_price=None, **params):
                return strategy.generate_signal(data, index, position_side=position_side,
                                                entry_price=entry_price)
            results = bt_engine.run_backtest(data, strategy_func)
        else:
            pending_signal = None
            pending_limit_price = None

            for i in range(len(data)):
                row = data.iloc[i]
                current_close = row['close']
                current_open = row['open']
                current_low = row['low']

                if not bt_engine.update_equity(current_close):
                    bt_engine.close_position(row.name, current_close, reason="爆仓强平")
                    break

                current_position_side = bt_engine.position.side.value
                entry_price_val = bt_engine.position.entry_price if bt_engine.position.side != PositionSide.FLAT else None
                signal = strategy.generate_signal(data, i, position_side=current_position_side,
                                                  entry_price=entry_price_val)

                if pending_signal is not None and i > 0:
                    exec_price = current_open
                    total_equity = bt_engine.capital + bt_engine._calc_market_value(exec_price)

                    if pending_signal == 'long' and pending_limit_price is not None:
                        if current_open <= pending_limit_price:
                            exec_price = current_open
                        elif current_low <= pending_limit_price:
                            exec_price = pending_limit_price
                        else:
                            exec_price = None

                    if pending_signal == 'long' and exec_price is not None and bt_engine.position.side != PositionSide.LONG:
                        qty = (total_equity * 0.5) / exec_price
                        factor = 10 ** bt_engine.quantity_precision
                        qty = math.floor(qty * factor) / factor
                        if qty > 0:
                            if bt_engine.position.side == PositionSide.SHORT:
                                bt_engine.close_position(row.name, exec_price, reason="反向开多")
                            bt_engine.open_long(row.name, exec_price, qty, reason="策略信号:long(限价)")

                    elif pending_signal == 'short' and bt_engine.position.side != PositionSide.SHORT:
                        qty = (total_equity * 0.5) / exec_price
                        factor = 10 ** bt_engine.quantity_precision
                        qty = math.floor(qty * factor) / factor
                        if qty > 0:
                            if bt_engine.position.side == PositionSide.LONG:
                                bt_engine.close_position(row.name, exec_price, reason="反向开空")
                            bt_engine.open_short(row.name, exec_price, qty, reason="策略信号:short")

                    elif pending_signal == 'close' and bt_engine.position.side != PositionSide.FLAT:
                        bt_engine.close_position(row.name, exec_price, reason="策略信号:close")

                    pending_signal = None
                    pending_limit_price = None

                if signal in ('long', 'short', 'close'):
                    pending_signal = signal
                    if signal == 'long' and use_limit_order:
                        pending_limit_price = strategy.calc_limit_price(current_close)
                    else:
                        pending_limit_price = None

            if bt_engine.position.side != PositionSide.FLAT:
                last_price = data.iloc[-1]['close']
                bt_engine.close_position(data.index[-1], last_price, reason="回测结束")

            results = bt_engine.calculate_metrics()

        total_return = (results.total_pnl / 10000) if results.total_pnl else 0

        return {
            'symbol': symbol,
            'total_return': total_return,
            'max_drawdown_pct': results.max_drawdown_pct,
            'sharpe_ratio': results.sharpe_ratio,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'total_trades': results.total_trades,
            'total_pnl': results.total_pnl,
        }
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Top100 币种策略批量回测')
    parser.add_argument('--strategy', type=str, default='all', choices=list(STRATEGY_CHOICES.keys()),
                        help='策略编号')
    parser.add_argument('--days', type=int, default=90, help='回测天数')
    parser.add_argument('--start', type=int, default=1, help='币种排名起始位')
    parser.add_argument('--end', type=int, default=100, help='币种排名结束位')
    parser.add_argument('--workers', type=int, default=0, help='并发数（0=自动）')

    args = parser.parse_args()
    days = args.days
    rank_start = args.start
    rank_end = args.end
    max_workers = args.workers or None

    if args.strategy == 'all':
        strategy_list = [(k, v[0], v[1]) for k, v in STRATEGY_CHOICES.items() if v[0] != 'ALL']
    else:
        strategy_name, strategy_class = STRATEGY_CHOICES[args.strategy]
        strategy_list = [(args.strategy, strategy_name, strategy_class)]

    print("\n[1/3] 获取 Binance 排名...")
    symbols = get_top_symbols_by_volume(rank_start, rank_end)
    print(f"      共获取 {len(symbols)} 个 USDT 交易对")

    all_summary = []

    for idx, (skey, sname, sclass) in enumerate(strategy_list):
        print(f"\n{'=' * 70}")
        print(f"  [{idx + 1}/{len(strategy_list)}] 策略: {sname}  |  天数: {days}  |  币种: #{rank_start}~#{rank_end}")
        print(f"{'=' * 70}")

        print(f"\n[2/3] 开始批量回测 ({len(symbols)} 个币种)...")
        results = []
        done = 0

        strategy_kwargs = {'name': sname, 'class': sclass}
        task_args = [(sym, strategy_kwargs, days) for sym in symbols]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_backtest_single, ta): ta[0] for ta in task_args}
            for future in as_completed(futures):
                done += 1
                sym = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if 'error' not in result:
                        ret_str = f"{result['total_return']:+.2%}"
                        dd_str = f"{result['max_drawdown_pct']:.2%}"
                        print(f"  [{done}/{len(symbols)}] {sym}: 收益={ret_str}, 回撤={dd_str}, 交易={result['total_trades']}次")
                except Exception as e:
                    results.append({'symbol': sym, 'error': str(e)})
                    print(f"  [{done}/{len(symbols)}] {sym}: 异常 {e}")

        print(f"\n[3/3] 汇总分析")

        valid = [r for r in results if 'error' not in r]

        if not valid:
            print("没有有效回测结果！")
            all_summary.append({
                'strategy': sname, 'valid': 0, 'profitable_pct': 0,
                'avg_return': 0, 'median_return': 0, 'avg_drawdown': 0,
                'avg_sharpe': 0, 'avg_winrate': 0, 'avg_pf': 0
            })
            continue

        df = pd.DataFrame(valid)
        profitable = df[df['total_return'] > 0]

        summary = {
            'strategy': sname,
            'valid': len(valid),
            'profitable_pct': len(profitable) / len(valid),
            'avg_return': df['total_return'].mean(),
            'median_return': df['total_return'].median(),
            'avg_drawdown': df['max_drawdown_pct'].mean(),
            'avg_sharpe': df['sharpe_ratio'].mean(),
            'avg_winrate': df['win_rate'].mean(),
            'avg_pf': df['profit_factor'].mean(),
        }
        all_summary.append(summary)

        print(f"\n  有效回测: {len(valid)} | 盈利: {len(profitable)} ({len(profitable) / len(valid):.1%}) | 亏损: {len(valid) - len(profitable)}")
        print(f"  平均收益率: {summary['avg_return']:+.2%}")
        print(f"  中位数收益率: {summary['median_return']:+.2%}")
        print(f"  平均最大回撤: {summary['avg_drawdown']:.2%}")
        print(f"  平均夏普比率: {summary['avg_sharpe']:.2f}")
        print(f"  平均胜率: {summary['avg_winrate']:.2%}")
        print(f"  平均盈亏比: {summary['avg_pf']:.2f}")

        df_sorted = df.sort_values('total_return', ascending=False)
        print(f"\n  Top 5 盈利:")
        print(f"  {'币种':<12} {'收益率':>10} {'回撤':>10} {'夏普':>8} {'胜率':>8} {'交易':>6}")
        print(f"  {'-' * 56}")
        for _, row in df_sorted.head(5).iterrows():
            print(f"  {row['symbol']:<12} {row['total_return']:>+9.2%} {row['max_drawdown_pct']:>9.2%} {row['sharpe_ratio']:>8.2f} {row['win_rate']:>7.2%} {row['total_trades']:>6}")

        output_file = f"rank{rank_start}-{rank_end}_{sname}_{days}d.csv"
        df_sorted.to_csv(output_file, index=False)
        print(f"\n  详细结果已保存至: {output_file}")

    if len(all_summary) > 1:
        print(f"\n\n{'=' * 70}")
        print(f"  策略对比汇总  |  回测周期: {days}天  |  初始资金: $10,000")
        print(f"{'=' * 70}")
        print(f"  {'策略':<22} {'盈利占比':>8} {'平均收益':>10} {'中位收益':>10} {'平均回撤':>10} {'平均夏普':>8} {'平均胜率':>8} {'平均盈亏比':>10}")
        print(f"  {'-' * 90}")
        for s in sorted(all_summary, key=lambda x: x['avg_return'], reverse=True):
            print(f"  {s['strategy']:<22} "
                  f"{s['profitable_pct']:>7.1%} "
                  f"{s['avg_return']:>+9.2%} "
                  f"{s['median_return']:>+9.2%} "
                  f"{s['avg_drawdown']:>9.2%} "
                  f"{s['avg_sharpe']:>8.2f} "
                  f"{s['avg_winrate']:>7.2%} "
                  f"{s['avg_pf']:>10.2f}")

        compare_df = pd.DataFrame(all_summary).sort_values('avg_return', ascending=False)
        compare_file = f"rank{rank_start}-{rank_end}_all_strategies_compare_{days}d.csv"
        compare_df.to_csv(compare_file, index=False)
        print(f"\n  策略对比已保存至: {compare_file}")


if __name__ == "__main__":
    main()