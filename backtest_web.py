"""
回测可视化 Web 服务
提供 K线图 + 指标叠加 + 逐笔交易记录的 TradingView 风格交互式回测页面
启动: python backtest_web.py
访问: http://localhost:5000
"""
import json
import math
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string, request

from src.backtest.engine import BacktestEngine, PositionSide
from src.strategies import (
    TrendStrategy, MACDStrategy, MeanReversionStrategy,
    RSIStrategy, MomentumStrategy,
    KDJStrategy, KDJCrossStrategy,
    KDJExitAtK50, KDJExitOverbought, KDJExitWithSLTP,
    VolumeSurgeStrategy
)

# 导入因子框架（自动注册内置因子）
import src.factors.builtin  # noqa: F401
from src.factors import FactorRegistry, FactorEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

STRATEGIES = {
    'TrendStrategy': TrendStrategy,
    'MACDStrategy': MACDStrategy,
    'MeanReversionStrategy': MeanReversionStrategy,
    'RSIStrategy': RSIStrategy,
    'MomentumStrategy': MomentumStrategy,
    'KDJStrategy': KDJStrategy,
    'KDJCrossStrategy': KDJCrossStrategy,
    'KDJExitAtK50': KDJExitAtK50,
    'KDJExitOverbought': KDJExitOverbought,
    'KDJExitWithSLTP': KDJExitWithSLTP,
    'VolumeSurgeStrategy': VolumeSurgeStrategy,
}


# ─── 数据获取 ───────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str = '1h', days: int = 90) -> pd.DataFrame:
    url = "https://data-api.binance.vision/api/v3/klines"
    end_ms = int(datetime.now().timestamp() * 1000)
    start_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_data, cur = [], start_ms
    max_retries = 3
    while cur < end_ms:
        params = {'symbol': symbol, 'interval': interval, 'startTime': cur, 'endTime': end_ms, 'limit': 1000}
        batch = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                batch = resp.json()
                break
            except Exception as e:
                logger.warning(f"fetch_klines 请求失败 (symbol={symbol}, start={cur}, 第{attempt+1}次): {e}")
                if attempt < max_retries - 1:
                    import time; time.sleep(1 * (attempt + 1))
        if not isinstance(batch, list) or not batch:
            if not batch:
                logger.warning(f"fetch_klines 第{max_retries}次重试仍失败，跳过本批次 (symbol={symbol}, start={cur})")
            break
        all_data.extend(batch)
        cur = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    if not all_data:
        logger.error(f"fetch_klines 未获取到数据: symbol={symbol}, interval={interval}, days={days}")
        return pd.DataFrame()
    logger.info(f"fetch_klines: {symbol} 获取 {len(all_data)} 根K线")
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('time')
    return df[['open', 'high', 'low', 'close', 'volume']]


def get_top_symbols(limit: int = 30) -> list:
    url = "https://data-api.binance.vision/api/v3/ticker/24hr"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"获取币种列表失败: {e}, 使用默认列表")
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT',
                'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT']
    pairs = [i for i in data if i['symbol'].endswith('USDT') and float(i.get('quoteVolume', 0)) > 0]
    pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
    return [i['symbol'] for i in pairs[:limit]]


# ─── 指标计算 ───────────────────────────────────────────────

def _record_close(trade_pairs, entry_info, exit_price, exit_time, reason):
    """记录一笔平仓交易"""
    if entry_info['side'] == 'long':
        pnl = (exit_price - entry_info['entry_price']) * entry_info['quantity'] - exit_price * entry_info['quantity'] * 0.001
    else:
        pnl = (entry_info['entry_price'] - exit_price) * entry_info['quantity'] - exit_price * entry_info['quantity'] * 0.001
    cost = entry_info['entry_price'] * entry_info['quantity']
    trade_pairs.append({
        'entry_time': entry_info['entry_time'],
        'exit_time': exit_time,
        'side': entry_info['side'],
        'entry_price': entry_info['entry_price'],
        'exit_price': exit_price,
        'quantity': entry_info['quantity'],
        'pnl': pnl,
        'pnl_pct': pnl / cost if cost > 0 else 0,
        'reason': reason,
    })



def calc_ma(data: pd.DataFrame, period: int) -> pd.Series:
    return data['close'].rolling(window=period).mean()


def calc_bollinger(data: pd.DataFrame, period: int = 20, std_mult: float = 2.0):
    mid = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    return mid, mid + std_mult * std, mid - std_mult * std


def calc_macd(data: pd.DataFrame, fast=12, slow=26, signal=9):
    close = data['close']
    ema_f = close.ewm(span=fast).mean()
    ema_s = close.ewm(span=slow).mean()
    macd_line = ema_f - ema_s
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_rsi(data: pd.DataFrame, period: int = 14):
    close = data['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(span=period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_kdj(data: pd.DataFrame, period: int = 9, k_period=3, d_period=3):
    low_min = data['low'].rolling(window=period).min()
    high_max = data['high'].rolling(window=period).max()
    diff = (high_max - low_min).replace(0, 1e-10)
    rsv = (data['close'] - low_min) / diff * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(com=k_period - 1, adjust=False).mean()
    d = k.ewm(com=d_period - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calc_volume_ma(data: pd.DataFrame, period: int = 20):
    return data['volume'].rolling(window=period).mean()


# ─── 增强版回测：记录每个 bar 的信号和持仓状态 ─────────────
def run_backtest_with_details(symbol: str, strategy_name: str, days: int, interval: str = '1h',
                               initial_capital: float = 10000):
    data = fetch_klines(symbol, interval, days)
    if data is None or len(data) < 50:
        return None, "数据不足"

    strategy_class = STRATEGIES.get(strategy_name)
    if not strategy_class:
        return None, f"未知策略: {strategy_name}"

    strategy = strategy_class()
    engine = BacktestEngine(initial_capital=initial_capital, commission=0.001, slippage=0.0005)

    def strategy_func(d, idx, position_side='flat', entry_price=None, **params):
        return strategy.generate_signal(d, idx, position_side=position_side, entry_price=entry_price)

    # 手动运行回测以记录每个 bar 的详情
    bt_engine = BacktestEngine(initial_capital=initial_capital, commission=0.001, slippage=0.0005)
    pending_signal = None
    pending_limit_price = None  # 限价单价格（VolumeSurgeStrategy 用）
    trade_pairs = []  # 配对的开平仓记录
    entry_info = None  # 当前持仓的入场信息

    # 判断是否使用限价单入场
    use_limit_order = (hasattr(strategy, 'entry_mode') and strategy.entry_mode == 'limit'
                       and hasattr(strategy, 'calc_limit_price'))

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
        signal = strategy_func(data, i, position_side=current_position_side, entry_price=entry_price_val)

        # 执行上一根的 pending signal
        if pending_signal is not None and i > 0:
            exec_price = current_open
            total_equity = bt_engine.capital + bt_engine._calc_market_value(exec_price)

            # ── 限价单执行逻辑 ──
            if pending_signal == 'long' and pending_limit_price is not None:
                # 限价单：检查下一根K线是否触及限价
                if current_open <= pending_limit_price:
                    # 开盘价已低于限价，直接以开盘价成交（更优价格）
                    exec_price = current_open
                elif current_low <= pending_limit_price:
                    # K线最低价触及限价，以限价成交
                    exec_price = pending_limit_price
                else:
                    # 限价未触及，放弃入场
                    exec_price = None

            if pending_signal == 'long' and exec_price is not None and bt_engine.position.side != PositionSide.LONG:
                qty = (total_equity * 0.5) / exec_price
                factor = 10 ** bt_engine.quantity_precision
                qty = math.floor(qty * factor) / factor
                if qty > 0:
                    # 先平反向仓位
                    if bt_engine.position.side == PositionSide.SHORT and entry_info:
                        _record_close(trade_pairs, entry_info, exec_price, int(row.name.timestamp()), '反向开多')
                        entry_info = None
                    reason = "策略信号:long(限价)" if pending_limit_price else "策略信号:long"
                    bt_engine.open_long(row.name, exec_price, qty, reason=reason)
                    entry_info = {
                        'entry_time': int(row.name.timestamp()),
                        'entry_price': exec_price,
                        'quantity': qty,
                        'side': 'long',
                    }

            elif pending_signal == 'short' and bt_engine.position.side != PositionSide.SHORT:
                qty = (total_equity * 0.5) / exec_price
                factor = 10 ** bt_engine.quantity_precision
                qty = math.floor(qty * factor) / factor
                if qty > 0:
                    if bt_engine.position.side == PositionSide.LONG and entry_info:
                        _record_close(trade_pairs, entry_info, exec_price, int(row.name.timestamp()), '反向开空')
                        entry_info = None
                    bt_engine.open_short(row.name, exec_price, qty, reason="策略信号:short")
                    entry_info = {
                        'entry_time': int(row.name.timestamp()),
                        'entry_price': exec_price,
                        'quantity': qty,
                        'side': 'short',
                    }

            elif pending_signal == 'close' and bt_engine.position.side != PositionSide.FLAT:
                if entry_info:
                    _record_close(trade_pairs, entry_info, exec_price, int(row.name.timestamp()), '策略平仓')
                    entry_info = None
                bt_engine.close_position(row.name, exec_price, reason="策略信号:close")

            pending_signal = None
            pending_limit_price = None

        # 记录本根K线产生的信号，等下一根K线开盘执行
        if signal in ('long', 'short', 'close'):
            pending_signal = signal
            # 计算限价单价格
            if signal == 'long' and use_limit_order:
                pending_limit_price = strategy.calc_limit_price(current_close)
            else:
                pending_limit_price = None

    # 回测结束处理未平仓位
    if bt_engine.position.side != PositionSide.FLAT and entry_info:
        last_price = data.iloc[-1]['close']
        last_time = int(data.index[-1].timestamp())
        _record_close(trade_pairs, entry_info, last_price, last_time, '回测结束')

    # 计算回测指标
    result = bt_engine.calculate_metrics()
    total_return = result.total_pnl / initial_capital if initial_capital > 0 else 0

    # 构建 K 线数据
    candles = []
    for idx in range(len(data)):
        row = data.iloc[idx]
        candles.append({
            'time': int(data.index[idx].timestamp()),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
        })

    # 成交量数据（放量K线用醒目颜色标记）
    volume_ratio_threshold = getattr(strategy, 'volume_ratio_threshold', 10.0) if strategy_name == 'VolumeSurgeStrategy' else None
    volumes = []
    for idx in range(len(data)):
        row = data.iloc[idx]
        is_up = row['close'] >= row['open']
        if volume_ratio_threshold and idx > 0 and data['volume'].iloc[idx - 1] > 0:
            vr = row['volume'] / data['volume'].iloc[idx - 1]
            if vr >= volume_ratio_threshold:
                # 放量K线用黄色高亮
                color = 'rgba(240,185,11,0.8)'  # 黄色 - 放量
            else:
                color = 'rgba(38,166,91,0.5)' if is_up else 'rgba(234,57,67,0.5)'
        else:
            color = 'rgba(38,166,91,0.5)' if is_up else 'rgba(234,57,67,0.5)'
        volumes.append({
            'time': int(data.index[idx].timestamp()),
            'value': float(row['volume']),
            'color': color,
        })

    # 开平仓标记
    markers = []
    for tp in trade_pairs:
        # 入场标记
        side_text = '做多' if tp['side'] == 'long' else '做空'
        markers.append({
            'time': tp['entry_time'],
            'position': 'belowBar' if tp['side'] == 'long' else 'aboveBar',
            'color': '#26a65b' if tp['side'] == 'long' else '#ea3943',
            'shape': 'arrowUp' if tp['side'] == 'long' else 'arrowDown',
            'text': f'{side_text} @{tp["entry_price"]:.2f}',
        })
        # 出场标记
        pnl_text = f'+{tp["pnl"]:.2f}' if tp['pnl'] >= 0 else f'{tp["pnl"]:.2f}'
        markers.append({
            'time': tp['exit_time'],
            'position': 'aboveBar' if tp['side'] == 'long' else 'belowBar',
            'color': '#ea3943' if tp['side'] == 'long' else '#26a65b',
            'shape': 'arrowDown' if tp['side'] == 'long' else 'arrowUp',
            'text': f'平仓 {pnl_text}',
        })

    # 指标数据
    # MA
    ma7 = calc_ma(data, 7)
    ma25 = calc_ma(data, 25)
    ma99 = calc_ma(data, 99)

    # Bollinger
    boll_mid, boll_upper, boll_lower = calc_bollinger(data)

    # MACD
    macd_line, signal_line, histogram = calc_macd(data)

    # RSI
    rsi = calc_rsi(data)

    # KDJ
    k_line, d_line, j_line = calc_kdj(data)

    # Volume MA
    vol_ma = calc_volume_ma(data)

    # Volume Ratio（当前量/前一根量）
    vol_ratio = data['volume'] / data['volume'].shift(1)
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan)

    indicators = {
        'ma7': [{'time': int(i.timestamp()), 'value': float(v)} for i, v in ma7.items() if not pd.isna(v)],
        'ma25': [{'time': int(i.timestamp()), 'value': float(v)} for i, v in ma25.items() if not pd.isna(v)],
        'ma99': [{'time': int(i.timestamp()), 'value': float(v)} for i, v in ma99.items() if not pd.isna(v)],
        'boll_mid': [{'time': int(i.timestamp()), 'value': float(v)} for i, v in boll_mid.items() if not pd.isna(v)],
        'boll_upper': [{'time': int(i.timestamp()), 'value': float(v)} for i, v in boll_upper.items() if not pd.isna(v)],
        'boll_lower': [{'time': int(i.timestamp()), 'value': float(v)} for i, v in boll_lower.items() if not pd.isna(v)],
        'macd': [{'time': int(i.timestamp()), 'value': float(v), 'color': '#26a65b' if v >= 0 else '#ea3943'} for i, v in histogram.items() if not pd.isna(v)],
        'macd_line': [{'time': int(i.timestamp()), 'value': float(v)} for i, v in macd_line.items() if not pd.isna(v)],
        'macd_signal': [{'time': int(i.timestamp()), 'value': float(v)} for i, v in signal_line.items() if not pd.isna(v)],
        'rsi': [{'time': int(i.timestamp()), 'value': float(v)} for i, v in rsi.items() if not pd.isna(v)],
        'kdj_k': [{'time': int(i.timestamp()), 'value': float(v)} for i, v in k_line.items() if not pd.isna(v)],
        'kdj_d': [{'time': int(i.timestamp()), 'value': float(v)} for i, v in d_line.items() if not pd.isna(v)],
        'kdj_j': [{'time': int(i.timestamp()), 'value': float(v)} for i, v in j_line.items() if not pd.isna(v)],
        'vol_ma': [{'time': int(i.timestamp()), 'value': float(v)} for i, v in vol_ma.items() if not pd.isna(v)],
        'vol_ratio': [{'time': int(i.timestamp()), 'value': float(v), 'color': '#f0b90b' if v >= 10 else 'rgba(139,148,158,0.27)'} for i, v in vol_ratio.items() if not pd.isna(v)],
    }

    # 权益曲线
    equity_curve = []
    for i, val in enumerate(result.equity_curve):
        if i < len(data):
            equity_curve.append({
                'time': int(data.index[i].timestamp()),
                'value': float(val),
            })

    summary = {
        'symbol': symbol,
        'strategy': strategy_name,
        'interval': interval,
        'days': days,
        'initial_capital': initial_capital,
        'total_return': total_return,
        'total_pnl': result.total_pnl,
        'win_rate': result.win_rate,
        'total_trades': result.total_trades,
        'winning_trades': result.winning_trades,
        'losing_trades': result.losing_trades,
        'sharpe_ratio': result.sharpe_ratio,
        'sortino_ratio': result.sortino_ratio,
        'max_drawdown_pct': result.max_drawdown_pct,
        'profit_factor': result.profit_factor,
        'avg_trade_pnl': result.avg_trade_pnl,
        'avg_winning': result.avg_winning,
        'avg_losing': result.avg_losing,
    }

    return {
        'candles': candles,
        'volumes': volumes,
        'markers': markers,
        'indicators': indicators,
        'trade_pairs': trade_pairs,
        'equity_curve': equity_curve,
        'summary': summary,
    }, None


# ─── API 路由 ───────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/symbols')
def api_symbols():
    limit = request.args.get('limit', 30, type=int)
    try:
        symbols = get_top_symbols(limit)
        return jsonify({'symbols': symbols})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategies')
def api_strategies():
    return jsonify({'strategies': list(STRATEGIES.keys())})


@app.route('/api/factors')
def api_factors():
    """返回所有已注册因子及其元信息"""
    return jsonify({'factors': FactorRegistry.info_all()})


@app.route('/api/factor_evaluate', methods=['POST'])
def api_factor_evaluate():
    """因子评估：计算IC/IR/分层回测/衰减分析"""
    body = request.json or {}
    symbol = body.get('symbol', 'BTCUSDT')
    factor_name = body.get('factor', 'volume_ratio')
    days = body.get('days', 90)
    interval = body.get('interval', '1h')
    forward_period = body.get('forward_period', 24)

    data = fetch_klines(symbol, interval, days)
    if data is None or len(data) < 50:
        return jsonify({'error': '数据不足'}), 400

    factor = FactorRegistry.get(factor_name)
    if factor is None:
        return jsonify({'error': f'未注册的因子: {factor_name}'}), 400

    try:
        ev = FactorEvaluator(data, factor=factor)
        ic_mean, ic_std, ir = ev.evaluate(forward_period=forward_period)
        decay = ev.decay_analysis()
        layers = ev.layer_backtest(n_layers=5, forward_period=forward_period)

        return jsonify({
            'symbol': symbol,
            'factor': factor_name,
            'interval': interval,
            'forward_period': forward_period,
            'ic': round(ic_mean, 4),
            'ic_std': round(ic_std, 4),
            'ir': round(ir, 4),
            'decay': decay.to_dict(orient='records'),
            'layers': layers.reset_index().to_dict(orient='records') if not layers.empty else [],
        })
    except Exception as e:
        logger.exception("因子评估异常")
        return jsonify({'error': f'因子评估异常: {str(e)}'}), 500


@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    body = request.json or {}
    symbol = body.get('symbol', 'BTCUSDT')
    strategy = body.get('strategy', 'MACDStrategy')
    days = body.get('days', 90)
    interval = body.get('interval', '1h')
    capital = body.get('capital', 10000)

    try:
        result, error = run_backtest_with_details(symbol, strategy, days, interval, capital)
    except Exception as e:
        logger.exception("回测执行异常")
        return jsonify({'error': f'回测执行异常: {str(e)}'}), 500
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)


# ─── 前端 HTML ──────────────────────────────────────────────
HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化回测系统</title>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #c9d1d9; --text-dim: #8b949e; --accent: #58a6ff;
  --green: #26a65b; --red: #ea3943; --orange: #d29922;
}
body { background: var(--bg); color: var(--text); font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; }
.app { display: flex; height: 100vh; overflow: hidden; }

/* ── 左侧面板 ── */
.sidebar {
  width: 300px; min-width: 300px; background: var(--surface);
  border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow-y: auto;
}
.sidebar-section { padding: 16px; border-bottom: 1px solid var(--border); }
.sidebar-section h3 { font-size: 13px; color: var(--text-dim); text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.5px; }
.sidebar-section label { display: block; font-size: 13px; margin-bottom: 4px; color: var(--text-dim); }
.sidebar-section select, .sidebar-section input {
  width: 100%; padding: 8px 10px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--bg); color: var(--text); font-size: 13px; margin-bottom: 10px;
}
.sidebar-section select:focus, .sidebar-section input:focus { outline: none; border-color: var(--accent); }

.btn-run {
  width: 100%; padding: 10px; border: none; border-radius: 6px; cursor: pointer;
  font-size: 14px; font-weight: 600; background: #238636; color: #fff; margin-top: 4px;
  transition: background 0.2s;
}
.btn-run:hover { background: #2ea043; }
.btn-run:disabled { background: #30363d; color: #8b949e; cursor: not-allowed; }

/* 指标开关 */
.indicator-toggle {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 0; font-size: 13px;
}
.indicator-toggle span { flex: 1; }
.toggle-switch {
  position: relative; width: 36px; height: 20px; flex-shrink: 0;
}
.toggle-switch input { opacity: 0; width: 0; height: 0; }
.toggle-switch .slider {
  position: absolute; inset: 0; background: var(--border); border-radius: 10px; cursor: pointer; transition: 0.2s;
}
.toggle-switch .slider::before {
  content: ''; position: absolute; width: 16px; height: 16px; left: 2px; bottom: 2px;
  background: #fff; border-radius: 50%; transition: 0.2s;
}
.toggle-switch input:checked + .slider { background: var(--accent); }
.toggle-switch input:checked + .slider::before { transform: translateX(16px); }

.indicator-color {
  display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px;
}

/* ── 主区域 ── */
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.top-bar {
  height: 48px; background: var(--surface); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 20px; gap: 20px; flex-shrink: 0;
}
.top-bar .symbol-name { font-size: 16px; font-weight: 600; }
.top-bar .stat { font-size: 13px; color: var(--text-dim); }
.top-bar .stat b { color: var(--text); }

.chart-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; position: relative; }
#main-chart { flex: 3; min-height: 0; }
#volume-chart { flex: 0; height: 80px; border-bottom: 1px solid var(--border); }

/* ── 悬停信息面板 ── */
.hover-info {
  position: absolute; top: 52px; left: 12px; z-index: 20;
  pointer-events: none; font-size: 12px; line-height: 1.7;
  background: rgba(13,17,23,0.88); border-radius: 6px; padding: 8px 12px;
  border: 1px solid var(--border); display: none; min-width: 200px;
  backdrop-filter: blur(4px);
}
.hover-info.active { display: block; }
.hover-info .hi-row { display: flex; align-items: center; gap: 6px; }
.hover-info .hi-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.hover-info .hi-label { color: var(--text-dim); min-width: 50px; }
.hover-info .hi-val { color: var(--text); font-variant-numeric: tabular-nums; }
.hover-info .hi-sep { height: 1px; background: var(--border); margin: 4px 0; }

.sub-charts { display: flex; flex-direction: column; flex-shrink: 0; }
.sub-chart-container { width: 100%; height: 120px; border-bottom: 1px solid var(--border); position: relative; }
.sub-chart-container:last-child { border-bottom: none; }
.sub-chart-label {
  position: absolute; top: 4px; left: 8px; font-size: 11px; color: var(--text-dim); z-index: 10;
  pointer-events: none;
}

/* ── 底部交易记录 ── */
.bottom-panel {
  height: 260px; flex-shrink: 0; background: var(--surface); border-top: 1px solid var(--border);
  display: flex; flex-direction: column; overflow: hidden;
}
.bottom-panel .tabs {
  display: flex; border-bottom: 1px solid var(--border); flex-shrink: 0;
}
.bottom-panel .tab {
  padding: 8px 16px; font-size: 13px; cursor: pointer; border-bottom: 2px solid transparent;
  color: var(--text-dim); transition: 0.2s;
}
.bottom-panel .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.bottom-panel .tab:hover { color: var(--text); }
.tab-content { flex: 1; overflow: auto; }

/* 交易表格 */
table { width: 100%; border-collapse: collapse; font-size: 12px; }
thead { position: sticky; top: 0; z-index: 1; }
th { background: #1c2128; padding: 8px 12px; text-align: left; color: var(--text-dim); font-weight: 500; white-space: nowrap; }
td { padding: 7px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }
tr:hover td { background: rgba(88,166,255,0.05); }
.pnl-pos { color: var(--green); }
.pnl-neg { color: var(--red); }
.side-long { color: var(--green); }
.side-short { color: var(--red); }

/* 统计卡片 */
.stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; padding: 16px; }
.stat-card {
  background: var(--bg); border-radius: 8px; padding: 14px; border: 1px solid var(--border);
}
.stat-card .label { font-size: 12px; color: var(--text-dim); margin-bottom: 4px; }
.stat-card .value { font-size: 20px; font-weight: 600; }
.stat-card .sub { font-size: 12px; color: var(--text-dim); margin-top: 2px; }

/* 加载动画 */
.loading-overlay {
  position: fixed; inset: 0; background: rgba(13,17,23,0.8); display: flex;
  align-items: center; justify-content: center; z-index: 1000; display: none;
}
.loading-overlay.active { display: flex; }
.spinner { width: 40px; height: 40px; border: 3px solid var(--border); border-top-color: var(--accent);
  border-radius: 50%; animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-text { margin-top: 12px; font-size: 14px; color: var(--text-dim); }
</style>
</head>
<body>

<div class="app">
  <!-- 左侧面板 -->
  <div class="sidebar">
    <div class="sidebar-section">
      <h3>回测参数</h3>
      <label>交易对 <span style="color:var(--text-dim);font-size:11px">(可选/手动输入)</span></label>
      <div style="position:relative">
        <input type="text" id="inp-symbol" list="dl-symbols" value="BTCUSDT" placeholder="输入或选择交易对" autocomplete="off">
        <datalist id="dl-symbols"></datalist>
      </div>
      <label>策略</label>
      <select id="sel-strategy"></select>
      <label>K线周期</label>
      <select id="sel-interval" onchange="onIntervalChange()">
        <option value="5m">5分钟</option>
        <option value="15m">15分钟</option>
        <option value="1h" selected>1小时</option>
        <option value="4h">4小时</option>
        <option value="1d">1天</option>
      </select>
      <label>回测天数</label>
      <input type="number" id="inp-days" value="90" min="7" max="365">
      <label>初始资金 (USDT)</label>
      <input type="number" id="inp-capital" value="10000" min="100" step="1000">
      <button class="btn-run" id="btn-run" onclick="runBacktest()">运行回测</button>
    </div>

    <div class="sidebar-section">
      <h3>主图指标</h3>
      <div class="indicator-toggle">
        <span><span class="indicator-color" style="background:#f0b90b"></span>MA7</span>
        <label class="toggle-switch"><input type="checkbox" data-indicator="ma7" data-pane="main" checked><span class="slider"></span></label>
      </div>
      <div class="indicator-toggle">
        <span><span class="indicator-color" style="background:#e0336c"></span>MA25</span>
        <label class="toggle-switch"><input type="checkbox" data-indicator="ma25" data-pane="main" checked><span class="slider"></span></label>
      </div>
      <div class="indicator-toggle">
        <span><span class="indicator-color" style="background:#8b5cf6"></span>MA99</span>
        <label class="toggle-switch"><input type="checkbox" data-indicator="ma99" data-pane="main"><span class="slider"></span></label>
      </div>
      <div class="indicator-toggle">
        <span><span class="indicator-color" style="background:#3b82f6"></span>布林带</span>
        <label class="toggle-switch"><input type="checkbox" data-indicator="boll" data-pane="main"><span class="slider"></span></label>
      </div>
    </div>

    <div class="sidebar-section">
      <h3>副图指标</h3>
      <div class="indicator-toggle">
        <span>MACD</span>
        <label class="toggle-switch"><input type="checkbox" data-indicator="macd" data-pane="sub" checked><span class="slider"></span></label>
      </div>
      <div class="indicator-toggle">
        <span>RSI</span>
        <label class="toggle-switch"><input type="checkbox" data-indicator="rsi" data-pane="sub" checked><span class="slider"></span></label>
      </div>
      <div class="indicator-toggle">
        <span>KDJ</span>
        <label class="toggle-switch"><input type="checkbox" data-indicator="kdj" data-pane="sub"><span class="slider"></span></label>
      </div>
      <div class="indicator-toggle">
        <span>成交量MA</span>
        <label class="toggle-switch"><input type="checkbox" data-indicator="vol_ma" data-pane="volume"><span class="slider"></span></label>
      </div>
      <div class="indicator-toggle">
        <span><span class="indicator-color" style="background:#f0b90b"></span>量比</span>
        <label class="toggle-switch"><input type="checkbox" data-indicator="vol_ratio" data-pane="volume"><span class="slider"></span></label>
      </div>
    </div>
  </div>

  <!-- 主区域 -->
  <div class="main">
    <div class="top-bar">
      <span class="symbol-name" id="display-symbol">BTCUSDT</span>
      <span class="stat">收益率 <b id="display-return">--</b></span>
      <span class="stat">夏普 <b id="display-sharpe">--</b></span>
      <span class="stat">回撤 <b id="display-dd">--</b></span>
      <span class="stat">胜率 <b id="display-wr">--</b></span>
      <span class="stat">交易 <b id="display-trades">--</b></span>
    </div>

    <div class="chart-area">
      <div class="hover-info" id="hover-info"></div>
      <div id="main-chart"></div>
      <div id="volume-chart"></div>
      <div class="sub-charts">
        <div class="sub-chart-container" id="macd-chart">
          <span class="sub-chart-label">MACD</span>
        </div>
        <div class="sub-chart-container" id="rsi-chart">
          <span class="sub-chart-label">RSI (14)</span>
        </div>
        <div class="sub-chart-container" id="kdj-chart">
          <span class="sub-chart-label">KDJ</span>
        </div>
      </div>
    </div>

    <div class="bottom-panel">
      <div class="tabs">
        <div class="tab active" data-tab="trades" onclick="switchTab('trades')">逐笔交易</div>
        <div class="tab" data-tab="stats" onclick="switchTab('stats')">统计概览</div>
        <div class="tab" data-tab="equity" onclick="switchTab('equity')">权益曲线</div>
        <div class="tab" data-tab="factor" onclick="switchTab('factor')">因子挖掘</div>
      </div>
      <div class="tab-content" id="tab-trades">
        <table>
          <thead><tr><th>#</th><th>方向</th><th>开仓时间</th><th>开仓价</th><th>平仓时间</th><th>平仓价</th><th>数量</th><th>盈亏</th><th>盈亏%</th><th>原因</th></tr></thead>
          <tbody id="tbody-trades"></tbody>
        </table>
      </div>
      <div class="tab-content" id="tab-stats" style="display:none">
        <div class="stats-grid" id="stats-grid"></div>
      </div>
      <div class="tab-content" id="tab-equity" style="display:none">
        <div id="equity-chart" style="height:100%"></div>
      </div>
      <div class="tab-content" id="tab-factor" style="display:none;overflow:auto;padding:16px">
        <div style="display:flex;gap:12px;align-items:flex-end;margin-bottom:16px;flex-wrap:wrap">
          <div>
            <label style="font-size:12px;color:var(--text-dim);display:block;margin-bottom:4px">因子</label>
            <select id="sel-factor" style="padding:6px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:13px;min-width:160px"></select>
          </div>
          <div>
            <label style="font-size:12px;color:var(--text-dim);display:block;margin-bottom:4px">预测周期(K线数)</label>
            <input type="number" id="inp-fwd" value="24" min="1" max="168" style="padding:6px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:13px;width:80px">
          </div>
          <button onclick="runFactorEval()" style="padding:6px 16px;border:none;border-radius:6px;background:var(--accent);color:#fff;cursor:pointer;font-size:13px">评估因子</button>
        </div>
        <div id="factor-result" style="font-size:13px;color:var(--text)"></div>
      </div>
    </div>
  </div>
</div>

<div class="loading-overlay" id="loading">
  <div style="text-align:center">
    <div class="spinner"></div>
    <div class="loading-text">正在获取数据并回测...</div>
  </div>
</div>

<script>
// ─── 全局状态 ──────────────────────────────────────────
let chartData = null;
let mainChart, volumeChart, macdChart, rsiChart, kdjChart, equityChart;
let candleSeries, volumeSeries;
let indicatorSeries = {};
let syncCallbacks = [];

// ─── 初始化图表 ────────────────────────────────────────
function initCharts() {
  // 主图
  mainChart = LightweightCharts.createChart(document.getElementById('main-chart'), {
    layout: { background: { color: '#0d1117' }, textColor: '#8b949e' },
    grid: { vertLines: { color: '#1c2128' }, horzLines: { color: '#1c2128' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#30363d' },
    timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
  });
  candleSeries = mainChart.addCandlestickSeries({
    upColor: '#26a65b', downColor: '#ea3943',
    borderUpColor: '#26a65b', borderDownColor: '#ea3943',
    wickUpColor: '#26a65b', wickDownColor: '#ea3943',
  });

  // 成交量图
  volumeChart = LightweightCharts.createChart(document.getElementById('volume-chart'), {
    layout: { background: { color: '#0d1117' }, textColor: '#8b949e' },
    grid: { vertLines: { color: '#1c2128' }, horzLines: { color: '#1c2128' } },
    rightPriceScale: { borderColor: '#30363d' },
    timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
  });
  volumeSeries = volumeChart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: '',
  });
  volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });

  // MACD 副图
  macdChart = LightweightCharts.createChart(document.getElementById('macd-chart'), {
    layout: { background: { color: '#0d1117' }, textColor: '#8b949e' },
    grid: { vertLines: { color: '#1c2128' }, horzLines: { color: '#1c2128' } },
    rightPriceScale: { borderColor: '#30363d' },
    timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
  });

  // RSI 副图
  rsiChart = LightweightCharts.createChart(document.getElementById('rsi-chart'), {
    layout: { background: { color: '#0d1117' }, textColor: '#8b949e' },
    grid: { vertLines: { color: '#1c2128' }, horzLines: { color: '#1c2128' } },
    rightPriceScale: { borderColor: '#30363d' },
    timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
  });

  // KDJ 副图
  kdjChart = LightweightCharts.createChart(document.getElementById('kdj-chart'), {
    layout: { background: { color: '#0d1117' }, textColor: '#8b949e' },
    grid: { vertLines: { color: '#1c2128' }, horzLines: { color: '#1c2128' } },
    rightPriceScale: { borderColor: '#30363d' },
    timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
  });

  // 权益曲线
  equityChart = LightweightCharts.createChart(document.getElementById('equity-chart'), {
    layout: { background: { color: '#161b22' }, textColor: '#8b949e' },
    grid: { vertLines: { color: '#1c2128' }, horzLines: { color: '#1c2128' } },
    rightPriceScale: { borderColor: '#30363d' },
    timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
  });

  // 同步时间轴
  syncCharts([mainChart, volumeChart, macdChart, rsiChart, kdjChart]);

  // 响应窗口变化
  const ro = new ResizeObserver(entries => {
    for (const e of entries) {
      const { width, height } = e.contentRect;
      const id = e.target.id;
      if (id === 'main-chart') mainChart.applyOptions({ width, height });
      else if (id === 'volume-chart') volumeChart.applyOptions({ width, height });
      else if (id === 'macd-chart') macdChart.applyOptions({ width, height });
      else if (id === 'rsi-chart') rsiChart.applyOptions({ width, height });
      else if (id === 'kdj-chart') kdjChart.applyOptions({ width, height });
      else if (id === 'equity-chart') equityChart.applyOptions({ width, height });
    }
  });
  ['main-chart','volume-chart','macd-chart','rsi-chart','kdj-chart','equity-chart'].forEach(id => {
    ro.observe(document.getElementById(id));
  });
}

function syncCharts(charts) {
  let isSyncing = false;
  charts.forEach(chart => {
    chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (isSyncing || !range) return;
      isSyncing = true;
      charts.forEach(c => { if (c !== chart) c.timeScale().setVisibleLogicalRange(range); });
      isSyncing = false;
    });
  });
}

// ─── K线周期切换时自适应天数建议 ──────────────────────
function onIntervalChange() {
  const interval = document.getElementById('sel-interval').value;
  const inp = document.getElementById('inp-days');
  const suggest = { '5m': 7, '15m': 14, '1h': 90, '4h': 180, '1d': 365 };
  const maxDays = { '5m': 14, '15m': 30, '1h': 365, '4h': 365, '1d': 365 };
  inp.max = maxDays[interval] || 365;
  if (parseInt(inp.value) > parseInt(inp.max)) inp.value = inp.max;
  if (parseInt(inp.value) < 7) inp.value = suggest[interval] || 90;
}

// ─── 加载策略列表和币种列表 ────────────────────────────
async function loadOptions() {
  const [symRes, strRes] = await Promise.all([fetch('/api/symbols?limit=50'), fetch('/api/strategies')]);
  const symData = await symRes.json();
  const strData = await strRes.json();

  const selSym = document.getElementById('dl-symbols');
  selSym.innerHTML = '';
  symData.symbols.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s;
    selSym.appendChild(opt);
  });

  const selStr = document.getElementById('sel-strategy');
  selStr.innerHTML = '';
  strData.strategies.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s; opt.textContent = s;
    if (s === 'MACDStrategy') opt.selected = true;
    selStr.appendChild(opt);
  });
}

// ─── 运行回测 ──────────────────────────────────────────
async function runBacktest() {
  const btn = document.getElementById('btn-run');
  btn.disabled = true;
  document.getElementById('loading').classList.add('active');

  const params = {
    symbol: document.getElementById('inp-symbol').value.trim().toUpperCase(),
    strategy: document.getElementById('sel-strategy').value,
    interval: document.getElementById('sel-interval').value,
    days: parseInt(document.getElementById('inp-days').value),
    capital: parseFloat(document.getElementById('inp-capital').value),
  };

  if (!params.symbol) { alert('请输入交易对'); btn.disabled = false; return; }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120000); // 120秒超时
    const res = await fetch('/api/backtest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `HTTP ${res.status}`);
    }
    const data = await res.json();
    if (data.error) { alert('回测失败: ' + data.error); return; }
    chartData = data;
    renderChart(data);
    renderTrades(data.trade_pairs);
    renderStats(data.summary);
    renderEquity(data.equity_curve);
  } catch (e) {
    if (e.name === 'AbortError') {
      alert('请求超时（120秒），可能是网络问题或数据量过大，请尝试减少回测天数');
    } else {
      alert('请求失败: ' + e.message);
    }
  } finally {
    btn.disabled = false;
    document.getElementById('loading').classList.remove('active');
  }
}

// ─── 渲染图表 ──────────────────────────────────────────
function renderChart(data) {
  // K线
  candleSeries.setData(data.candles);

  // 清除旧指标
  Object.values(indicatorSeries).forEach(s => {
    try { s.remove(); } catch(e) {}
  });
  indicatorSeries = {};

  // 成交量
  volumeSeries.setData(data.volumes);

  // 交易标记
  candleSeries.setMarkers(data.markers.sort((a,b) => a.time - b.time));

  // 指标
  if (data.indicators) {
    // 主图指标
    indicatorSeries.ma7 = mainChart.addLineSeries({ color: '#f0b90b', lineWidth: 1, priceLineVisible: false, lastValueVisible: true });
    indicatorSeries.ma7.setData(data.indicators.ma7);

    indicatorSeries.ma25 = mainChart.addLineSeries({ color: '#e0336c', lineWidth: 1, priceLineVisible: false, lastValueVisible: true });
    indicatorSeries.ma25.setData(data.indicators.ma25);

    indicatorSeries.ma99 = mainChart.addLineSeries({ color: '#8b5cf6', lineWidth: 1, priceLineVisible: false, lastValueVisible: true });
    indicatorSeries.ma99.setData(data.indicators.ma99);

    indicatorSeries.boll_upper = mainChart.addLineSeries({ color: '#3b82f6', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: true });
    indicatorSeries.boll_upper.setData(data.indicators.boll_upper);
    indicatorSeries.boll_mid = mainChart.addLineSeries({ color: '#3b82f6', lineWidth: 1, priceLineVisible: false, lastValueVisible: true });
    indicatorSeries.boll_mid.setData(data.indicators.boll_mid);
    indicatorSeries.boll_lower = mainChart.addLineSeries({ color: '#3b82f6', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: true });
    indicatorSeries.boll_lower.setData(data.indicators.boll_lower);

    // MACD
    indicatorSeries.macd_hist = macdChart.addHistogramSeries({ priceFormat: { type: 'price', precision: 4 }, priceLineVisible: false, lastValueVisible: true });
    indicatorSeries.macd_hist.setData(data.indicators.macd);
    indicatorSeries.macd_line = macdChart.addLineSeries({ color: '#58a6ff', lineWidth: 1, priceLineVisible: false, lastValueVisible: true });
    indicatorSeries.macd_line.setData(data.indicators.macd_line);
    indicatorSeries.macd_signal = macdChart.addLineSeries({ color: '#d29922', lineWidth: 1, priceLineVisible: false, lastValueVisible: true });
    indicatorSeries.macd_signal.setData(data.indicators.macd_signal);

    // RSI
    indicatorSeries.rsi = rsiChart.addLineSeries({ color: '#a855f7', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true });
    indicatorSeries.rsi.setData(data.indicators.rsi);
    // RSI 参考线
    const rsiRef70 = data.indicators.rsi.map(d => ({ time: d.time, value: 70 }));
    const rsiRef30 = data.indicators.rsi.map(d => ({ time: d.time, value: 30 }));
    indicatorSeries.rsi70 = rsiChart.addLineSeries({ color: '#ea394366', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
    indicatorSeries.rsi70.setData(rsiRef70);
    indicatorSeries.rsi30 = rsiChart.addLineSeries({ color: '#26a65b66', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
    indicatorSeries.rsi30.setData(rsiRef30);

    // KDJ
    indicatorSeries.kdj_k = kdjChart.addLineSeries({ color: '#58a6ff', lineWidth: 1, priceLineVisible: false, lastValueVisible: true });
    indicatorSeries.kdj_k.setData(data.indicators.kdj_k);
    indicatorSeries.kdj_d = kdjChart.addLineSeries({ color: '#d29922', lineWidth: 1, priceLineVisible: false, lastValueVisible: true });
    indicatorSeries.kdj_d.setData(data.indicators.kdj_d);
    indicatorSeries.kdj_j = kdjChart.addLineSeries({ color: '#a855f7', lineWidth: 1, priceLineVisible: false, lastValueVisible: true });
    indicatorSeries.kdj_j.setData(data.indicators.kdj_j);

    // Volume MA
    indicatorSeries.vol_ma = volumeChart.addLineSeries({ color: '#f0b90b', lineWidth: 1, priceLineVisible: false, lastValueVisible: true, priceScaleId: 'vol_ma' });
    indicatorSeries.vol_ma.setData(data.indicators.vol_ma);
    indicatorSeries.vol_ma.priceScale().applyOptions({ scaleMargins: { top: 0.2, bottom: 0 } });

    // Volume Ratio（量比 - 叠加在成交量图上，独立价格轴）
    indicatorSeries.vol_ratio = volumeChart.addHistogramSeries({
      color: '#f0b90b', priceFormat: { type: 'price', precision: 1 },
      priceLineVisible: false, lastValueVisible: true, priceScaleId: 'vol_ratio',
    });
    indicatorSeries.vol_ratio.setData(data.indicators.vol_ratio);
    indicatorSeries.vol_ratio.priceScale().applyOptions({ scaleMargins: { top: 0, bottom: 0.6 } });
  }

  // 应用当前指标开关状态
  applyIndicatorVisibility();

  // 自动缩放
  mainChart.timeScale().fitContent();

  // 设置悬停信息
  setupHoverInfo();

  // 更新顶部栏
  const s = data.summary;
  document.getElementById('display-symbol').textContent = s.symbol;
  const retEl = document.getElementById('display-return');
  retEl.textContent = (s.total_return * 100).toFixed(2) + '%';
  retEl.style.color = s.total_return >= 0 ? 'var(--green)' : 'var(--red)';
  document.getElementById('display-sharpe').textContent = s.sharpe_ratio.toFixed(2);
  document.getElementById('display-dd').textContent = (s.max_drawdown_pct * 100).toFixed(2) + '%';
  document.getElementById('display-wr').textContent = (s.win_rate * 100).toFixed(1) + '%';
  document.getElementById('display-trades').textContent = s.total_trades;
}

// ─── 悬停指标信息 ──────────────────────────────────────
const HOVER_CONFIG = {
  ma7:     { label: 'MA7',   color: '#f0b90b', series: 'ma7',     digits: 2 },
  ma25:    { label: 'MA25',  color: '#e0336c', series: 'ma25',    digits: 2 },
  ma99:    { label: 'MA99',  color: '#8b5cf6', series: 'ma99',    digits: 2 },
  boll_up: { label: 'BOLL上', color: '#3b82f6', series: 'boll_upper', digits: 2 },
  boll_mid:{ label: 'BOLL中', color: '#3b82f6', series: 'boll_mid',  digits: 2 },
  boll_low:{ label: 'BOLL下', color: '#3b82f6', series: 'boll_lower', digits: 2 },
  macd_l:  { label: 'MACD',  color: '#58a6ff', series: 'macd_line',   digits: 4 },
  macd_s:  { label: 'Signal',color: '#d29922', series: 'macd_signal', digits: 4 },
  macd_h:  { label: 'Hist',  color: '#8b949e', series: 'macd_hist',   digits: 4 },
  rsi:     { label: 'RSI',   color: '#a855f7', series: 'rsi',         digits: 2 },
  kdj_k:   { label: 'K',     color: '#58a6ff', series: 'kdj_k',       digits: 2 },
  kdj_d:   { label: 'D',     color: '#d29922', series: 'kdj_d',       digits: 2 },
  kdj_j:   { label: 'J',     color: '#a855f7', series: 'kdj_j',       digits: 2 },
  vol_ma:  { label: 'VolMA', color: '#f0b90b', series: 'vol_ma',      digits: 0 },
  vol_ratio:{ label: '量比',  color: '#f0b90b', series: 'vol_ratio',   digits: 1 },
};

// 指标数据按 time 索引的查找表
let indicatorLookup = {};
function buildIndicatorLookup(data) {
  indicatorLookup = {};
  for (const key of Object.keys(HOVER_CONFIG)) {
    const cfg = HOVER_CONFIG[key];
    const arr = data.indicators[cfg.series];
    if (!arr) continue;
    const map = {};
    arr.forEach(d => { map[d.time] = d.value; });
    indicatorLookup[key] = map;
  }
}

function getIndicatorValue(key, time) {
  const map = indicatorLookup[key];
  if (!map) return null;
  return map[time] ?? null;
}

// 判断指标是否当前可见
function isIndicatorVisible(key) {
  const cfg = HOVER_CONFIG[key];
  const s = indicatorSeries[cfg.series];
  if (!s) return false;
  try { return s.options().visible !== false; } catch(e) { return true; }
}

let hoverSetupDone = false;
function setupHoverInfo() {
  if (!chartData) return;
  buildIndicatorLookup(chartData);

  if (hoverSetupDone) return; // 只绑定一次
  hoverSetupDone = true;

  const infoEl = document.getElementById('hover-info');

  mainChart.subscribeCrosshairMove(param => {
    if (!param.time || !param.point || param.point.x < 0 || param.point.y < 0) {
      infoEl.classList.remove('active');
      return;
    }
    const time = param.time;

    // OHLC
    const candle = param.seriesData.get(candleSeries);
    let html = '';
    if (candle) {
      const isUp = candle.close >= candle.open;
      const c = isUp ? 'var(--green)' : 'var(--red)';
      html += `<div class="hi-row"><span class="hi-label">O</span><span class="hi-val" style="color:${c}">${candle.open.toFixed(2)}</span></div>`;
      html += `<div class="hi-row"><span class="hi-label">H</span><span class="hi-val" style="color:${c}">${candle.high.toFixed(2)}</span></div>`;
      html += `<div class="hi-row"><span class="hi-label">L</span><span class="hi-val" style="color:${c}">${candle.low.toFixed(2)}</span></div>`;
      html += `<div class="hi-row"><span class="hi-label">C</span><span class="hi-val" style="color:${c}">${candle.close.toFixed(2)}</span></div>`;
    }

    // Volume
    const vol = param.seriesData.get(volumeSeries);
    if (vol) {
      html += `<div class="hi-row"><span class="hi-label">Vol</span><span class="hi-val">${formatNum(vol.value, 0)}</span></div>`;
    }

    // 主图指标
    const mainKeys = ['ma7','ma25','ma99','boll_up','boll_mid','boll_low'];
    let hasMain = false;
    mainKeys.forEach(key => {
      if (!isIndicatorVisible(key)) return;
      const v = getIndicatorValue(key, time);
      if (v === null) return;
      if (!hasMain) { html += '<div class="hi-sep"></div>'; hasMain = true; }
      const cfg = HOVER_CONFIG[key];
      html += `<div class="hi-row"><span class="hi-dot" style="background:${cfg.color}"></span><span class="hi-label">${cfg.label}</span><span class="hi-val">${v.toFixed(cfg.digits)}</span></div>`;
    });

    // 副图指标 - MACD
    if (isIndicatorVisible('macd_l')) {
      html += '<div class="hi-sep"></div>';
      ['macd_l','macd_s','macd_h'].forEach(key => {
        const v = getIndicatorValue(key, time);
        if (v === null) return;
        const cfg = HOVER_CONFIG[key];
        html += `<div class="hi-row"><span class="hi-dot" style="background:${cfg.color}"></span><span class="hi-label">${cfg.label}</span><span class="hi-val">${v.toFixed(cfg.digits)}</span></div>`;
      });
    }

    // RSI
    if (isIndicatorVisible('rsi')) {
      const v = getIndicatorValue('rsi', time);
      if (v !== null) {
        html += '<div class="hi-sep"></div>';
        const cfg = HOVER_CONFIG.rsi;
        html += `<div class="hi-row"><span class="hi-dot" style="background:${cfg.color}"></span><span class="hi-label">${cfg.label}</span><span class="hi-val">${v.toFixed(cfg.digits)}</span></div>`;
      }
    }

    // KDJ
    if (isIndicatorVisible('kdj_k')) {
      html += '<div class="hi-sep"></div>';
      ['kdj_k','kdj_d','kdj_j'].forEach(key => {
        const v = getIndicatorValue(key, time);
        if (v === null) return;
        const cfg = HOVER_CONFIG[key];
        html += `<div class="hi-row"><span class="hi-dot" style="background:${cfg.color}"></span><span class="hi-label">${cfg.label}</span><span class="hi-val">${v.toFixed(cfg.digits)}</span></div>`;
      });
    }

    // Vol MA
    if (isIndicatorVisible('vol_ma')) {
      const v = getIndicatorValue('vol_ma', time);
      if (v !== null) {
        html += '<div class="hi-sep"></div>';
        const cfg = HOVER_CONFIG.vol_ma;
        html += `<div class="hi-row"><span class="hi-dot" style="background:${cfg.color}"></span><span class="hi-label">${cfg.label}</span><span class="hi-val">${formatNum(v, 0)}</span></div>`;
      }
    }

    // Volume Ratio
    if (isIndicatorVisible('vol_ratio')) {
      const v = getIndicatorValue('vol_ratio', time);
      if (v !== null) {
        html += '<div class="hi-sep"></div>';
        const cfg = HOVER_CONFIG.vol_ratio;
        const colorStyle = v >= 10 ? 'color:#f0b90b;font-weight:bold' : '';
        html += `<div class="hi-row"><span class="hi-dot" style="background:${cfg.color}"></span><span class="hi-label">${cfg.label}</span><span class="hi-val" style="${colorStyle}">${v.toFixed(cfg.digits)}x</span></div>`;
      }
    }

    infoEl.innerHTML = html;
    infoEl.classList.add('active');
  });

  // 鼠标离开主图时隐藏
  document.getElementById('main-chart').addEventListener('mouseleave', () => {
    infoEl.classList.remove('active');
  });
}

function formatNum(v, d) {
  if (v >= 1e9) return (v/1e9).toFixed(2) + 'B';
  if (v >= 1e6) return (v/1e6).toFixed(2) + 'M';
  if (v >= 1e3) return (v/1e3).toFixed(1) + 'K';
  return v.toFixed(d);
}

// ─── 指标显隐 ──────────────────────────────────────────
function applyIndicatorVisibility() {
  document.querySelectorAll('.toggle-switch input').forEach(input => {
    const ind = input.dataset.indicator;
    const pane = input.dataset.pane;
    const visible = input.checked;

    if (ind === 'ma7' && indicatorSeries.ma7) indicatorSeries.ma7.applyOptions({ visible });
    if (ind === 'ma25' && indicatorSeries.ma25) indicatorSeries.ma25.applyOptions({ visible });
    if (ind === 'ma99' && indicatorSeries.ma99) indicatorSeries.ma99.applyOptions({ visible });
    if (ind === 'boll') {
      ['boll_upper','boll_mid','boll_lower'].forEach(k => {
        if (indicatorSeries[k]) indicatorSeries[k].applyOptions({ visible });
      });
    }
    if (ind === 'macd') {
      document.getElementById('macd-chart').style.display = visible ? '' : 'none';
      ['macd_hist','macd_line','macd_signal'].forEach(k => {
        if (indicatorSeries[k]) indicatorSeries[k].applyOptions({ visible });
      });
      if (visible) macdChart.timeScale().fitContent();
    }
    if (ind === 'rsi') {
      document.getElementById('rsi-chart').style.display = visible ? '' : 'none';
      ['rsi','rsi70','rsi30'].forEach(k => {
        if (indicatorSeries[k]) indicatorSeries[k].applyOptions({ visible });
      });
      if (visible) rsiChart.timeScale().fitContent();
    }
    if (ind === 'kdj') {
      document.getElementById('kdj-chart').style.display = visible ? '' : 'none';
      ['kdj_k','kdj_d','kdj_j'].forEach(k => {
        if (indicatorSeries[k]) indicatorSeries[k].applyOptions({ visible });
      });
      if (visible) kdjChart.timeScale().fitContent();
    }
    if (ind === 'vol_ma' && indicatorSeries.vol_ma) {
      indicatorSeries.vol_ma.applyOptions({ visible });
    }
    if (ind === 'vol_ratio' && indicatorSeries.vol_ratio) {
      indicatorSeries.vol_ratio.applyOptions({ visible });
    }
  });
}

document.querySelectorAll('.toggle-switch input').forEach(input => {
  input.addEventListener('change', applyIndicatorVisibility);
});

// ─── 渲染交易记录 ──────────────────────────────────────
function renderTrades(trades) {
  const tbody = document.getElementById('tbody-trades');
  tbody.innerHTML = '';
  if (!trades || !trades.length) {
    tbody.innerHTML = '<tr><td colspan="10" style="text-align:center;color:var(--text-dim);padding:30px">暂无交易记录</td></tr>';
    return;
  }
  trades.forEach((t, i) => {
    const entryDate = new Date(t.entry_time * 1000).toLocaleString('zh-CN');
    const exitDate = new Date(t.exit_time * 1000).toLocaleString('zh-CN');
    const sideClass = t.side === 'long' ? 'side-long' : 'side-short';
    const sideText = t.side === 'long' ? '做多' : '做空';
    const pnlClass = t.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
    const pnlPct = (t.pnl_pct * 100).toFixed(2);
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td class="${sideClass}">${sideText}</td>
      <td>${entryDate}</td>
      <td>${t.entry_price.toFixed(2)}</td>
      <td>${exitDate}</td>
      <td>${t.exit_price.toFixed(2)}</td>
      <td>${t.quantity.toFixed(6)}</td>
      <td class="${pnlClass}">${t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}</td>
      <td class="${pnlClass}">${t.pnl_pct >= 0 ? '+' : ''}${pnlPct}%</td>
      <td>${t.reason || ''}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ─── 渲染统计概览 ──────────────────────────────────────
function renderStats(s) {
  const grid = document.getElementById('stats-grid');
  const items = [
    { label: '总收益率', value: (s.total_return * 100).toFixed(2) + '%', color: s.total_return >= 0 ? 'var(--green)' : 'var(--red)' },
    { label: '总盈亏', value: (s.total_pnl >= 0 ? '+' : '') + s.total_pnl.toFixed(2) + ' USDT', color: s.total_pnl >= 0 ? 'var(--green)' : 'var(--red)' },
    { label: '胜率', value: (s.win_rate * 100).toFixed(1) + '%', sub: `${s.winning_trades} 盈 / ${s.losing_trades} 亏` },
    { label: '总交易次数', value: s.total_trades },
    { label: '夏普比率', value: s.sharpe_ratio.toFixed(2) },
    { label: '索提诺比率', value: s.sortino_ratio.toFixed(2) },
    { label: '最大回撤', value: (s.max_drawdown_pct * 100).toFixed(2) + '%', color: 'var(--red)' },
    { label: '盈亏比', value: s.profit_factor.toFixed(2) },
    { label: '平均交易盈亏', value: s.avg_trade_pnl.toFixed(2) + ' USDT' },
    { label: '平均盈利', value: s.avg_winning.toFixed(2) + ' USDT', color: 'var(--green)' },
    { label: '平均亏损', value: s.avg_losing.toFixed(2) + ' USDT', color: 'var(--red)' },
    { label: '初始资金', value: s.initial_capital + ' USDT' },
  ];
  grid.innerHTML = items.map(it => `
    <div class="stat-card">
      <div class="label">${it.label}</div>
      <div class="value" ${it.color ? `style="color:${it.color}"` : ''}>${it.value}</div>
      ${it.sub ? `<div class="sub">${it.sub}</div>` : ''}
    </div>
  `).join('');
}

// ─── 渲染权益曲线 ──────────────────────────────────────
let equityLineSeries = null;
function renderEquity(data) {
  if (!equityChart) return;
  if (equityLineSeries) { try { equityChart.removeSeries(equityLineSeries); } catch(e){} }
  equityLineSeries = equityChart.addAreaSeries({
    topColor: 'rgba(88,166,255,0.3)', bottomColor: 'rgba(88,166,255,0.02)',
    lineColor: '#58a6ff', lineWidth: 2,
  });
  equityLineSeries.setData(data);
  equityChart.timeScale().fitContent();
}

// ─── Tab 切换 ──────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.bottom-panel .tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  ['trades','stats','equity','factor'].forEach(t => {
    document.getElementById('tab-' + t).style.display = t === tab ? '' : 'none';
  });
  if (tab === 'equity' && chartData) {
    setTimeout(() => { equityChart.timeScale().fitContent(); }, 50);
  }
  if (tab === 'factor') {
    loadFactorOptions();
  }
}

// ─── 因子挖掘 ─────────────────────────────────────────
let factorListLoaded = false;
async function loadFactorOptions() {
  if (factorListLoaded) return;
  try {
    const res = await fetch('/api/factors');
    const data = await res.json();
    const sel = document.getElementById('sel-factor');
    sel.innerHTML = '';
    (data.factors || []).forEach(f => {
      const opt = document.createElement('option');
      opt.value = f.name;
      opt.textContent = `${f.name} (${f.category}) - ${f.description}`;
      sel.appendChild(opt);
    });
    factorListLoaded = true;
  } catch(e) {
    console.error('加载因子列表失败', e);
  }
}

async function runFactorEval() {
  const symbol = document.getElementById('inp-symbol').value.trim().toUpperCase();
  const factor = document.getElementById('sel-factor').value;
  const fwd = parseInt(document.getElementById('inp-fwd').value) || 24;
  const interval = document.getElementById('sel-interval').value;
  const days = parseInt(document.getElementById('inp-days').value) || 90;
  const el = document.getElementById('factor-result');
  el.innerHTML = '<span style="color:var(--text-dim)">评估中...</span>';

  try {
    const res = await fetch('/api/factor_evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, factor, days, interval, forward_period: fwd }),
    });
    const data = await res.json();
    if (data.error) { el.innerHTML = `<span style="color:var(--red)">${data.error}</span>`; return; }

    let html = `<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-bottom:16px">`;
    html += statCard('IC均值', data.ic, data.ic > 0.03 ? 'var(--green)' : data.ic < -0.03 ? 'var(--red)' : 'var(--text)');
    html += statCard('IC标准差', data.ic_std, 'var(--text)');
    html += statCard('IR (信息比率)', data.ir, data.ir > 0.5 ? 'var(--green)' : data.ir < -0.5 ? 'var(--red)' : 'var(--text)');
    html += statCard('预测周期', data.forward_period + '根K线', 'var(--text-dim)');
    html += `</div>`;

    // 衰减分析表
    if (data.decay && data.decay.length) {
      html += `<h4 style="color:var(--text-dim);margin:12px 0 8px">因子衰减分析</h4>`;
      html += `<table><thead><tr><th>预测周期</th><th>IC</th><th>|IC|</th><th>IR</th></tr></thead><tbody>`;
      data.decay.forEach(d => {
        const icColor = d.ic > 0.03 ? 'var(--green)' : d.ic < -0.03 ? 'var(--red)' : 'var(--text)';
        html += `<tr><td>${d.period}</td><td style="color:${icColor}">${d.ic.toFixed(4)}</td><td>${d.abs_ic.toFixed(4)}</td><td>${d.ir.toFixed(4)}</td></tr>`;
      });
      html += `</tbody></table>`;
    }

    // 分层回测表
    if (data.layers && data.layers.length) {
      html += `<h4 style="color:var(--text-dim);margin:12px 0 8px">因子分层回测（5层）</h4>`;
      html += `<table><thead><tr><th>层</th><th>平均因子值</th><th>平均未来收益</th><th>样本数</th></tr></thead><tbody>`;
      data.layers.forEach(l => {
        const retColor = l.avg_return > 0 ? 'var(--green)' : 'var(--red)';
        html += `<tr><td>第${l.layer + 1}层</td><td>${l.avg_factor.toFixed(4)}</td><td style="color:${retColor}">${(l.avg_return * 100).toFixed(2)}%</td><td>${l.count}</td></tr>`;
      });
      html += `</tbody></table>`;
    }

    el.innerHTML = html;
  } catch(e) {
    el.innerHTML = `<span style="color:var(--red)">评估失败: ${e.message}</span>`;
  }
}

function statCard(label, value, color) {
  return `<div style="background:var(--bg);border-radius:8px;padding:14px;border:1px solid var(--border)">
    <div style="font-size:12px;color:var(--text-dim);margin-bottom:4px">${label}</div>
    <div style="font-size:20px;font-weight:600;color:${color}">${value}</div>
  </div>`;
}

// ─── 启动 ──────────────────────────────────────────────
window.addEventListener('load', () => {
  initCharts();
  loadOptions();
});
</script>
</body>
</html>
'''


if __name__ == '__main__':
    print("=" * 60)
    print("  量化回测可视化系统")
    print("  访问: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
