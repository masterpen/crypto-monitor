"""
K线数据获取工具函数（通过 binance.vision 公开接口，无需 API Key）
"""
import requests
import time
from datetime import datetime, timedelta
import pandas as pd


def fetch_klines(symbol: str, interval: str = '1h', days: int = 90) -> pd.DataFrame:
    url = "https://data-api.binance.vision/api/v3/klines"
    end_ms = int(datetime.now().timestamp() * 1000)
    start_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_data = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': 1000
        }
        batch = None
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=15)
                batch = resp.json()
                break
            except Exception:
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))

        if not isinstance(batch, list) or len(batch) == 0:
            break

        all_data.extend(batch)
        current_start = batch[-1][0] + 1
        if len(batch) < 1000:
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df.index = pd.to_datetime(df['open_time'], unit='ms')
    return df[['open', 'high', 'low', 'close', 'volume']]


def get_top_symbols_by_volume(start: int = 1, end: int = 100) -> list:
    url = "https://data-api.binance.vision/api/v3/ticker/24hr"
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            data = resp.json()
            break
        except Exception:
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
            else:
                return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

    usdt_pairs = [
        item for item in data
        if item['symbol'].endswith('USDT')
        and float(item.get('quoteVolume', 0)) > 0
    ]
    usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)

    total = len(usdt_pairs)
    s = max(start - 1, 0)
    e = min(end, total)
    return [item['symbol'] for item in usdt_pairs[s:e]]