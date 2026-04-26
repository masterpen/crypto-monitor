import pandas as pd
from typing import Tuple


def calculate_kdj(data: pd.DataFrame, period: int = 9,
                  k_period: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    low_min = data['low'].rolling(window=period).min()
    high_max = data['high'].rolling(window=period).max()
    diff = (high_max - low_min).replace(0, 1e-10)
    rsv = (data['close'] - low_min) / diff * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(com=k_period - 1, adjust=False).mean()
    d = k.ewm(com=d_period - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    avg_gain = gains.ewm(span=period).mean()
    avg_loss = losses.ewm(span=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def calculate_macd(prices: pd.Series, fast_period: int = 12,
                   slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = prices.ewm(span=fast_period).mean()
    ema_slow = prices.ewm(span=slow_period).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period).mean()
    histogram = macd - signal
    return macd, signal, histogram