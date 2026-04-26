import copy
import pandas as pd
import numpy as np


def make_ohlcv_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """生成随机 OHLCV 数据用于测试"""
    rng = np.random.default_rng(seed)

    trends = rng.choice([-1, 0, 1], size=max(n // 40, 1))
    days = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='h')

    close = np.empty(n)
    close[0] = 100.0
    for i in range(1, n):
        trend_section = i // 40
        trend = trends[trend_section % len(trends)]
        change = rng.normal(trend * 0.5, 2.0)
        close[i] = close[i - 1] + change
        close[i] = max(close[i], 10.0)

    high = close + rng.uniform(0.5, 3, size=n)
    low = np.minimum(close - rng.uniform(0.5, 3, size=n), close - 0.01)
    open_ = low + rng.uniform(0, 1, size=n) * (high - low)
    volume = rng.uniform(100, 1000, size=n)

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=days)
    return df


def make_uptrend_data(n: int = 200) -> pd.DataFrame:
    """生成上升趋势数据"""
    rng = np.random.default_rng(99)
    days = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='h')
    base = np.linspace(100, 150, n)
    noise = rng.normal(0, 1.5, size=n)
    close = base + noise
    close = np.maximum(close, 10)

    high = close + rng.uniform(0.5, 2, size=n)
    low = np.minimum(close - rng.uniform(0.5, 2, size=n), close - 0.01)
    open_ = low + rng.uniform(0, 1, size=n) * (high - low)
    volume = rng.uniform(100, 1000, size=n)

    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=days)


def make_downtrend_data(n: int = 200) -> pd.DataFrame:
    """生成下降趋势数据"""
    rng = np.random.default_rng(77)
    days = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='h')
    base = np.linspace(150, 100, n)
    noise = rng.normal(0, 1.5, size=n)
    close = base + noise
    close = np.maximum(close, 10)

    high = close + rng.uniform(0.5, 2, size=n)
    low = np.minimum(close - rng.uniform(0.5, 2, size=n), close - 0.01)
    open_ = low + rng.uniform(0, 1, size=n) * (high - low)
    volume = rng.uniform(100, 1000, size=n)

    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=days)