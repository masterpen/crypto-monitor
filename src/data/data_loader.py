"""数据加载器模块"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List


class DataLoader:
    """数据加载器基类"""

    @staticmethod
    def load_csv(filepath: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
        """从CSV文件加载数据"""
        df = pd.read_csv(filepath)
        if parse_dates:
            for col in parse_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        return df

    @staticmethod
    def load_csv_with_index(filepath: str, index_col: str = 0) -> pd.DataFrame:
        """从CSV文件加载数据（带索引）"""
        df = pd.read_csv(filepath, index_col=index_col, parse_dates=True)
        return df

    @staticmethod
    def resample_klines(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        重采样K线数据

        Args:
            data: 原始K线数据
            timeframe: 目标时间周期，如 '1h', '4h', '1d'

        Returns:
            重采样后的数据
        """
        if 'close' not in data.columns:
            raise ValueError("数据必须包含 'close' 列")

        # 设置时间索引
        if isinstance(data.index, pd.DatetimeIndex):
            df = data.copy()
        else:
            df = data.set_index(pd.to_datetime(data.iloc[:, 0]))

        # 定义聚合规则
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        # 只保留存在的列
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

        # 重采样
        resampled = df.resample(timeframe).agg(agg_dict).dropna()
        return resampled

    @staticmethod
    def generate_sample_data(
        days: int = 30,
        initial_price: float = 40000,
        volatility: float = 0.02
    ) -> pd.DataFrame:
        """
        生成模拟K线数据用于测试

        Args:
            days: 天数
            initial_price: 初始价格
            volatility: 波动率

        Returns:
            模拟K线数据
        """
        hours = days * 24
        dates = pd.date_range(end=datetime.now(), periods=hours, freq='1h')

        # 生成随机价格变动
        returns = np.random.randn(hours) * volatility
        price_series = initial_price * np.exp(np.cumsum(returns))

        # 生成OHLC数据
        data = pd.DataFrame({
            'open': price_series * (1 + np.random.randn(hours) * 0.005),
            'high': price_series * (1 + np.abs(np.random.randn(hours)) * 0.01),
            'low': price_series * (1 - np.abs(np.random.randn(hours)) * 0.01),
            'close': price_series,
            'volume': np.random.rand(hours) * 100
        }, index=dates)

        # 确保high >= open, close, low
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)

        return data


class DataCache:
    """数据缓存"""

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None

    def set(self, key: str, data: pd.DataFrame):
        """设置缓存数据"""
        if len(self.cache) >= self.max_size:
            # 删除最久未访问的数据
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = data
        self.access_times[key] = datetime.now()

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()