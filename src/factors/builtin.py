"""
内置因子库 - 提供常用技术因子，启动时自动注册到 FactorRegistry

因子列表:
  volume_ratio    - 成交量比率（当前量/前一根量）
  volume_ma_ratio - 成交量/MA量比
  price_momentum  - 价格动量（N周期涨跌幅）
  volatility      - 波动率（收益率标准差）
  rsi             - RSI因子
  kdj_j           - KDJ-J值因子
  bollinger_pos   - 布林带位置因子（0=下轨, 1=上轨）
  obv_slope       - OBV斜率因子
  atr_ratio       - ATR/价格比率
"""
import pandas as pd
import numpy as np
from src.factors.base import FactorBase
from src.factors.registry import FactorRegistry


# ═══════════════════════════════════════════════════════════
#  成交量因子
# ═══════════════════════════════════════════════════════════

class VolumeRatioFactor(FactorBase):
    """成交量比率：当前K线成交量 / 前一根K线成交量"""
    name = "volume_ratio"
    description = "成交量比率（当前/前一根）"
    category = "volume"
    lookback = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        vr = data['volume'] / data['volume'].shift(1)
        return vr.replace([np.inf, -np.inf], np.nan)


class VolumeMARatioFactor(FactorBase):
    """成交量/MA量比：当前成交量 / N周期成交量均值"""
    name = "volume_ma_ratio"
    description = "成交量/N周期均量比"
    category = "volume"
    lookback = 20

    def __init__(self, period: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.lookback = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        vol_ma = data['volume'].rolling(window=self.period).mean()
        return (data['volume'] / vol_ma).replace([np.inf, -np.inf], np.nan)


# ═══════════════════════════════════════════════════════════
#  动量因子
# ═══════════════════════════════════════════════════════════

class PriceMomentumFactor(FactorBase):
    """价格动量：N周期涨跌幅"""
    name = "price_momentum"
    description = "价格N周期动量（涨跌幅）"
    category = "momentum"
    lookback = 20

    def __init__(self, period: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.lookback = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['close'].pct_change(self.period)


# ═══════════════════════════════════════════════════════════
#  波动率因子
# ═══════════════════════════════════════════════════════════

class VolatilityFactor(FactorBase):
    """波动率：N周期收益率标准差"""
    name = "volatility"
    description = "N周期波动率"
    category = "volatility"
    lookback = 20

    def __init__(self, period: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.lookback = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        returns = data['close'].pct_change()
        return returns.rolling(window=self.period).std()


class ATRRatioFactor(FactorBase):
    """ATR/价格比率：衡量相对波动程度"""
    name = "atr_ratio"
    description = "ATR/价格比率"
    category = "volatility"
    lookback = 14

    def __init__(self, period: int = 14, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.lookback = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift(1)).abs()
        low_close = (data['low'] - data['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        return (atr / data['close']).replace([np.inf, -np.inf], np.nan)


# ═══════════════════════════════════════════════════════════
#  均值回归因子
# ═══════════════════════════════════════════════════════════

class RSIFactor(FactorBase):
    """RSI因子"""
    name = "rsi"
    description = "RSI相对强弱指标"
    category = "mean_reversion"
    lookback = 14

    def __init__(self, period: int = 14, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.lookback = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(span=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=self.period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


class KDJJFactor(FactorBase):
    """KDJ-J值因子"""
    name = "kdj_j"
    description = "KDJ指标J值"
    category = "mean_reversion"
    lookback = 9

    def __init__(self, period: int = 9, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.lookback = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        low_min = data['low'].rolling(window=self.period).min()
        high_max = data['high'].rolling(window=self.period).max()
        diff = (high_max - low_min).replace(0, 1e-10)
        rsv = (data['close'] - low_min) / diff * 100
        rsv = rsv.fillna(50)
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        return j


class BollingerPositionFactor(FactorBase):
    """布林带位置因子：0=下轨, 0.5=中轨, 1=上轨"""
    name = "bollinger_position"
    description = "布林带位置（0~1）"
    category = "mean_reversion"
    lookback = 20

    def __init__(self, period: int = 20, std_mult: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.std_mult = std_mult
        self.lookback = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        mid = data['close'].rolling(window=self.period).mean()
        std = data['close'].rolling(window=self.period).std()
        upper = mid + self.std_mult * std
        lower = mid - self.std_mult * std
        return ((data['close'] - lower) / (upper - lower)).replace([np.inf, -np.inf], np.nan).clip(0, 1)


class OBVSlopeFactor(FactorBase):
    """OBV斜率因子：量价趋势强度"""
    name = "obv_slope"
    description = "OBV斜率（量价趋势）"
    category = "volume"
    lookback = 20

    def __init__(self, period: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.lookback = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        direction = np.where(data['close'] >= data['close'].shift(1), 1, -1)
        obv = (data['volume'] * direction).cumsum()
        # 计算OBV的N周期变化率（斜率近似）
        obv_slope = (obv - obv.shift(self.period)) / obv.shift(self.period).replace(0, np.nan)
        return obv_slope.replace([np.inf, -np.inf], np.nan)


# ═══════════════════════════════════════════════════════════
#  自动注册所有内置因子
# ═══════════════════════════════════════════════════════════

BUILTIN_FACTORS = [
    VolumeRatioFactor,
    VolumeMARatioFactor,
    PriceMomentumFactor,
    VolatilityFactor,
    ATRRatioFactor,
    RSIFactor,
    KDJJFactor,
    BollingerPositionFactor,
    OBVSlopeFactor,
]


def register_builtin_factors():
    """注册所有内置因子"""
    for factor_class in BUILTIN_FACTORS:
        instance = factor_class()
        FactorRegistry.register(instance.name, factor_class)


# 模块加载时自动注册
register_builtin_factors()
