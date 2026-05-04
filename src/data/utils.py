"""
K线数据获取与本地缓存（通过 binance.vision 公开接口，无需 API Key）

缓存策略：
- 按 symbol_interval.csv 分文件存储于 data/cache/ 目录
- 首次拉取写文件，后续读缓存
- 增量更新：记录文件最新时间戳，只拉取新增部分追加
- 自动去重：追加后按时间戳去重，保留最新值
"""
import os
import requests
import time
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from ..exceptions import DataNotFoundError, DataValidationError, APIError, ConnectionError, TimeoutError

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
_BINANCE_KLINE_URL = "https://data-api.binance.vision/api/v3/klines"
_BINANCE_TICKER_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"

_COLUMNS_RAW = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
    'taker_buy_quote', 'ignore'
]
_OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']


def _cache_path(symbol: str, interval: str) -> Path:
    """
    获取缓存文件路径
    
    Args:
        symbol: 交易对
        interval: K线周期
        
    Returns:
        缓存文件路径
    """
    return _CACHE_DIR / f"{symbol}_{interval}.csv"


def _raw_to_df(raw_batches: list) -> pd.DataFrame:
    """
    将原始数据转换为DataFrame
    
    Args:
        raw_batches: 原始数据批次
        
    Returns:
        转换后的DataFrame
    """
    df: pd.DataFrame = pd.DataFrame(raw_batches, columns=_COLUMNS_RAW)
    for col in _OHLCV_COLS:
        df[col] = df[col].astype(float)
    df.index = pd.to_datetime(df['open_time'], unit='ms')
    return df[_OHLCV_COLS]


def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    原子化写 CSV：先写临时文件再重命名，避免多进程写入冲突
    
    Args:
        df: 要写入的DataFrame
        path: 目标文件路径
    """
    fd, tmp = tempfile.mkstemp(suffix='.csv', dir=path.parent)
    try:
        os.close(fd)
        df.to_csv(tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _fetch_raw(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """
    获取原始K线数据
    
    Args:
        symbol: 交易对
        interval: K线周期
        start_ms: 开始时间戳（毫秒）
        end_ms: 结束时间戳（毫秒）
        
    Returns:
        原始数据列表
    """
    all_data: list = []
    current_start: int = start_ms

    while current_start < end_ms:
        params: dict = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': 1000
        }
        batch: Optional[list] = None
        last_exception: Optional[Exception] = None
        
        for attempt in range(3):
            try:
                resp = requests.get(_BINANCE_KLINE_URL, params=params, timeout=15)
                resp.raise_for_status()  # 检查HTTP状态码
                batch = resp.json()
                break
            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"请求超时 (尝试 {attempt + 1}/3): {symbol} {interval}")
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(f"连接错误 (尝试 {attempt + 1}/3): {symbol} {interval}")
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
            except requests.exceptions.HTTPError as e:
                last_exception = e
                logger.warning(f"HTTP错误 (尝试 {attempt + 1}/3): {symbol} {interval} - {e}")
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
            except Exception as e:
                last_exception = e
                logger.warning(f"未知错误 (尝试 {attempt + 1}/3): {symbol} {interval} - {e}")
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
        
        if batch is None:
            raise ConnectionError(
                message=f"无法获取K线数据: {symbol} {interval}",
                host=_BINANCE_KLINE_URL,
                details={"symbol": symbol, "interval": interval, "last_error": str(last_exception)}
            )
        
        if not isinstance(batch, list):
            raise DataValidationError(
                f"API返回数据格式错误: {symbol} {interval}",
                validation_errors=["返回数据不是列表格式"]
            )
        
        if len(batch) == 0:
            break

        all_data.extend(batch)
        current_start = batch[-1][0] + 1
        if len(batch) < 1000:
            break

    return all_data


def fetch_klines(symbol: str, interval: str = '1h', days: int = 90,
                 use_cache: bool = True) -> pd.DataFrame:
    """
    获取历史K线数据，支持本地缓存。

    Args:
        symbol: 交易对，如 BTCUSDT
        interval: K线周期，如 1h/4h/1d
        days: 需要的天数
        use_cache: 是否使用本地缓存（默认开启）
        
    Returns:
        K线数据DataFrame
        
    Raises:
        DataNotFoundError: 未找到数据
        DataValidationError: 数据验证失败
        ConnectionError: 网络连接错误
        TimeoutError: 请求超时
    """
    if days <= 0:
        raise DataValidationError(
            f"天数必须为正数，当前值: {days}",
            validation_errors=["days参数无效"]
        )
    
    end_ms: int = int(datetime.now().timestamp() * 1000)
    start_ms: int = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    if not use_cache:
        return _raw_to_df(_fetch_raw(symbol, interval, start_ms, end_ms))

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file: Path = _cache_path(symbol, interval)

    try:
        if cache_file.exists():
            cached: pd.DataFrame = pd.read_csv(cache_file, index_col=0, parse_dates=True)

            if cached.empty:
                result: pd.DataFrame = _raw_to_df(_fetch_raw(symbol, interval, start_ms, end_ms))
                _safe_write_csv(result, cache_file)
                return result

            latest_ts = cached.index.max()
            required_start = datetime.fromtimestamp(start_ms / 1000, tz=latest_ts.tz)

            # 缓存已经覆盖所需时间范围
            if latest_ts >= required_start and cached.index.min() <= required_start:
                result = cached[cached.index >= required_start].copy()
                if not isinstance(result.index, pd.DatetimeIndex):
                    result.index = pd.to_datetime(result.index)
                return result

            # 增量拉取：只拉缓存最新时间到当前时间的K线
            fetch_start_ms: int = int(latest_ts.timestamp() * 1000) + 1
            if fetch_start_ms < end_ms:
                logger.info(f"{symbol} {interval} 增量更新: {latest_ts} -> {datetime.now()}")
                new_data: list = _fetch_raw(symbol, interval, fetch_start_ms, end_ms)
                if new_data:
                    new_df: pd.DataFrame = _raw_to_df(new_data)
                    cached = pd.concat([cached, new_df])
                    cached = cached[~cached.index.duplicated(keep='last')]
                    cached.sort_index(inplace=True)
                    _safe_write_csv(cached, cache_file)
        else:
            cached = pd.DataFrame()

        # 检查缓存是否仍不满足范围
        if cached.empty or cached.index.max() < datetime.fromtimestamp(start_ms / 1000, tz=cached.index.tz):
            logger.info(f"{symbol} {interval} 缓存不足，全量拉取 {days} 天")
            result = _raw_to_df(_fetch_raw(symbol, interval, start_ms, end_ms))
            _safe_write_csv(result, cache_file)
            return result

        result = cached[cached.index >= datetime.fromtimestamp(start_ms / 1000, tz=cached.index.tz)].copy()
        return result
    except Exception as e:
        if isinstance(e, (DataNotFoundError, DataValidationError, ConnectionError, TimeoutError)):
            raise
        raise DataNotFoundError(
            f"获取K线数据失败: {symbol} {interval}",
            symbol=symbol,
            timeframe=interval,
            details={"original_error": str(e)}
        )


def wipe_cache(symbol: Optional[str] = None, interval: Optional[str] = None) -> None:
    """
    清除本地缓存。

    Args:
        symbol: 指定交易对（None = 全部）
        interval: 指定周期（None = 全部）
    """
    if not _CACHE_DIR.exists():
        return

    if symbol and interval:
        f: Path = _cache_path(symbol, interval)
        if f.exists():
            f.unlink()
    else:
        pattern: str = f"*_{interval}.csv" if interval else "*.csv"
        if symbol:
            pattern = f"{symbol}_*.csv"
        for f in _CACHE_DIR.glob(pattern):
            f.unlink()
    logger.info(f"缓存已清除")


def cache_stats() -> Dict[str, Any]:
    """
    返回各缓存文件的大小统计
    
    Returns:
        缓存统计信息字典
    """
    if not _CACHE_DIR.exists():
        return {'files': 0, 'total_size_mb': 0, 'details': []}

    details: List[Dict[str, Any]] = []
    total: int = 0
    for f in sorted(_CACHE_DIR.glob("*.csv")):
        size: int = f.stat().st_size
        total += size
        parts: List[str] = f.stem.rsplit('_', 1)
        details.append({
            'file': f.name,
            'symbol': parts[0],
            'interval': parts[1] if len(parts) == 2 else '',
            'size_kb': round(size / 1024, 1)
        })

    return {
        'files': len(details),
        'total_size_mb': round(total / (1024 * 1024), 2),
        'details': details
    }


def get_top_symbols_by_volume(start: int = 1, end: int = 100) -> List[str]:
    """
    获取按交易量排序的前N个交易对
    
    Args:
        start: 开始排名（从1开始）
        end: 结束排名
        
    Returns:
        交易对列表
        
    Raises:
        ConnectionError: 网络连接错误
        TimeoutError: 请求超时
        DataValidationError: 参数验证失败
    """
    if start < 1:
        raise DataValidationError(
            f"start必须大于等于1，当前值: {start}",
            validation_errors=["start参数无效"]
        )
    if end < start:
        raise DataValidationError(
            f"end必须大于等于start，当前值: start={start}, end={end}",
            validation_errors=["end参数无效"]
        )
    
    url: str = _BINANCE_TICKER_URL
    last_exception: Optional[Exception] = None
    data: Optional[list] = None
    
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.Timeout as e:
            last_exception = e
            logger.warning(f"获取交易对列表超时 (尝试 {attempt + 1}/3)")
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            logger.warning(f"获取交易对列表连接错误 (尝试 {attempt + 1}/3)")
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
        except requests.exceptions.HTTPError as e:
            last_exception = e
            logger.warning(f"获取交易对列表HTTP错误 (尝试 {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
        except Exception as e:
            last_exception = e
            logger.warning(f"获取交易对列表未知错误 (尝试 {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
    
    if data is None:
        logger.error(f"无法获取交易对列表，使用默认列表: {last_exception}")
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

    usdt_pairs: List[Dict[str, Any]] = [
        item for item in data
        if item['symbol'].endswith('USDT')
        and float(item.get('quoteVolume', 0)) > 0
    ]
    usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)

    total: int = len(usdt_pairs)
    s: int = max(start - 1, 0)
    e: int = min(end, total)
    return [item['symbol'] for item in usdt_pairs[s:e]]