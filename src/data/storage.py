"""
数据存储管理模块
支持多种存储后端（内存、文件、Redis、PostgreSQL）
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path

from ..exceptions import DataError, ConfigurationError

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """存储后端基类"""
    
    @abstractmethod
    def save(self, key: str, data: pd.DataFrame, metadata: Optional[Dict] = None) -> bool:
        """
        保存数据
        
        Args:
            key: 数据键
            data: 数据
            metadata: 元数据
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def load(self, key: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        加载数据
        
        Args:
            key: 数据键
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            数据
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        删除数据
        
        Args:
            key: 数据键
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        检查数据是否存在
        
        Args:
            key: 数据键
            
        Returns:
            是否存在
        """
        pass
    
    @abstractmethod
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        列出所有键
        
        Args:
            pattern: 匹配模式
            
        Returns:
            键列表
        """
        pass
    
    @abstractmethod
    def get_metadata(self, key: str) -> Optional[Dict]:
        """
        获取元数据
        
        Args:
            key: 数据键
            
        Returns:
            元数据
        """
        pass


class MemoryStorage(StorageBackend):
    """内存存储后端"""
    
    def __init__(self):
        self.storage: Dict[str, pd.DataFrame] = {}
        self.metadata_storage: Dict[str, Dict] = {}
    
    def save(self, key: str, data: pd.DataFrame, metadata: Optional[Dict] = None) -> bool:
        """保存数据到内存"""
        try:
            self.storage[key] = data.copy()
            if metadata:
                self.metadata_storage[key] = metadata.copy()
            return True
        except Exception as e:
            logger.error(f"保存数据失败: {key}, 错误: {e}")
            return False
    
    def load(self, key: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """从内存加载数据"""
        if key not in self.storage:
            return None
        
        data = self.storage[key].copy()
        
        # 按日期过滤
        if start_date or end_date:
            if isinstance(data.index, pd.DatetimeIndex):
                if start_date:
                    data = data[data.index >= start_date]
                if end_date:
                    data = data[data.index <= end_date]
        
        return data
    
    def delete(self, key: str) -> bool:
        """从内存删除数据"""
        if key in self.storage:
            del self.storage[key]
            self.metadata_storage.pop(key, None)
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """检查数据是否存在于内存"""
        return key in self.storage
    
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """列出内存中的所有键"""
        keys = list(self.storage.keys())
        
        if pattern:
            import fnmatch
            keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        
        return keys
    
    def get_metadata(self, key: str) -> Optional[Dict]:
        """获取元数据"""
        return self.metadata_storage.get(key)


class FileStorage(StorageBackend):
    """文件存储后端"""
    
    def __init__(self, base_dir: str = "data/storage"):
        """
        初始化文件存储
        
        Args:
            base_dir: 基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, key: str) -> Path:
        """获取文件路径"""
        # 将键转换为安全的文件名
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.base_dir / f"{safe_key}.parquet"
    
    def _get_metadata_path(self, key: str) -> Path:
        """获取元数据文件路径"""
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.base_dir / f"{safe_key}_metadata.json"
    
    def save(self, key: str, data: pd.DataFrame, metadata: Optional[Dict] = None) -> bool:
        """保存数据到文件"""
        try:
            file_path = self._get_file_path(key)
            data.to_parquet(file_path)
            
            # 保存元数据
            if metadata:
                metadata_path = self._get_metadata_path(key)
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"保存数据失败: {key}, 错误: {e}")
            return False
    
    def load(self, key: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """从文件加载数据"""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            return None
        
        try:
            data = pd.read_parquet(file_path)
            
            # 按日期过滤
            if start_date or end_date:
                if isinstance(data.index, pd.DatetimeIndex):
                    if start_date:
                        data = data[data.index >= start_date]
                    if end_date:
                        data = data[data.index <= end_date]
            
            return data
        except Exception as e:
            logger.error(f"加载数据失败: {key}, 错误: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """删除文件"""
        try:
            file_path = self._get_file_path(key)
            metadata_path = self._get_metadata_path(key)
            
            if file_path.exists():
                file_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            return True
        except Exception as e:
            logger.error(f"删除数据失败: {key}, 错误: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查文件是否存在"""
        file_path = self._get_file_path(key)
        return file_path.exists()
    
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """列出所有键"""
        keys = []
        
        for file_path in self.base_dir.glob("*.parquet"):
            key = file_path.stem
            if not key.endswith("_metadata"):
                keys.append(key)
        
        if pattern:
            import fnmatch
            keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        
        return keys
    
    def get_metadata(self, key: str) -> Optional[Dict]:
        """获取元数据"""
        metadata_path = self._get_metadata_path(key)
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取元数据失败: {key}, 错误: {e}")
            return None


class RedisStorage(StorageBackend):
    """Redis存储后端"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        """
        初始化Redis存储
        
        Args:
            host: Redis主机
            port: Redis端口
            db: 数据库编号
            password: 密码
        """
        try:
            import redis
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False
            )
            # 测试连接
            self.redis_client.ping()
            logger.info(f"Redis连接成功: {host}:{port}")
        except ImportError:
            raise ConfigurationError("Redis库未安装，请运行: pip install redis")
        except Exception as e:
            raise ConfigurationError(f"Redis连接失败: {e}")
    
    def save(self, key: str, data: pd.DataFrame, metadata: Optional[Dict] = None) -> bool:
        """保存数据到Redis"""
        try:
            # 将DataFrame转换为JSON
            data_json = data.to_json(date_format='iso')
            
            # 保存数据
            self.redis_client.set(f"data:{key}", data_json)
            
            # 保存元数据
            if metadata:
                metadata_json = json.dumps(metadata, ensure_ascii=False)
                self.redis_client.set(f"metadata:{key}", metadata_json)
            
            # 设置过期时间（7天）
            self.redis_client.expire(f"data:{key}", 7 * 24 * 3600)
            if metadata:
                self.redis_client.expire(f"metadata:{key}", 7 * 24 * 3600)
            
            return True
        except Exception as e:
            logger.error(f"保存数据到Redis失败: {key}, 错误: {e}")
            return False
    
    def load(self, key: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """从Redis加载数据"""
        try:
            data_json = self.redis_client.get(f"data:{key}")
            
            if data_json is None:
                return None
            
            data = pd.read_json(data_json)
            
            # 按日期过滤
            if start_date or end_date:
                if isinstance(data.index, pd.DatetimeIndex):
                    if start_date:
                        data = data[data.index >= start_date]
                    if end_date:
                        data = data[data.index <= end_date]
            
            return data
        except Exception as e:
            logger.error(f"从Redis加载数据失败: {key}, 错误: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """从Redis删除数据"""
        try:
            self.redis_client.delete(f"data:{key}")
            self.redis_client.delete(f"metadata:{key}")
            return True
        except Exception as e:
            logger.error(f"从Redis删除数据失败: {key}, 错误: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查数据是否存在于Redis"""
        try:
            return self.redis_client.exists(f"data:{key}") > 0
        except Exception as e:
            logger.error(f"检查Redis数据失败: {key}, 错误: {e}")
            return False
    
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """列出Redis中的所有键"""
        try:
            if pattern:
                search_pattern = f"data:{pattern}"
            else:
                search_pattern = "data:*"
            
            keys = []
            for key in self.redis_client.scan_iter(match=search_pattern):
                # 移除前缀 "data:"
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                keys.append(key_str[5:])  # 移除 "data:" 前缀
            
            return keys
        except Exception as e:
            logger.error(f"列出Redis键失败: {e}")
            return []
    
    def get_metadata(self, key: str) -> Optional[Dict]:
        """获取元数据"""
        try:
            metadata_json = self.redis_client.get(f"metadata:{key}")
            
            if metadata_json is None:
                return None
            
            return json.loads(metadata_json)
        except Exception as e:
            logger.error(f"获取Redis元数据失败: {key}, 错误: {e}")
            return None


class StorageManager:
    """存储管理器"""
    
    def __init__(self, default_backend: str = "memory", **kwargs):
        """
        初始化存储管理器
        
        Args:
            default_backend: 默认存储后端 ('memory', 'file', 'redis')
            **kwargs: 后端配置参数
        """
        self.backends: Dict[str, StorageBackend] = {}
        self.default_backend_name = default_backend
        
        # 初始化默认后端
        if default_backend == "memory":
            self.backends["memory"] = MemoryStorage()
        elif default_backend == "file":
            base_dir = kwargs.get("base_dir", "data/storage")
            self.backends["file"] = FileStorage(base_dir)
        elif default_backend == "redis":
            self.backends["redis"] = RedisStorage(**kwargs)
        else:
            raise ConfigurationError(f"不支持的存储后端: {default_backend}")
    
    def add_backend(self, name: str, backend: StorageBackend) -> None:
        """添加存储后端"""
        self.backends[name] = backend
    
    def get_backend(self, name: Optional[str] = None) -> StorageBackend:
        """获取存储后端"""
        backend_name = name or self.default_backend_name
        
        if backend_name not in self.backends:
            raise ConfigurationError(f"存储后端不存在: {backend_name}")
        
        return self.backends[backend_name]
    
    def save(
        self,
        key: str,
        data: pd.DataFrame,
        metadata: Optional[Dict] = None,
        backend: Optional[str] = None
    ) -> bool:
        """
        保存数据
        
        Args:
            key: 数据键
            data: 数据
            metadata: 元数据
            backend: 存储后端名称
            
        Returns:
            是否成功
        """
        storage = self.get_backend(backend)
        return storage.save(key, data, metadata)
    
    def load(
        self,
        key: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        backend: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        加载数据
        
        Args:
            key: 数据键
            start_date: 开始日期
            end_date: 结束日期
            backend: 存储后端名称
            
        Returns:
            数据
        """
        storage = self.get_backend(backend)
        return storage.load(key, start_date, end_date)
    
    def delete(self, key: str, backend: Optional[str] = None) -> bool:
        """
        删除数据
        
        Args:
            key: 数据键
            backend: 存储后端名称
            
        Returns:
            是否成功
        """
        storage = self.get_backend(backend)
        return storage.delete(key)
    
    def exists(self, key: str, backend: Optional[str] = None) -> bool:
        """
        检查数据是否存在
        
        Args:
            key: 数据键
            backend: 存储后端名称
            
        Returns:
            是否存在
        """
        storage = self.get_backend(backend)
        return storage.exists(key)
    
    def list_keys(self, pattern: Optional[str] = None, backend: Optional[str] = None) -> List[str]:
        """
        列出所有键
        
        Args:
            pattern: 匹配模式
            backend: 存储后端名称
            
        Returns:
            键列表
        """
        storage = self.get_backend(backend)
        return storage.list_keys(pattern)
    
    def get_metadata(self, key: str, backend: Optional[str] = None) -> Optional[Dict]:
        """
        获取元数据
        
        Args:
            key: 数据键
            backend: 存储后端名称
            
        Returns:
            元数据
        """
        storage = self.get_backend(backend)
        return storage.get_metadata(key)
    
    def migrate(self, source_key: str, target_key: str, source_backend: Optional[str] = None, target_backend: Optional[str] = None) -> bool:
        """
        迁移数据
        
        Args:
            source_key: 源键
            target_key: 目标键
            source_backend: 源后端
            target_backend: 目标后端
            
        Returns:
            是否成功
        """
        try:
            # 从源加载数据
            data = self.load(source_key, backend=source_backend)
            if data is None:
                logger.warning(f"源数据不存在: {source_key}")
                return False
            
            # 获取元数据
            metadata = self.get_metadata(source_key, backend=source_backend)
            
            # 保存到目标
            return self.save(target_key, data, metadata, backend=target_backend)
        except Exception as e:
            logger.error(f"数据迁移失败: {source_key} -> {target_key}, 错误: {e}")
            return False


# 全局存储管理器实例
_default_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """获取全局存储管理器"""
    global _default_storage_manager
    if _default_storage_manager is None:
        _default_storage_manager = StorageManager()
    return _default_storage_manager


def set_storage_manager(manager: StorageManager) -> None:
    """设置全局存储管理器"""
    global _default_storage_manager
    _default_storage_manager = manager