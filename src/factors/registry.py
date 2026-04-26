"""
因子注册中心 - 管理因子的注册、发现、实例化
"""
from typing import Dict, Type, Optional, List
from src.factors.base import FactorBase


class FactorRegistry:
    """因子注册中心（全局单例模式）"""

    _registry: Dict[str, Type[FactorBase]] = {}

    @classmethod
    def register(cls, name: str, factor_class: Type[FactorBase]) -> None:
        """注册因子"""
        if not issubclass(factor_class, FactorBase):
            raise TypeError(f"{factor_class} 必须是 FactorBase 的子类")
        cls._registry[name] = factor_class

    @classmethod
    def get(cls, name: str, **kwargs) -> Optional[FactorBase]:
        """获取因子实例"""
        factor_class = cls._registry.get(name)
        if factor_class is None:
            return None
        return factor_class(**kwargs)

    @classmethod
    def list_factors(cls) -> List[str]:
        """列出所有已注册的因子名"""
        return list(cls._registry.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """按类别筛选因子"""
        result = []
        for name, factor_class in cls._registry.items():
            instance = factor_class()
            if instance.category == category:
                result.append(name)
        return result

    @classmethod
    def info(cls, name: str) -> Optional[dict]:
        """获取因子元信息"""
        factor_class = cls._registry.get(name)
        if factor_class is None:
            return None
        instance = factor_class()
        return {
            'name': instance.name,
            'description': instance.description,
            'category': instance.category,
            'lookback': instance.lookback,
            'params': instance.params,
        }

    @classmethod
    def info_all(cls) -> List[dict]:
        """获取所有因子的元信息"""
        return [cls.info(name) for name in cls._registry]

    @classmethod
    def unregister(cls, name: str) -> None:
        """注销因子"""
        cls._registry.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        """清空所有已注册的因子"""
        cls._registry.clear()
