"""
量化交易系统日志配置模块
提供统一的日志配置和管理
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class TradingSystemLogger:
    """交易系统日志管理器"""
    
    _instance: Optional['TradingSystemLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not TradingSystemLogger._initialized:
            self.loggers: Dict[str, logging.Logger] = {}
            self.log_dir: Optional[Path] = None
            self.default_level: int = logging.INFO
            TradingSystemLogger._initialized = True
    
    def setup(
        self,
        log_dir: str = "logs",
        default_level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        log_format: Optional[str] = None
    ) -> None:
        """
        设置日志配置
        
        Args:
            log_dir: 日志目录
            default_level: 默认日志级别
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
            max_file_size: 单个日志文件最大大小（字节）
            backup_count: 保留的备份文件数量
            log_format: 日志格式
        """
        self.log_dir = Path(log_dir)
        self.default_level = default_level
        
        # 创建日志目录
        if file_output:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置默认格式
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(default_level)
        
        # 清除现有的处理器
        root_logger.handlers.clear()
        
        # 添加控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(default_level)
            console_formatter = logging.Formatter(log_format)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # 添加文件处理器
        if file_output and self.log_dir:
            # 主日志文件
            main_log_file = self.log_dir / "quant_trading.log"
            file_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(default_level)
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
            # 错误日志文件
            error_log_file = self.log_dir / "error.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_formatter = logging.Formatter(log_format)
            error_handler.setFormatter(error_formatter)
            root_logger.addHandler(error_handler)
            
            # 交易日志文件
            trade_log_file = self.log_dir / "trades.log"
            trade_handler = logging.handlers.RotatingFileHandler(
                trade_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            trade_handler.setLevel(logging.INFO)
            trade_formatter = logging.Formatter(log_format)
            trade_handler.setFormatter(trade_formatter)
            
            # 创建交易专用日志记录器
            trade_logger = logging.getLogger('trades')
            trade_logger.addHandler(trade_handler)
            trade_logger.setLevel(logging.INFO)
    
    def get_logger(self, name: str, level: Optional[int] = None) -> logging.Logger:
        """
        获取日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别（可选）
            
        Returns:
            日志记录器实例
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(level or self.default_level)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def get_trade_logger(self) -> logging.Logger:
        """获取交易专用日志记录器"""
        return logging.getLogger('trades')
    
    def set_level(self, name: str, level: int) -> None:
        """
        设置日志记录器级别
        
        Args:
            name: 日志记录器名称
            level: 日志级别
        """
        if name in self.loggers:
            self.loggers[name].setLevel(level)
    
    def add_file_handler(
        self,
        logger_name: str,
        filename: str,
        level: Optional[int] = None,
        max_file_size: int = 10 * 1024 * 1024,
        backup_count: int = 5
    ) -> None:
        """
        为指定日志记录器添加文件处理器
        
        Args:
            logger_name: 日志记录器名称
            filename: 文件名
            level: 日志级别
            max_file_size: 单个日志文件最大大小
            backup_count: 保留的备份文件数量
        """
        if logger_name not in self.loggers:
            return
        
        logger = self.loggers[logger_name]
        
        if self.log_dir:
            log_file = self.log_dir / filename
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            handler.setLevel(level or self.default_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)


# 全局日志管理器实例
logger_manager = TradingSystemLogger()


def setup_logging(
    log_dir: str = "logs",
    default_level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True
) -> None:
    """
    设置日志配置的便捷函数
    
    Args:
        log_dir: 日志目录
        default_level: 默认日志级别
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
    """
    logger_manager.setup(
        log_dir=log_dir,
        default_level=default_level,
        console_output=console_output,
        file_output=file_output
    )


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    获取日志记录器的便捷函数
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        
    Returns:
        日志记录器实例
    """
    return logger_manager.get_logger(name, level)


def get_trade_logger() -> logging.Logger:
    """获取交易专用日志记录器的便捷函数"""
    return logger_manager.get_trade_logger()


class LoggerMixin:
    """日志记录器混入类"""
    
    @property
    def logger(self) -> logging.Logger:
        """获取当前类的日志记录器"""
        return get_logger(self.__class__.__name__)


def log_function_call(func):
    """函数调用日志装饰器"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"调用函数: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {e}")
            raise
    return wrapper


def log_execution_time(func):
    """执行时间日志装饰器"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        logger.debug(f"开始执行: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.debug(f"函数 {func.__name__} 执行完成，耗时: {duration:.3f}秒")
            return result
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"函数 {func.__name__} 执行失败，耗时: {duration:.3f}秒，错误: {e}")
            raise
    return wrapper