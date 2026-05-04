"""量化交易系统"""
__version__ = "1.0.0"

from .exceptions import (
    QuantTradingError,
    DataError,
    APIError,
    DataNotFoundError,
    DataValidationError,
    StrategyError,
    StrategyParameterError,
    SignalGenerationError,
    BacktestError,
    InsufficientCapitalError,
    PositionError,
    RiskError,
    RiskLimitExceededError,
    ExecutionError,
    OrderError,
    ConfigurationError,
    ConnectionError,
    TimeoutError,
    handle_api_error,
    validate_dataframe,
    validate_positive_value,
    validate_range_value
)

from .logging_config import (
    setup_logging,
    get_logger,
    get_trade_logger,
    LoggerMixin,
    log_function_call,
    log_execution_time
)

from .monitoring import (
    PerformanceMetrics,
    AlertRule,
    Alert,
    PerformanceTracker,
    AlertManager,
    ReportGenerator,
    MonitoringSystem,
    get_monitoring_system,
    set_monitoring_system
)

__all__ = [
    'QuantTradingError',
    'DataError',
    'APIError',
    'DataNotFoundError',
    'DataValidationError',
    'StrategyError',
    'StrategyParameterError',
    'SignalGenerationError',
    'BacktestError',
    'InsufficientCapitalError',
    'PositionError',
    'RiskError',
    'RiskLimitExceededError',
    'ExecutionError',
    'OrderError',
    'ConfigurationError',
    'ConnectionError',
    'TimeoutError',
    'handle_api_error',
    'validate_dataframe',
    'validate_positive_value',
    'validate_range_value',
    'setup_logging',
    'get_logger',
    'get_trade_logger',
    'LoggerMixin',
    'log_function_call',
    'log_execution_time',
    'PerformanceMetrics',
    'AlertRule',
    'Alert',
    'PerformanceTracker',
    'AlertManager',
    'ReportGenerator',
    'MonitoringSystem',
    'get_monitoring_system',
    'set_monitoring_system',
]
