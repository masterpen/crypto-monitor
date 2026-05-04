"""
量化交易系统自定义异常模块
提供统一的异常处理和错误分类
"""


class QuantTradingError(Exception):
    """量化交易系统基础异常类"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class DataError(QuantTradingError):
    """数据相关异常"""
    
    def __init__(self, message: str, symbol: str = None, error_code: str = "DATA_ERROR", **kwargs):
        self.symbol = symbol
        details = {"symbol": symbol} if symbol else {}
        details.update(kwargs)
        super().__init__(message, error_code, details)


class APIError(DataError):
    """API请求异常"""
    
    def __init__(self, message: str, status_code: int = None, response_data: dict = None, **kwargs):
        self.status_code = status_code
        self.response_data = response_data or {}
        details = {
            "status_code": status_code,
            "response_data": response_data
        }
        super().__init__(message, error_code="API_ERROR", **details, **kwargs)


class DataNotFoundError(DataError):
    """数据未找到异常"""
    
    def __init__(self, message: str, symbol: str = None, timeframe: str = None, **kwargs):
        self.timeframe = timeframe
        details = {"timeframe": timeframe} if timeframe else {}
        super().__init__(message, symbol=symbol, error_code="DATA_NOT_FOUND", **details, **kwargs)


class DataValidationError(DataError):
    """数据验证异常"""
    
    def __init__(self, message: str, validation_errors: list = None, **kwargs):
        self.validation_errors = validation_errors or []
        details = {"validation_errors": self.validation_errors}
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", **details, **kwargs)


class StrategyError(QuantTradingError):
    """策略相关异常"""
    
    def __init__(self, message: str, strategy_name: str = None, error_code: str = "STRATEGY_ERROR", **kwargs):
        self.strategy_name = strategy_name
        details = {"strategy_name": strategy_name} if strategy_name else {}
        details.update(kwargs)
        super().__init__(message, error_code, details)


class StrategyParameterError(StrategyError):
    """策略参数异常"""
    
    def __init__(self, message: str, parameter_name: str = None, parameter_value=None, **kwargs):
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        details = {
            "parameter_name": parameter_name,
            "parameter_value": parameter_value
        }
        super().__init__(message, error_code="STRATEGY_PARAMETER_ERROR", **details, **kwargs)


class SignalGenerationError(StrategyError):
    """信号生成异常"""
    
    def __init__(self, message: str, index: int = None, **kwargs):
        self.index = index
        details = {"index": index} if index is not None else {}
        super().__init__(message, error_code="SIGNAL_GENERATION_ERROR", **details, **kwargs)


class BacktestError(QuantTradingError):
    """回测相关异常"""
    
    def __init__(self, message: str, error_code: str = "BACKTEST_ERROR", **kwargs):
        super().__init__(message, error_code, kwargs)


class InsufficientCapitalError(BacktestError):
    """资金不足异常"""
    
    def __init__(self, message: str, required: float = None, available: float = None, **kwargs):
        self.required = required
        self.available = available
        details = {
            "required": required,
            "available": available
        }
        super().__init__(message, error_code="INSUFFICIENT_CAPITAL", **details, **kwargs)


class PositionError(BacktestError):
    """持仓异常"""
    
    def __init__(self, message: str, position_side: str = None, **kwargs):
        self.position_side = position_side
        details = {"position_side": position_side} if position_side else {}
        super().__init__(message, error_code="POSITION_ERROR", **details, **kwargs)


class RiskError(QuantTradingError):
    """风控相关异常"""
    
    def __init__(self, message: str, risk_type: str = None, error_code: str = "RISK_ERROR", **kwargs):
        self.risk_type = risk_type
        details = {"risk_type": risk_type} if risk_type else {}
        details.update(kwargs)
        super().__init__(message, error_code, details)


class RiskLimitExceededError(RiskError):
    """风控限制超出异常"""
    
    def __init__(self, message: str, limit_type: str = None, current_value: float = None, 
                 limit_value: float = None, **kwargs):
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value
        details = {
            "limit_type": limit_type,
            "current_value": current_value,
            "limit_value": limit_value
        }
        super().__init__(message, risk_type=limit_type, error_code="RISK_LIMIT_EXCEEDED", **details, **kwargs)


class ExecutionError(QuantTradingError):
    """执行相关异常"""
    
    def __init__(self, message: str, order_id: str = None, error_code: str = "EXECUTION_ERROR", **kwargs):
        self.order_id = order_id
        details = {"order_id": order_id} if order_id else {}
        details.update(kwargs)
        super().__init__(message, error_code, details)


class OrderError(ExecutionError):
    """订单异常"""
    
    def __init__(self, message: str, order_type: str = None, symbol: str = None, **kwargs):
        self.order_type = order_type
        self.symbol = symbol
        details = {
            "order_type": order_type,
            "symbol": symbol
        }
        super().__init__(message, error_code="ORDER_ERROR", **details, **kwargs)


class ConfigurationError(QuantTradingError):
    """配置相关异常"""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        self.config_key = config_key
        details = {"config_key": config_key} if config_key else {}
        super().__init__(message, error_code="CONFIG_ERROR", details=details)


class ConnectionError(QuantTradingError):
    """连接异常"""
    
    def __init__(self, message: str, host: str = None, port: int = None, **kwargs):
        self.host = host
        self.port = port
        details = {"host": host, "port": port}
        super().__init__(message, error_code="CONNECTION_ERROR", details=details)


class TimeoutError(QuantTradingError):
    """超时异常"""
    
    def __init__(self, message: str, timeout_seconds: float = None, operation: str = None, **kwargs):
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        details = {
            "timeout_seconds": timeout_seconds,
            "operation": operation
        }
        super().__init__(message, error_code="TIMEOUT_ERROR", details=details)


# 异常处理工具函数
def handle_api_error(response, endpoint: str = None) -> None:
    """处理API响应错误"""
    if response.status_code != 200:
        try:
            error_data = response.json()
        except:
            error_data = {"raw_response": response.text}
        
        raise APIError(
            message=f"API请求失败: {endpoint}",
            status_code=response.status_code,
            response_data=error_data
        )


def validate_dataframe(df, required_columns: list, name: str = "DataFrame") -> None:
    """验证DataFrame是否包含必需的列"""
    if df is None or df.empty:
        raise DataNotFoundError(f"{name}为空")
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise DataValidationError(
            f"{name}缺少必需的列: {missing_columns}",
            validation_errors=[f"缺少列: {col}" for col in missing_columns]
        )


def validate_positive_value(value: float, name: str) -> None:
    """验证数值是否为正数"""
    if value <= 0:
        raise ValueError(f"{name}必须为正数，当前值: {value}")


def validate_range_value(value: float, min_val: float, max_val: float, name: str) -> None:
    """验证数值是否在指定范围内"""
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name}必须在{min_val}和{max_val}之间，当前值: {value}")