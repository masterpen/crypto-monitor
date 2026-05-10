# 量化交易系统

基于 Python 的模块化量化交易系统，支持加密货币自动交易与策略研究。

## 项目结构

```
quant-trading-system/
├── src/                        # 核心代码
│   ├── data/                   # 数据模块（加载、存储、缓存）
│   ├── factors/                # 因子模块（基类、内置因子、评估器）
│   ├── strategies/             # 策略模块（15+策略、优化器、比较器）
│   ├── backtest/               # 回测模块（引擎、向量化引擎）
│   ├── risk/                   # 风控模块（仓位管理、动态风控）
│   ├── engine/                 # 执行模块（订单管理、执行算法）
│   ├── monitoring/             # 监控模块（性能跟踪、报告生成）
│   ├── config.py               # 默认配置
│   ├── exceptions.py           # 自定义异常
│   ├── logging_config.py       # 日志配置
│   └── market_state.py         # 市场状态识别
│
├── scripts/                    # 入口脚本
│   ├── run_backtest.py         # 运行回测
│   ├── run_trading.py          # 运行交易
│   ├── run_dashboard.py        # 运行 Dashboard
│   └── batch_backtest.py       # 批量回测
│
├── tests/                      # 测试代码
├── config/                     # 配置文件
├── data/                       # 数据目录
│   └── real_cache/             # 真实数据缓存
│
├── Dockerfile                  # Docker 部署
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt

# 回测
python scripts/run_backtest.py --symbol BTCUSDT --strategy TrendStrategy --days 365

# 交易
python scripts/run_trading.py --symbol BTCUSDT --strategy TrendStrategy

# Dashboard
streamlit run scripts/run_dashboard.py
```

## 策略列表

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| TrendStrategy | 双均线交叉 | 趋势行情 |
| MACDStrategy | MACD指标 | 中期趋势 |
| RSIStrategy | RSI超买超卖 | 反转行情 |
| MomentumStrategy | 动量策略 | 动量行情 |
| MeanReversionStrategy | 均值回归 | 震荡行情 |
| KDJStrategy | KDJ指标 | 反转行情 |
| VolumeSurgeStrategy | 放量突破 | 突破行情 |
| CryptoTrendStrategy | 加密货币优化 | 趋势行情 |
| FactorStrategy | 因子组合 | 多因子选股 |

## 因子框架

```python
from src.factors import FactorRegistry, FactorEvaluator

# 查看所有因子
print(FactorRegistry.list_factors())

# 评估因子
ev = FactorEvaluator(data, factor_name='atr_ratio')
ic, ir = ev.evaluate(forward_period=24)
```

## 策略研究

```python
from src.strategies import FactorStrategy
from src.backtest import BacktestEngine

strategy = FactorStrategy(entry_threshold=0.5, exit_threshold=0.2)
engine = BacktestEngine(initial_capital=10000)
result = engine.run_backtest(data, strategy.generate_signal)
```

## 注意事项

⚠️ **风险提示**: 本系统仅供研究使用，实盘交易存在风险。