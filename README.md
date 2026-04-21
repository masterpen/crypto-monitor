# 量化交易系统

基于 Python 的模块化量化交易系统，支持数字货币（BTC/ETH）自动交易。

## 功能特性

- 📊 **数据采集**: 支持 Binance 实时行情和历史数据获取
- 📈 **策略回测**: 完整的回测框架，支持多策略验证
- ⚡ **交易执行**: 异步订单执行，实时持仓管理
- 🛡️ **风险控制**: 多维度风控规则，止损止盈
- 📉 **可视化**: Streamlit Dashboard，实时监控

## 策略列表

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| 双均线策略 | 快速/慢速均线交叉 | 趋势行情 |
| MACD策略 | MACD指标信号 | 中期趋势 |
| 布林带策略 | 价格触及布林带上下轨 | 震荡行情 |
| RSI策略 | RSI超买超卖信号 | 反转行情 |
| 动量策略 | 价格动量加速/减弱 | 动量行情 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config/config.yaml`：

```yaml
api:
  binance:
    api_key: "your_api_key"
    api_secret: "your_api_secret"
    testnet: true  # 使用测试网

trading:
  symbols: ["BTCUSDT", "ETHUSDT"]
  timeframes: ["1h", "4h"]

risk:
  max_position_pct: 0.2
  max_daily_loss_pct: 0.05
  stop_loss_pct: 0.02
```

### 3. 运行回测

```bash
python run_backtest.py --symbol BTCUSDT --strategy TrendStrategy --days 365
```

### 4. 启动实盘交易

```bash
python run_trading.py --symbol BTCUSDT --strategy TrendStrategy
```

### 5. 启动 Dashboard

```bash
streamlit run dashboard.py
```

## 项目结构

```
quant-trading-system/
├── config/
│   └── config.yaml          # 配置文件
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── binance_client.py  # Binance API 客户端
│   │   └── data_loader.py     # 数据加载器
│   ├── strategies/
│   │   └── __init__.py        # 交易策略
│   ├── backtest/
│   │   └── engine.py          # 回测引擎
│   ├── engine/
│   │   └── __init__.py        # 交易执行引擎
│   └── risk/
│       └── __init__.py        # 风控模块
├── run_backtest.py           # 回测入口
├── run_trading.py            # 实盘交易入口
├── dashboard.py              # 可视化 Dashboard
├── requirements.txt
└── README.md
```

## 回测指标

| 指标 | 说明 |
|------|------|
| 总收益率 | 账户总收益百分比 |
| 夏普比率 | 风险调整后收益 |
| 最大回撤 | 历史最大亏损 |
| 胜率 | 盈利交易占比 |
| 盈亏比 | 平均盈利/平均亏损 |

## 风控规则

- 单品种最大仓位: 20%
- 总仓位上限: 60%
- 日度最大亏损: 5%
- 最大回撤限制: 15%
- 止损比例: 2%
- 止盈比例: 5%
- 最小交易间隔: 60秒

## 注意事项

⚠️ **风险提示**: 本系统仅供学习和研究使用，实盘交易存在风险，请谨慎操作。

1. 建议先在测试网验证策略
2. 不要投入超过承受能力的资金
3. 定期检查策略表现和风控状态
4. 保持对市场行情的关注

## 许可证

MIT License