-- 量化交易系统数据库初始化脚本

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 创建交易记录表
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    commission DECIMAL(20, 8) DEFAULT 0,
    pnl DECIMAL(20, 8) DEFAULT 0,
    strategy_name VARCHAR(50),
    signal_reason TEXT,
    exchange_order_id VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建持仓表
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    strategy_name VARCHAR(50),
    status VARCHAR(20) DEFAULT 'open',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建权益曲线表
CREATE TABLE IF NOT EXISTS equity_curve (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    equity DECIMAL(20, 8) NOT NULL,
    peak_equity DECIMAL(20, 8) NOT NULL,
    drawdown DECIMAL(20, 8) DEFAULT 0,
    drawdown_pct DECIMAL(10, 4) DEFAULT 0,
    daily_pnl DECIMAL(20, 8) DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建策略性能表
CREATE TABLE IF NOT EXISTS strategy_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_name VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(10, 4) DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    avg_trade_pnl DECIMAL(20, 8) DEFAULT 0,
    profit_factor DECIMAL(10, 4) DEFAULT 0,
    sharpe_ratio DECIMAL(10, 4) DEFAULT 0,
    max_drawdown DECIMAL(20, 8) DEFAULT 0,
    max_drawdown_pct DECIMAL(10, 4) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_name, date)
);

-- 创建告警记录表
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id VARCHAR(50) UNIQUE NOT NULL,
    rule_id VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT,
    metric_value DECIMAL(20, 8),
    threshold DECIMAL(20, 8),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建配置表
CREATE TABLE IF NOT EXISTS configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_name);

CREATE INDEX IF NOT EXISTS idx_equity_curve_timestamp ON equity_curve(timestamp);

CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy ON strategy_performance(strategy_name);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_date ON strategy_performance(date);

CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);

-- 创建更新时间触发器
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_trades_updated_at BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_configurations_updated_at BEFORE UPDATE ON configurations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 插入默认配置
INSERT INTO configurations (config_key, config_value, description) VALUES
('risk.max_position_pct', '0.2', '单品种最大仓位比例'),
('risk.max_total_position_pct', '0.6', '总仓位上限'),
('risk.max_daily_loss_pct', '0.05', '日度最大亏损比例'),
('risk.max_drawdown_pct', '0.15', '最大回撤限制'),
('risk.stop_loss_pct', '0.02', '止损比例'),
('risk.take_profit_pct', '0.05', '止盈比例'),
('risk.max_trades_per_day', '10', '日度最大交易次数'),
('risk.min_trade_interval_sec', '60', '最小交易间隔（秒）'),
('trading.initial_capital', '10000', '初始资金'),
('trading.commission', '0.001', '手续费率'),
('trading.slippage', '0.0005', '滑点率')
ON CONFLICT (config_key) DO NOTHING;

-- 创建视图：每日交易统计
CREATE OR REPLACE VIEW daily_trade_stats AS
SELECT
    DATE(created_at) as date,
    COUNT(*) as total_trades,
    COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
    COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl,
    MAX(pnl) as max_pnl,
    MIN(pnl) as min_pnl,
    SUM(commission) as total_commission
FROM trades
WHERE status = 'filled'
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- 创建视图：策略性能汇总
CREATE OR REPLACE VIEW strategy_summary AS
SELECT
    strategy_name,
    COUNT(*) as total_trades,
    COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
    COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
    ROUND(COUNT(CASE WHEN pnl > 0 THEN 1 END)::DECIMAL / COUNT(*) * 100, 2) as win_rate,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_trade_pnl,
    ROUND(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) / ABS(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END)), 2) as profit_factor
FROM trades
WHERE status = 'filled'
GROUP BY strategy_name
ORDER BY total_pnl DESC;

-- 创建视图：当前持仓
CREATE OR REPLACE VIEW current_positions AS
SELECT
    p.symbol,
    p.side,
    p.quantity,
    p.entry_price,
    p.current_price,
    p.unrealized_pnl,
    p.strategy_name,
    p.created_at as entry_time,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - p.created_at)) / 3600 as holding_hours
FROM positions p
WHERE p.status = 'open'
ORDER BY p.created_at DESC;

COMMENT ON TABLE trades IS '交易记录表';
COMMENT ON TABLE positions IS '持仓表';
COMMENT ON TABLE equity_curve IS '权益曲线表';
COMMENT ON TABLE strategy_performance IS '策略性能表';
COMMENT ON TABLE alerts IS '告警记录表';
COMMENT ON TABLE configurations IS '配置表';