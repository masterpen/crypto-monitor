from src.risk import (
    RiskConfig, RiskMetrics, RiskManager,
    PositionSizer, AlertManager
)


class TestRiskConfig:
    def test_defaults(self):
        c = RiskConfig()
        assert c.max_position_pct == 0.2
        assert c.stop_loss_pct == 0.02

    def test_custom(self):
        c = RiskConfig(max_position_pct=0.3, stop_loss_pct=0.05)
        assert c.max_position_pct == 0.3
        assert c.stop_loss_pct == 0.05


class TestRiskMetrics:
    def test_defaults(self):
        m = RiskMetrics()
        assert m.daily_pnl == 0.0
        assert m.current_drawdown == 0.0


class TestRiskManager:
    def test_initial_state(self):
        config = RiskConfig()
        rm = RiskManager(config=config, initial_capital=10000)
        report = rm.get_risk_report()
        assert report['metrics']['current_equity'] == 10000
        assert report['metrics']['peak_equity'] == 10000

    def test_update_equity(self):
        rm = RiskManager(initial_capital=10000)
        rm.update_equity(11000)
        assert rm.metrics.current_equity == 11000
        assert rm.metrics.peak_equity == 11000
        assert rm.metrics.current_drawdown == 0

    def test_update_equity_drawdown(self):
        rm = RiskManager(initial_capital=10000)
        rm.update_equity(11000)
        rm.update_equity(10500)
        assert rm.metrics.current_drawdown > 0

    def test_check_order_allowed(self):
        rm = RiskManager(initial_capital=10000)
        result = rm.check_order('BTCUSDT', 'buy', 0.001, price=50000)
        assert result['allowed'] is True

    def test_check_order_position_limit(self):
        config = RiskConfig(max_position_pct=0.01)
        rm = RiskManager(config=config, initial_capital=10000)
        result = rm.check_order('BTCUSDT', 'buy', 10, price=50000)
        assert result['allowed'] is False

    def test_update_position(self):
        rm = RiskManager(initial_capital=10000)
        rm.update_position('BTCUSDT', 'long', 0.1, 50000)
        assert 'BTCUSDT' in rm.positions

    def test_update_position_zero_removes(self):
        rm = RiskManager(initial_capital=10000)
        rm.update_position('BTCUSDT', 'long', 0.1, 50000)
        rm.update_position('BTCUSDT', 'flat', 0, 0)
        assert 'BTCUSDT' not in rm.positions

    def test_check_stop_loss_no_position(self):
        rm = RiskManager(initial_capital=10000)
        triggered, reason = rm.check_stop_loss('BTCUSDT', 50000)
        assert not triggered

    def test_check_take_profit_no_position(self):
        rm = RiskManager(initial_capital=10000)
        triggered, reason = rm.check_take_profit('BTCUSDT', 50000)
        assert not triggered

    def test_record_trade(self):
        rm = RiskManager(initial_capital=10000)
        rm.record_trade({'symbol': 'BTCUSDT', 'pnl': 100})
        assert len(rm.trade_log) == 1

    def test_disable_rule(self):
        rm = RiskManager(initial_capital=10000)
        rm.risk_rules_enabled['position_limit'] = False
        config = RiskConfig(max_position_pct=0.0001)
        rm.config = config
        result = rm.check_order('BTCUSDT', 'buy', 10, price=50000)
        assert result['allowed'] is True

    def test_get_risk_report_has_status(self):
        rm = RiskManager(initial_capital=10000)
        report = rm.get_risk_report()
        assert 'status' in report
        assert 'metrics' in report
        assert 'limits' in report

    def test_check_stop_loss_triggers(self):
        rm = RiskManager(initial_capital=10000)
        rm.update_position('BTCUSDT', 'long', 0.1, 50000)
        triggered, reason = rm.check_stop_loss('BTCUSDT', 48000)
        assert triggered

    def test_check_stop_loss_not_triggered(self):
        rm = RiskManager(initial_capital=10000)
        rm.update_position('BTCUSDT', 'long', 0.1, 50000)
        triggered, reason = rm.check_stop_loss('BTCUSDT', 50000)
        assert not triggered

    def test_check_take_profit_triggers(self):
        rm = RiskManager(initial_capital=10000)
        rm.update_position('BTCUSDT', 'long', 0.1, 50000)
        triggered, reason = rm.check_take_profit('BTCUSDT', 53000)
        assert triggered

    def test_check_stop_loss_short_triggers(self):
        rm = RiskManager(initial_capital=10000)
        rm.update_position('BTCUSDT', 'short', 0.1, 50000)
        triggered, reason = rm.check_stop_loss('BTCUSDT', 52000)
        assert triggered


class TestPositionSizer:
    def test_fixed_fraction(self):
        ps = PositionSizer(initial_capital=10000)
        assert ps.fixed_fraction(0.1) == 1000

    def test_kelly_div_by_zero(self):
        ps = PositionSizer()
        k = ps.calculate_kelly_fraction(0.5, 1, 0)
        assert k == 0

    def test_kelly_zero_with_no_win(self):
        ps = PositionSizer()
        k = ps.calculate_kelly_fraction(0.5, 0, 0.1)
        assert k == 0.0

    def test_kelly_returns_positive(self):
        ps = PositionSizer()
        k = ps.calculate_kelly_fraction(0.6, 200, 100)
        assert k >= 0

    def test_calculate_position_size(self):
        ps = PositionSizer(initial_capital=10000)
        qty = ps.calculate_position_size(100, 98)
        assert qty > 0

    def test_position_size_zero_risk(self):
        ps = PositionSizer(initial_capital=10000)
        qty = ps.calculate_position_size(100, 100)
        assert qty == 0

    def test_position_size_with_risk_amount(self):
        ps = PositionSizer()
        qty = ps.calculate_position_size(100, 95, risk_amount=100)
        assert qty == 20


class TestAlertManager:
    def test_send_alert(self):
        am = AlertManager()
        am.send_alert('warning', 'Test', 'Message')
        assert len(am.alerts) == 1
        assert am.alerts[0]['level'] == 'warning'

    def test_recent_alerts(self):
        am = AlertManager()
        am.send_alert('info', 'Test', 'Recent')
        recent = am.get_recent_alerts(24)
        assert len(recent) == 1

    def test_clear_alerts(self):
        am = AlertManager()
        am.send_alert('error', 'Test', 'Clear me')
        am.clear_alerts()
        assert len(am.alerts) == 0

    def test_callback_invoked(self):
        calls = []

        def cb(alert):
            calls.append(alert)

        am = AlertManager()
        am.add_callback(cb)
        am.send_alert('info', 'Test', 'With callback')
        assert len(calls) == 1
        assert calls[0]['level'] == 'info'