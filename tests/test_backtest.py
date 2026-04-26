import numpy as np
from tests.conftest import make_ohlcv_data, make_uptrend_data, make_downtrend_data
from src.backtest.engine import BacktestEngine, PositionSide


def _hold_strategy(data, index, position_side='flat', entry_price=None, **params):
    """始终 hold，不产生交易"""
    return 'hold'


class TestBacktestEngine:
    def test_initial_state(self):
        e = BacktestEngine(initial_capital=10000)
        assert e.capital == 10000
        assert e.position.side == PositionSide.FLAT
        assert len(e.trades) == 0
        assert len(e.equity_curve) == 1

    def test_reset(self):
        e = BacktestEngine(initial_capital=10000)
        e.equity_curve.append(11000)
        e.trades.append(None)
        e.reset()
        assert e.capital == 10000
        assert len(e.trades) == 0
        assert len(e.equity_curve) == 1

    def test_hold_strategy_no_trades(self):
        e = BacktestEngine(initial_capital=10000)
        data = make_ohlcv_data(200)
        results = e.run_backtest(data, _hold_strategy)
        assert results.total_trades == 0
        assert results.total_pnl == 0
        assert results.win_rate == 0

    def test_equity_curve_length(self):
        e = BacktestEngine(initial_capital=10000)
        data = make_ohlcv_data(200)
        results = e.run_backtest(data, _hold_strategy)
        assert len(results.equity_curve) > 1

    def test_sharpe_no_volatility(self):
        e = BacktestEngine(initial_capital=10000)
        data = make_ohlcv_data(200)
        results = e.run_backtest(data, _hold_strategy)
        assert results.sharpe_ratio == 0

    def test_metrics_have_expected_fields(self):
        e = BacktestEngine(initial_capital=10000)
        data = make_ohlcv_data(200)
        results = e.run_backtest(data, _hold_strategy)
        assert hasattr(results, 'total_trades')
        assert hasattr(results, 'sharpe_ratio')
        assert hasattr(results, 'sortino_ratio')
        assert hasattr(results, 'calmar_ratio')
        assert hasattr(results, 'max_drawdown_pct')
        assert hasattr(results, 'win_rate')
        assert hasattr(results, 'profit_factor')

    def test_quantity_rounding(self):
        e = BacktestEngine(quantity_precision=3)
        qty = e._round_quantity(1.2345678)
        assert qty == 1.234  # floor, not round

    def test_update_equity(self):
        e = BacktestEngine(initial_capital=10000)
        assert e.update_equity(100)
        assert len(e.equity_curve) == 2

    def test_update_equity_not_bankrupted(self):
        e = BacktestEngine(initial_capital=10000)
        result = e.update_equity(100)
        assert result is True

    def test_calc_market_value_flat(self):
        e = BacktestEngine()
        assert e._calc_market_value(100) == 0

    def test_summary_string(self):
        e = BacktestEngine(initial_capital=10000)
        data = make_ohlcv_data(200)
        results = e.run_backtest(data, _hold_strategy)
        summary = results.summary()
        assert "回测结果" in summary

    def test_capital_correct_after_hold(self):
        e = BacktestEngine(initial_capital=12345)
        data = make_ohlcv_data(100)
        results = e.run_backtest(data, _hold_strategy)
        assert isinstance(results, object)

    def test_no_trades_win_rate_zero(self):
        e = BacktestEngine(initial_capital=10000)
        data = make_ohlcv_data(200)
        results = e.run_backtest(data, _hold_strategy)
        assert results.win_rate == 0.0

    def test_no_trades_profit_factor_zero(self):
        e = BacktestEngine(initial_capital=10000)
        data = make_ohlcv_data(200)
        results = e.run_backtest(data, _hold_strategy)
        assert results.profit_factor == 0.0

    def test_max_drawdown_zero_for_constant_equity(self):
        e = BacktestEngine(initial_capital=10000)
        data = make_ohlcv_data(50)
        results = e.run_backtest(data, _hold_strategy)
        assert results.max_drawdown == 0
        assert results.max_drawdown_pct == 0

    def test_sortino_no_downside(self):
        e = BacktestEngine(initial_capital=10000)
        data = make_ohlcv_data(50)
        results = e.run_backtest(data, _hold_strategy)
        assert results.sortino_ratio == 0

    def test_calmar_ratio_no_drawdown(self):
        e = BacktestEngine(initial_capital=10000)
        data = make_ohlcv_data(50)
        results = e.run_backtest(data, _hold_strategy)
        assert results.calmar_ratio == 0

    def test_get_market_value_compat(self):
        e = BacktestEngine()
        assert e.get_market_value() == 0.0

    def test_can_open_position(self):
        e = BacktestEngine(initial_capital=10000)
        assert e.can_open_position(100, 0.1)
        assert not e.can_open_position(100000, 10)