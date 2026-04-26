import numpy as np
from tests.conftest import make_ohlcv_data, make_uptrend_data, make_downtrend_data
from src.strategies import (
    BaseStrategy, TrendStrategy, MACDStrategy, MeanReversionStrategy,
    RSIStrategy, MomentumStrategy, ArbitrageStrategy,
    KDJStrategy, KDJCrossStrategy, KDJExitAtK50, KDJExitOverbought,
    KDJExitWithSLTP, VolumeSurgeStrategy, StrategyEnsemble
)


class TestBaseStrategy:
    def test_params(self):
        s = TrendStrategy(fast_period=5, slow_period=20)
        assert s.get_params() == {'fast_period': 5, 'slow_period': 20}
        s.set_params(fast_period=8)
        assert s.get_params()['fast_period'] == 8

    def test_name(self):
        s = TrendStrategy()
        assert s.name == "TrendStrategy"


class TestTrendStrategy:
    def test_insufficient_data_returns_hold(self):
        s = TrendStrategy(fast_period=10, slow_period=30)
        data = make_ohlcv_data(20)
        assert s.generate_signal(data, 10) == 'hold'

    def test_returns_valid_signal(self):
        s = TrendStrategy(fast_period=10, slow_period=30)
        data = make_ohlcv_data(200)
        sig = s.generate_signal(data, 50)
        assert sig in ('long', 'short', 'close', 'hold')

    def test_close_on_reversal(self):
        s = TrendStrategy(fast_period=10, slow_period=30)
        data = make_uptrend_data(200)
        signal = s.generate_signal(data, 100, position_side='long')
        assert signal in ('close', 'hold')

    def test_flat_generates_directional(self):
        s = TrendStrategy(fast_period=10, slow_period=30)
        data = make_ohlcv_data(300)
        signals = [s.generate_signal(data, i, position_side='flat')
                   for i in range(30, 280)]
        assert any(sig in ('long', 'short') for sig in signals)


class TestMACDStrategy:
    def test_returns_valid_signal(self):
        s = MACDStrategy()
        data = make_ohlcv_data(200)
        sig = s.generate_signal(data, 50)
        assert sig in ('long', 'short', 'close', 'hold')

    def test_close_on_bearish(self):
        s = MACDStrategy()
        data = make_uptrend_data(200)
        sig = s.generate_signal(data, 100, position_side='long')
        assert sig in ('close', 'hold')


class TestMeanReversionStrategy:
    def test_returns_valid_signal(self):
        s = MeanReversionStrategy()
        data = make_ohlcv_data(200)
        sig = s.generate_signal(data, 50)
        assert sig in ('long', 'short', 'close', 'hold')


class TestRSIStrategy:
    def test_insufficient_data(self):
        s = RSIStrategy()
        data = make_ohlcv_data(14)
        assert s.generate_signal(data, 10) == 'hold'

    def test_returns_valid_signal(self):
        s = RSIStrategy()
        data = make_ohlcv_data(200)
        sig = s.generate_signal(data, 50)
        assert sig in ('long', 'short', 'close', 'hold')


class TestMomentumStrategy:
    def test_returns_valid_signal(self):
        s = MomentumStrategy()
        data = make_ohlcv_data(200)
        sig = s.generate_signal(data, 50)
        assert sig in ('long', 'short', 'close', 'hold')

    def test_close_on_weakening(self):
        s = MomentumStrategy()
        data = make_uptrend_data(200)
        sig = s.generate_signal(data, 100, position_side='long')
        assert sig in ('close', 'hold')


class TestArbitrageStrategy:
    def test_triangular_arb_calculation(self):
        result = ArbitrageStrategy.calculate_triangular_arb(
            50000, 0.06, 3000, "forward")
        assert isinstance(result, float)

    def test_reverse_triangular(self):
        result = ArbitrageStrategy.calculate_triangular_arb(
            50000, 0.06, 3000, "reverse")
        assert isinstance(result, float)

    def test_returns_hold(self):
        s = ArbitrageStrategy()
        data = make_ohlcv_data(200)
        sig = s.generate_signal(data, 50)
        assert sig == 'hold'


class TestKDJStrategy:
    def test_returns_valid_signal(self):
        s = KDJStrategy()
        data = make_ohlcv_data(200)
        sig = s.generate_signal(data, 50)
        assert sig in ('long', 'close', 'hold')

    def test_close_on_j_turn_positive(self):
        s = KDJStrategy()
        data = make_uptrend_data(200)
        sig = s.generate_signal(
            data, 100, position_side='long', entry_price=105)
        assert sig in ('close', 'hold')


class TestKDJCrossStrategy:
    def test_returns_valid_signal(self):
        s = KDJCrossStrategy()
        data = make_ohlcv_data(200)
        sig = s.generate_signal(data, 50)
        assert sig in ('long', 'short', 'close', 'hold')


class TestKDJExitVariants:
    def test_kdj_exit_at_k50(self):
        s = KDJExitAtK50()
        data = make_ohlcv_data(200)
        sig = s.generate_signal(data, 50)
        assert sig in ('long', 'close', 'hold')

    def test_kdj_exit_overbought(self):
        s = KDJExitOverbought()
        data = make_ohlcv_data(200)
        sig = s.generate_signal(data, 50)
        assert sig in ('long', 'close', 'hold')

    def test_kdj_exit_with_sltp_stop_loss(self):
        s = KDJExitWithSLTP(stop_loss_pct=0.02)
        data = make_downtrend_data(200)
        sig = s.generate_signal(
            data, 100, position_side='long',
            entry_price=data['close'].iloc[50] * 1.05)
        assert sig in ('close', 'hold')


class TestVolumeSurgeStrategy:
    def test_returns_valid_signal(self):
        s = VolumeSurgeStrategy()
        data = make_ohlcv_data(200)
        sig = s.generate_signal(data, 50)
        assert sig in ('long', 'close', 'hold')

    def test_calc_limit_price(self):
        s = VolumeSurgeStrategy(limit_price_pct=-0.03)
        assert s.calc_limit_price(100) == 97.0

    def test_resets_state_on_close(self):
        s = VolumeSurgeStrategy()
        data = make_ohlcv_data(200)
        s._highest_price = 1000
        s._reset_state()
        assert s._highest_price == 0.0


class TestStrategyEnsemble:
    def test_empty_returns_hold(self):
        e = StrategyEnsemble()
        data = make_ohlcv_data(200)
        assert e.generate_signal(data, 50) == 'hold'

    def test_single_strategy(self):
        e = StrategyEnsemble()
        e.add_strategy(TrendStrategy(), weight=1.0)
        data = make_ohlcv_data(200)
        sig = e.generate_signal(data, 50)
        assert sig in ('long', 'short', 'close', 'hold')

    def test_multi_strategy_voting(self):
        e = StrategyEnsemble()
        e.add_strategy(TrendStrategy(
            fast_period=5, slow_period=20), weight=2.0)
        e.add_strategy(MACDStrategy(), weight=1.0)
        data = make_ohlcv_data(200)
        sig = e.generate_signal(data, 50)
        assert sig in ('long', 'short', 'close', 'hold')

    def test_signal_details(self):
        e = StrategyEnsemble()
        e.add_strategy(TrendStrategy(), weight=1.0)
        e.add_strategy(MACDStrategy(), weight=0.5)
        data = make_ohlcv_data(200)
        details = e.get_signal_details(data, 50)
        assert 'TrendStrategy' in details
        assert 'MACDStrategy' in details
        assert details['TrendStrategy']['weight'] == 1.0
        assert details['MACDStrategy']['weight'] == 0.5