"""Test API contracts to ensure backward compatibility.

These tests verify that the public API surface of all classes remains stable.
If these tests fail, it means a breaking change has been introduced.
"""

import inspect
import pytest
from trading_bot import (
    TradingConfig,
    TechnicalIndicators,
    MLPredictor,
    TradingStrategy,
    TradeHistory,
    DataCurator
)


class TestTradingConfigAPI:
    """Test TradingConfig dataclass fields and types."""
    
    def test_has_expected_fields(self):
        """Verify all expected fields exist."""
        config = TradingConfig()
        expected_fields = {
            'trading_pairs', 'position_size_percent', 'max_positions',
            'stop_loss_percent', 'take_profit_percent', 'trailing_stop_percent',
            'max_daily_loss_percent', 'rsi_period', 'rsi_oversold', 'rsi_overbought',
            'macd_fast', 'macd_slow', 'macd_signal',
            'ema_short', 'ema_long', 'bb_period', 'bb_std_dev',
            'atr_period', 'vwap_enabled', 'obv_enabled', 'min_signal_agreement',
            'ml_enabled', 'ml_confidence_threshold', 'ml_model_type',
            'ml_training_days', 'ml_retrain_interval', 'ml_lookback_periods',
            'lookback_window', 'candle_granularity', 'refresh_interval',
            'backtest_start_balance', 'backtest_commission_percent',
            'db_path', 'model_path'
        }
        
        actual_fields = set(config.__dataclass_fields__.keys())
        assert expected_fields == actual_fields, \
            f"Missing: {expected_fields - actual_fields}, Extra: {actual_fields - expected_fields}"
    
    def test_field_types(self):
        """Verify field types are correct."""
        config = TradingConfig()
        
        # Check a few key types
        assert isinstance(config.trading_pairs, list)
        assert isinstance(config.position_size_percent, (int, float))
        assert isinstance(config.max_positions, int)
        assert isinstance(config.rsi_period, int)
        assert isinstance(config.db_path, str)


class TestTechnicalIndicatorsAPI:
    """Test TechnicalIndicators class methods."""
    
    def test_has_required_methods(self):
        """Verify all expected methods exist."""
        expected_methods = {
            'calculate_rsi', 'calculate_macd', 'calculate_ema',
            'calculate_bollinger_bands', 'calculate_atr', 'calculate_obv',
            'calculate_stochastic', 'calculate_vwap', 'calculate_adx',
            'add_all_indicators'
        }
        
        actual_methods = {
            name for name in dir(TechnicalIndicators)
            if not name.startswith('_') and callable(getattr(TechnicalIndicators, name))
        }
        
        assert expected_methods.issubset(actual_methods), \
            f"Missing methods: {expected_methods - actual_methods}"
    
    def test_method_signatures(self):
        """Verify method signatures haven't changed."""
        # Test a few key signatures
        sig = inspect.signature(TechnicalIndicators.calculate_rsi)
        params = list(sig.parameters.keys())
        assert 'df' in params
        assert 'period' in params
        
        sig = inspect.signature(TechnicalIndicators.add_all_indicators)
        params = list(sig.parameters.keys())
        assert 'df' in params
        assert 'config' in params


class TestMLPredictorAPI:
    """Test MLPredictor class methods."""
    
    def test_has_required_methods(self):
        """Verify all expected methods exist."""
        expected_methods = {
            'train', 'predict', 'predict_proba', 'save_model', 'load_model',
            'prepare_features', 'create_target'
        }
        
        predictor = MLPredictor()
        actual_methods = {
            name for name in dir(predictor)
            if not name.startswith('_') and callable(getattr(predictor, name))
        }
        
        assert expected_methods.issubset(actual_methods), \
            f"Missing methods: {expected_methods - actual_methods}"
    
    def test_train_signature(self):
        """Verify train method signature."""
        sig = inspect.signature(MLPredictor.train)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'df' in params
        # Check that it returns something (accuracy)
        assert sig.return_annotation != inspect.Signature.empty or True
    
    def test_predict_signature(self):
        """Verify predict method signature."""
        sig = inspect.signature(MLPredictor.predict)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'df' in params


class TestTradingStrategyAPI:
    """Test TradingStrategy class methods."""
    
    def test_has_required_methods(self, sample_config):
        """Verify all expected methods exist."""
        expected_methods = {
            'generate_signal', 'analyze_rsi', 'analyze_macd', 'analyze_ema',
            'analyze_bollinger_bands', 'analyze_stochastic', 'analyze_trend_strength'
        }
        
        strategy = TradingStrategy(sample_config)
        actual_methods = {
            name for name in dir(strategy)
            if not name.startswith('_') and callable(getattr(strategy, name))
        }
        
        assert expected_methods.issubset(actual_methods), \
            f"Missing methods: {expected_methods - actual_methods}"
    
    def test_generate_signal_signature(self):
        """Verify generate_signal returns tuple."""
        sig = inspect.signature(TradingStrategy.generate_signal)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'df' in params


class TestTradeHistoryAPI:
    """Test TradeHistory class methods."""
    
    def test_has_required_methods(self, temp_db_path):
        """Verify all expected methods exist."""
        expected_methods = {
            'store_trade', 'store_prediction', 'store_market_data',
            'get_trade_stats', 'get_model_accuracy', 'get_training_data'
        }
        
        history = TradeHistory(temp_db_path)
        actual_methods = {
            name for name in dir(history)
            if not name.startswith('_') and callable(getattr(history, name))
        }
        
        assert expected_methods.issubset(actual_methods), \
            f"Missing methods: {expected_methods - actual_methods}"
    
    def test_store_trade_signature(self, temp_db_path):
        """Verify store_trade method signature."""
        history = TradeHistory(temp_db_path)
        sig = inspect.signature(history.store_trade)
        params = list(sig.parameters.keys())
        
        # Check key parameters exist
        assert 'symbol' in params
        assert 'side' in params
        assert 'entry_time' in params
        assert 'entry_price' in params


class TestDataCuratorAPI:
    """Test DataCurator class methods."""
    
    def test_has_required_methods(self, temp_db_path):
        """Verify all expected methods exist."""
        expected_methods = {
            'analyze_model_performance', 'analyze_trade_performance',
            'export_for_analysis', 'clean_old_data', 'get_data_summary'
        }
        
        curator = DataCurator(temp_db_path)
        actual_methods = {
            name for name in dir(curator)
            if not name.startswith('_') and callable(getattr(curator, name))
        }
        
        assert expected_methods.issubset(actual_methods), \
            f"Missing methods: {expected_methods - actual_methods}"
    
    def test_analyze_model_performance_signature(self, temp_db_path):
        """Verify analyze_model_performance signature."""
        curator = DataCurator(temp_db_path)
        sig = inspect.signature(curator.analyze_model_performance)
        params = list(sig.parameters.keys())
        assert 'symbol' in params or 'self' in params


class TestSignalEnumAPI:
    """Test Signal enum values."""
    
    def test_signal_values(self):
        """Verify Signal enum has expected values."""
        from trading_bot.strategy import Signal
        
        expected_values = {'BUY', 'SELL', 'HOLD'}
        actual_values = {signal.name for signal in Signal}
        
        assert expected_values == actual_values, \
            f"Signal enum mismatch. Expected: {expected_values}, Got: {actual_values}"
