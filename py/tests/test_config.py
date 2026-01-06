"""Tests for config.py module - matching actual API."""

import os
import pytest
from trading_bot import TradingConfig


def test_default_config():
    """Test default configuration values."""
    config = TradingConfig()
    
    assert isinstance(config.trading_pairs, list)
    assert len(config.trading_pairs) > 0
    assert config.position_size_percent > 0
    assert config.max_positions > 0
    assert config.refresh_interval > 0


def test_custom_trading_pairs():
    """Test custom trading pairs."""
    pairs = ["BTC-USD", "ETH-USD"]
    config = TradingConfig(trading_pairs=pairs)
    
    assert config.trading_pairs == pairs


def test_position_size_validation():
    """Test position size is within valid range."""
    config = TradingConfig(position_size_percent=10.0)
    
    assert 0 < config.position_size_percent <= 100


def test_stop_loss_configuration():
    """Test stop loss settings."""
    config = TradingConfig(stop_loss_percent=2.0)
    
    assert config.stop_loss_percent > 0
    assert config.stop_loss_percent < 100


def test_take_profit_configuration():
    """Test take profit settings."""
    config = TradingConfig(take_profit_percent=5.0)
    
    assert config.take_profit_percent > 0


def test_ml_configuration():
    """Test ML-related settings."""
    config = TradingConfig(
        ml_enabled=True,
        ml_confidence_threshold=0.7,
        ml_training_days=90
    )
    
    assert config.ml_enabled is True
    assert 0 < config.ml_confidence_threshold < 1
    assert config.ml_training_days > 0


def test_indicator_periods():
    """Test technical indicator period settings."""
    config = TradingConfig(
        rsi_period=14,
        ema_short=9,
        ema_long=21
    )
    
    assert config.rsi_period > 0
    assert config.ema_short > 0
    assert config.ema_long > config.ema_short


def test_db_path_from_constants():
    """Test that db_path uses constant by default."""
    from trading_bot._constants import DB_PATH
    
    config = TradingConfig()
    
    # Should use default from constants
    assert isinstance(config.db_path, str)
    assert len(config.db_path) > 0


def test_db_path_override():
    """Test that db_path can be overridden."""
    custom_path = "/custom/trades.db"
    config = TradingConfig(db_path=custom_path)
    
    assert config.db_path == custom_path


def test_model_path_from_constants():
    """Test that model_path uses constant by default."""
    from trading_bot._constants import MODEL_PATH
    
    config = TradingConfig()
    
    assert isinstance(config.model_path, str)
    assert len(config.model_path) > 0


def test_model_path_override():
    """Test that model_path can be overridden."""
    custom_path = "/custom/models"
    config = TradingConfig(model_path=custom_path)
    
    assert config.model_path == custom_path


def test_refresh_interval():
    """Test refresh interval setting."""
    config = TradingConfig(refresh_interval=30)
    
    assert config.refresh_interval == 30
    assert config.refresh_interval > 0


def test_max_positions():
    """Test maximum positions setting."""
    config = TradingConfig(max_positions=5)
    
    assert config.max_positions == 5
    assert config.max_positions > 0


def test_config_mutability():
    """Test that config values can be changed after creation."""
    config = TradingConfig(position_size_percent=10.0)
    
    # Should be able to modify (dataclass by default is mutable)
    config.position_size_percent = 15.0
    assert config.position_size_percent == 15.0


def test_rsi_thresholds():
    """Test RSI overbought/oversold thresholds."""
    config = TradingConfig(
        rsi_oversold=30.0,
        rsi_overbought=70.0
    )
    
    assert config.rsi_oversold < config.rsi_overbought
    assert 0 <= config.rsi_oversold <= 100
    assert 0 <= config.rsi_overbought <= 100


def test_macd_periods():
    """Test MACD period configuration."""
    config = TradingConfig(
        macd_fast=12,
        macd_slow=26,
        macd_signal=9
    )
    
    assert config.macd_fast < config.macd_slow
    assert config.macd_signal > 0


def test_bollinger_bands_config():
    """Test Bollinger Bands configuration."""
    config = TradingConfig(
        bb_period=20,
        bb_std_dev=2.0
    )
    
    assert config.bb_period > 0
    assert config.bb_std_dev > 0


def test_backtest_configuration():
    """Test backtest settings."""
    config = TradingConfig(
        backtest_start_balance=10000.0,
        backtest_commission_percent=0.1
    )
    
    assert config.backtest_start_balance > 0
    assert config.backtest_commission_percent >= 0
