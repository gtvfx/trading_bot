"""Pytest configuration and shared fixtures."""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def temp_model_dir():
    """Create a temporary model directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    
    # Generate realistic price data
    base_price = 50000.00
    prices = [base_price]
    for _ in range(99):
        change = prices[-1] * (0.98 + 0.04 * np.random.random())
        prices.append(change)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': [p * (0.995 + 0.01 * np.random.random()) for p in prices],
        'volume': [1000 + 500 * np.random.random() for _ in prices]
    })
    
    return df


@pytest.fixture
def sample_config():
    """Create a sample trading configuration."""
    from trading_bot import TradingConfig
    
    config = TradingConfig(
        trading_pairs=["BTC-USD", "ETH-USD"],
        position_size_percent=10.0,
        max_positions=2,
        stop_loss_percent=2.0,
        take_profit_percent=5.0,
        ml_enabled=False,  # Disable ML for most tests
        refresh_interval=5
    )
    
    return config


@pytest.fixture
def mock_exchange_client():
    """Create a mock exchange client for testing."""
    from unittest.mock import MagicMock
    from trading_bot.exchange_client import ExchangeClient
    
    client = MagicMock(spec=ExchangeClient)
    
    # Mock methods
    client.get_account_balance.return_value = {'USD': 10000.0, 'BTC': 0.5}
    client.get_buying_power.return_value = 10000.0
    client.get_current_price.return_value = 50000.0
    client.place_market_order.return_value = {
        'id': 'order_123',
        'status': 'filled',
        'filled_size': 0.1
    }
    
    return client


@pytest.fixture
def sample_df_with_indicators(sample_ohlcv_data):
    """Generate sample data with technical indicators."""
    from trading_bot import TechnicalIndicators, TradingConfig
    
    config = TradingConfig()
    df = TechnicalIndicators.add_all_indicators(sample_ohlcv_data, config)
    
    return df


# Set environment variables for testing
@pytest.fixture(autouse=True)
def set_test_env(temp_db_path, temp_model_dir):
    """Set test environment variables."""
    old_db = os.environ.get("TRADING_BOT_DB")
    old_models = os.environ.get("TRADING_BOT_MODELS")
    
    os.environ["TRADING_BOT_DB"] = temp_db_path
    os.environ["TRADING_BOT_MODELS"] = temp_model_dir
    
    yield
    
    # Restore
    if old_db:
        os.environ["TRADING_BOT_DB"] = old_db
    else:
        os.environ.pop("TRADING_BOT_DB", None)
    
    if old_models:
        os.environ["TRADING_BOT_MODELS"] = old_models
    else:
        os.environ.pop("TRADING_BOT_MODELS", None)
