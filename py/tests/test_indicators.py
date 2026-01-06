"""Tests for indicators.py module - matching actual API."""

import pytest
import pandas as pd
import numpy as np
from trading_bot import TechnicalIndicators, TradingConfig


def test_calculate_rsi(sample_ohlcv_data):
    """Test RSI calculation."""
    df = sample_ohlcv_data.copy()
    rsi = TechnicalIndicators.calculate_rsi(df, period=14)
    
    # RSI should be between 0 and 100
    rsi_values = rsi.dropna()
    assert len(rsi_values) > 0
    assert (rsi_values >= 0).all()
    assert (rsi_values <= 100).all()


def test_calculate_macd(sample_ohlcv_data):
    """Test MACD calculation."""
    df = sample_ohlcv_data.copy()
    macd, signal, hist = TechnicalIndicators.calculate_macd(df, fast=12, slow=26, signal=9)
    
    # Check that values exist (not all NaN)
    assert macd.notna().any()
    assert signal.notna().any()
    assert hist.notna().any()


def test_calculate_bollinger_bands(sample_ohlcv_data):
    """Test Bollinger Bands calculation."""
    df = sample_ohlcv_data.copy()
    upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(df, period=20, std_dev=2.0)
    
    # Upper band should be above lower band
    valid_mask = upper.notna() & lower.notna()
    assert (upper[valid_mask] > lower[valid_mask]).all()


def test_calculate_ema(sample_ohlcv_data):
    """Test EMA calculation."""
    df = sample_ohlcv_data.copy()
    ema = TechnicalIndicators.calculate_ema(df, period=9)
    
    assert ema.notna().any()


def test_calculate_atr(sample_ohlcv_data):
    """Test ATR (Average True Range) calculation."""
    df = sample_ohlcv_data.copy()
    atr = TechnicalIndicators.calculate_atr(df, period=14)
    
    # ATR should be positive
    atr_values = atr.dropna()
    assert len(atr_values) > 0
    assert (atr_values > 0).all()


def test_calculate_obv(sample_ohlcv_data):
    """Test OBV (On-Balance Volume) calculation."""
    df = sample_ohlcv_data.copy()
    obv = TechnicalIndicators.calculate_obv(df)
    
    assert obv.notna().any()


def test_calculate_stochastic(sample_ohlcv_data):
    """Test Stochastic Oscillator calculation."""
    df = sample_ohlcv_data.copy()
    k, d = TechnicalIndicators.calculate_stochastic(df, k_period=14, d_period=3)
    
    # Stochastic should be between 0 and 100
    k_values = k.dropna()
    d_values = d.dropna()
    
    assert len(k_values) > 0
    assert (k_values >= 0).all() and (k_values <= 100).all()
    assert (d_values >= 0).all() and (d_values <= 100).all()


def test_add_all_indicators(sample_ohlcv_data):
    """Test adding all indicators at once."""
    config = TradingConfig()
    df = TechnicalIndicators.add_all_indicators(sample_ohlcv_data.copy(), config)
    
    # Check that major indicators are present
    expected_indicators = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower',
        'ema_short', 'ema_long', 'atr', 'obv',
        'stoch_k', 'stoch_d'
    ]
    
    for indicator in expected_indicators:
        assert indicator in df.columns


def test_indicators_with_minimal_data():
    """Test indicators with minimal data points."""
    config = TradingConfig()
    
    # Create minimal dataset
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
        'open': [100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    # Should handle gracefully without errors
    result = TechnicalIndicators.add_all_indicators(df, config)
    assert isinstance(result, pd.DataFrame)


def test_rsi_different_periods(sample_ohlcv_data):
    """Test RSI with different periods."""
    df = sample_ohlcv_data.copy()
    
    rsi_7 = TechnicalIndicators.calculate_rsi(df, period=7)
    rsi_14 = TechnicalIndicators.calculate_rsi(df, period=14)
    
    # Different periods should produce different results
    # At least some values should differ
    assert not rsi_7.equals(rsi_14)


def test_calculate_vwap(sample_ohlcv_data):
    """Test VWAP calculation."""
    df = sample_ohlcv_data.copy()
    vwap = TechnicalIndicators.calculate_vwap(df)
    
    assert vwap.notna().any()


def test_calculate_adx(sample_ohlcv_data):
    """Test ADX calculation."""
    df = sample_ohlcv_data.copy()
    adx = TechnicalIndicators.calculate_adx(df, period=14)
    
    # ADX should be between 0 and 100
    adx_values = adx.dropna()
    if len(adx_values) > 0:
        assert (adx_values >= 0).all()
        assert (adx_values <= 100).all()


def test_bollinger_band_width(sample_ohlcv_data):
    """Test Bollinger Band width calculation."""
    df = sample_ohlcv_data.copy()
    upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(df)
    
    # Calculate band width
    width = upper - lower
    
    # Width should be positive
    assert (width.dropna() > 0).all()
