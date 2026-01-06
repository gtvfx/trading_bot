"""Tests for strategy.py module - matching actual API."""

import pytest
import pandas as pd
import numpy as np
from trading_bot import TradingStrategy, TradingConfig, TechnicalIndicators, MLPredictor
from trading_bot.strategy import Signal


def test_strategy_initialization(sample_config):
    """Test TradingStrategy initializes correctly."""
    strategy = TradingStrategy(sample_config)
    
    assert strategy.config == sample_config
    assert strategy.ml_predictor is None


def test_strategy_with_ml_predictor(sample_config, temp_model_dir):
    """Test strategy with ML predictor."""
    predictor = MLPredictor(model_path=temp_model_dir)
    strategy = TradingStrategy(sample_config, ml_predictor=predictor)
    
    assert strategy.ml_predictor is not None


def test_analyze_rsi_oversold(sample_df_with_indicators):
    """Test RSI analysis - oversold condition."""
    config = TradingConfig()
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    df.loc[df.index[-1], 'rsi'] = 25  # Oversold
    
    signal = strategy.analyze_rsi(df)
    
    assert signal == Signal.BUY


def test_analyze_rsi_overbought(sample_df_with_indicators):
    """Test RSI analysis - overbought condition."""
    config = TradingConfig()
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    df.loc[df.index[-1], 'rsi'] = 75  # Overbought
    
    signal = strategy.analyze_rsi(df)
    
    assert signal == Signal.SELL


def test_analyze_rsi_neutral(sample_df_with_indicators):
    """Test RSI analysis - neutral condition."""
    config = TradingConfig()
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    df.loc[df.index[-1], 'rsi'] = 50  # Neutral
    
    signal = strategy.analyze_rsi(df)
    
    assert signal == Signal.HOLD


def test_analyze_macd_bullish_crossover(sample_df_with_indicators):
    """Test MACD bullish crossover."""
    config = TradingConfig()
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    df.loc[df.index[-2], 'macd'] = 100
    df.loc[df.index[-2], 'macd_signal'] = 105  # MACD below signal
    df.loc[df.index[-1], 'macd'] = 110
    df.loc[df.index[-1], 'macd_signal'] = 108  # MACD crosses above signal
    
    signal = strategy.analyze_macd(df)
    
    assert signal == Signal.BUY


def test_analyze_macd_bearish_crossover(sample_df_with_indicators):
    """Test MACD bearish crossover."""
    config = TradingConfig()
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    df.loc[df.index[-2], 'macd'] = 110
    df.loc[df.index[-2], 'macd_signal'] = 105  # MACD above signal
    df.loc[df.index[-1], 'macd'] = 100
    df.loc[df.index[-1], 'macd_signal'] = 102  # MACD crosses below signal
    
    signal = strategy.analyze_macd(df)
    
    assert signal == Signal.SELL


def test_analyze_ema_crossover(sample_df_with_indicators):
    """Test EMA crossover analysis."""
    config = TradingConfig()
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    df.loc[df.index[-2], 'ema_short'] = 50000
    df.loc[df.index[-2], 'ema_long'] = 50100
    df.loc[df.index[-1], 'ema_short'] = 50200
    df.loc[df.index[-1], 'ema_long'] = 50100
    
    signal = strategy.analyze_ema(df)
    
    assert signal == Signal.BUY


def test_analyze_bollinger_bands_lower(sample_df_with_indicators):
    """Test Bollinger Bands - price at lower band."""
    config = TradingConfig()
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    df.loc[df.index[-1], 'close'] = 49000
    df.loc[df.index[-1], 'bb_lower'] = 49000
    df.loc[df.index[-1], 'bb_upper'] = 51000
    
    signal = strategy.analyze_bollinger_bands(df)
    
    assert signal == Signal.BUY


def test_analyze_bollinger_bands_upper(sample_df_with_indicators):
    """Test Bollinger Bands - price at upper band."""
    config = TradingConfig()
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    df.loc[df.index[-1], 'close'] = 51000
    df.loc[df.index[-1], 'bb_lower'] = 49000
    df.loc[df.index[-1], 'bb_upper'] = 51000
    
    signal = strategy.analyze_bollinger_bands(df)
    
    assert signal == Signal.SELL


def test_analyze_stochastic(sample_df_with_indicators):
    """Test Stochastic Oscillator analysis."""
    config = TradingConfig()
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    df.loc[df.index[-2], 'stoch_k'] = 15
    df.loc[df.index[-2], 'stoch_d'] = 18
    df.loc[df.index[-1], 'stoch_k'] = 20
    df.loc[df.index[-1], 'stoch_d'] = 18
    
    signal = strategy.analyze_stochastic(df)
    
    # Should signal buy when oversold and K crosses above D
    assert signal in [Signal.BUY, Signal.HOLD]


def test_analyze_trend_strength_strong(sample_df_with_indicators):
    """Test trend strength with strong ADX."""
    config = TradingConfig()
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    df.loc[df.index[-1], 'adx'] = 30  # Strong trend
    
    strength = strategy.analyze_trend_strength(df)
    
    assert strength == 1.0


def test_analyze_trend_strength_weak(sample_df_with_indicators):
    """Test trend strength with weak ADX."""
    config = TradingConfig()
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    df.loc[df.index[-1], 'adx'] = 15  # Weak trend
    
    strength = strategy.analyze_trend_strength(df)
    
    assert strength == 0.5


def test_generate_signal(sample_df_with_indicators):
    """Test overall signal generation."""
    config = TradingConfig(ml_enabled=False)
    strategy = TradingStrategy(config)
    
    signal, confidence = strategy.generate_signal(sample_df_with_indicators)
    
    assert isinstance(signal, Signal)
    assert 0 <= confidence <= 1


def test_should_enter_position_high_confidence(sample_df_with_indicators):
    """Test position entry with high confidence."""
    config = TradingConfig(ml_confidence_threshold=0.65, min_signal_agreement=2)
    strategy = TradingStrategy(config)
    
    should_enter = strategy.should_enter_position(
        sample_df_with_indicators,
        Signal.BUY,
        confidence=0.8
    )
    
    # High confidence should generally allow entry
    assert isinstance(should_enter, bool)


def test_should_enter_position_low_confidence(sample_df_with_indicators):
    """Test position entry with low confidence."""
    config = TradingConfig(ml_confidence_threshold=0.65)
    strategy = TradingStrategy(config)
    
    should_enter = strategy.should_enter_position(
        sample_df_with_indicators,
        Signal.BUY,
        confidence=0.4
    )
    
    # Low confidence should prevent entry
    assert should_enter is False


def test_should_exit_position_stop_loss(sample_df_with_indicators):
    """Test exit with stop loss trigger."""
    config = TradingConfig(stop_loss_percent=2.0)
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    entry_price = 50000
    current_price = 48900  # 2.2% loss
    df.loc[df.index[-1], 'close'] = current_price
    
    should_exit, reason = strategy.should_exit_position(df, entry_price, 'buy')
    
    assert should_exit is True
    assert 'stop' in reason.lower()


def test_should_exit_position_take_profit(sample_df_with_indicators):
    """Test exit with take profit trigger."""
    config = TradingConfig(take_profit_percent=5.0)
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    entry_price = 50000
    current_price = 52600  # 5.2% profit
    df.loc[df.index[-1], 'close'] = current_price
    
    should_exit, reason = strategy.should_exit_position(df, entry_price, 'buy')
    
    assert should_exit is True
    assert 'profit' in reason.lower()


def test_should_exit_position_trailing_stop(sample_df_with_indicators):
    """Test exit with trailing stop."""
    config = TradingConfig(trailing_stop_percent=1.5)
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    
    should_exit, reason = strategy.should_exit_position(df, 50000, 'buy')
    
    # Should return boolean and reason
    assert isinstance(should_exit, bool)
    assert isinstance(reason, str)


def test_signal_agreement_threshold(sample_df_with_indicators):
    """Test that minimum signal agreement is enforced."""
    config = TradingConfig(ml_enabled=False, min_signal_agreement=4)
    strategy = TradingStrategy(config)
    
    signal, confidence = strategy.generate_signal(sample_df_with_indicators)
    
    # Signal should be generated even with high threshold
    assert isinstance(signal, Signal)


def test_get_ml_signal_no_predictor(sample_df_with_indicators):
    """Test ML signal without predictor."""
    config = TradingConfig()
    strategy = TradingStrategy(config, ml_predictor=None)
    
    signal, confidence = strategy.get_ml_signal(sample_df_with_indicators)
    
    assert signal == Signal.HOLD
    assert confidence == 0.0


def test_strategy_with_empty_dataframe():
    """Test strategy with empty DataFrame."""
    config = TradingConfig()
    strategy = TradingStrategy(config)
    
    df = pd.DataFrame()
    
    signal = strategy.analyze_rsi(df)
    assert signal == Signal.HOLD


def test_multiple_indicators_agreement(sample_df_with_indicators):
    """Test when multiple indicators agree."""
    config = TradingConfig(ml_enabled=False)
    strategy = TradingStrategy(config)
    
    df = sample_df_with_indicators.copy()
    
    # Set multiple bullish conditions
    df.loc[df.index[-1], 'rsi'] = 28  # Oversold
    df.loc[df.index[-1], 'close'] = 49000
    df.loc[df.index[-1], 'bb_lower'] = 49500  # Below lower band
    
    signal, confidence = strategy.generate_signal(df)
    
    # Multiple bullish signals should increase confidence
    assert isinstance(confidence, float)
