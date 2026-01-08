"""Tests for paper trading functionality."""

import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd
from datetime import datetime

from trading_bot import TradingBot, TradingConfig
from trading_bot.strategy import Signal


class TestPaperTrading:
    """Test paper trading mode."""
    
    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange client."""
        exchange = Mock()
        exchange.get_buying_power.return_value = 10000.0
        exchange.get_current_price.return_value = 50000.0
        exchange.get_account_balance.return_value = {'USD': 10000.0}
        exchange.get_holdings.return_value = {}
        exchange.place_market_order.return_value = {
            'id': 'test_order_123',
            'status': 'filled',
            'quantity': 0.001
        }
        
        # Mock historical candles with all required indicators
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
            'open': [50000.0] * 100,
            'high': [51000.0] * 100,
            'low': [49000.0] * 100,
            'close': [50000.0] * 100,
            'volume': [1000.0] * 100,
            'rsi': [50.0] * 100,
            'macd': [0.0] * 100,
            'macd_signal': [0.0] * 100,
            'macd_histogram': [0.0] * 100,
            'ema_short': [50000.0] * 100,
            'ema_long': [50000.0] * 100,
            'bb_upper': [51000.0] * 100,
            'bb_middle': [50000.0] * 100,
            'bb_lower': [49000.0] * 100,
            'atr': [500.0] * 100,
            'stoch_k': [50.0] * 100,
            'stoch_d': [50.0] * 100,
            'obv': [1000000.0] * 100,
            'vwap': [50000.0] * 100
        })
        exchange.get_historical_candles.return_value = df
        
        return exchange
    
    def test_paper_trading_config(self):
        """Test paper trading configuration."""
        config = TradingConfig(
            paper_trading=True,
            paper_trading_balance=10000.0
        )
        
        assert config.paper_trading is True
        assert config.paper_trading_balance == 10000.0
    
    def test_paper_trading_initialization(self, mock_exchange):
        """Test bot initializes with paper trading mode."""
        config = TradingConfig(
            paper_trading=True,
            paper_trading_balance=5000.0
        )
        
        bot = TradingBot(mock_exchange, config)
        
        assert bot.config.paper_trading is True
        assert bot.paper_balance == 5000.0
        assert bot.paper_equity == 0.0
    
    def test_paper_trading_no_real_orders(self, mock_exchange):
        """Test that paper trading doesn't place real orders."""
        config = TradingConfig(
            trading_pairs=["BTC-USD"],
            paper_trading=True,
            paper_trading_balance=10000.0,
            position_size_percent=10.0,
            ml_enabled=False  # Disable ML for simpler test
        )
        
        bot = TradingBot(mock_exchange, config)
        bot.initialize()
        
        # Mock strategy to generate buy signal
        bot.strategy.generate_signal = Mock(return_value=(Signal.BUY, 0.8))
        bot.strategy.should_enter_position = Mock(return_value=True)
        
        # Mock market data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'close': [50000.0] * 100
        })
        bot.market_data_cache['BTC-USD'] = df
        
        # Scan for opportunities (should create paper position)
        bot.scan_for_opportunities()
        
        # Verify real order was NOT placed
        mock_exchange.place_market_order.assert_not_called()
        
        # Verify position was created
        assert 'BTC-USD' in bot.positions
        assert bot.positions['BTC-USD'].entry_price == 50000.0
    
    def test_paper_balance_tracking(self, mock_exchange):
        """Test paper balance is properly tracked."""
        config = TradingConfig(
            trading_pairs=["BTC-USD"],
            paper_trading=True,
            paper_trading_balance=10000.0,
            position_size_percent=10.0,  # 10% = $1000
            ml_enabled=False  # Disable ML for simpler test
        )
        
        bot = TradingBot(mock_exchange, config)
        bot.initialize()
        
        initial_balance = bot.paper_balance
        
        # Mock strategy
        bot.strategy.generate_signal = Mock(return_value=(Signal.BUY, 0.8))
        bot.strategy.should_enter_position = Mock(return_value=True)
        
        # Mock market data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'close': [50000.0] * 100
        })
        bot.market_data_cache['BTC-USD'] = df
        
        # Open position
        bot.scan_for_opportunities()
        
        # Balance should decrease by 10%
        expected_used = initial_balance * 0.10
        assert bot.paper_balance == pytest.approx(initial_balance - expected_used, rel=0.01)
        assert bot.paper_equity == pytest.approx(expected_used, rel=0.01)
    
    def test_paper_trading_override(self, mock_exchange):
        """Test paper_trading can be overridden in run() method."""
        config = TradingConfig(
            paper_trading=False,  # Start with live trading disabled
            ml_enabled=False  # Disable ML for simpler test
        )
        
        bot = TradingBot(mock_exchange, config)
        
        # Mock to stop after initialization
        bot.run_iteration = Mock(return_value=False)
        
        # Override to paper trading
        bot.run(iterations=1, paper_trading=True)
        
        # Should now be in paper trading mode
        assert bot.config.paper_trading is True
        assert bot.paper_balance == config.paper_trading_balance
    
    def test_live_trading_uses_exchange(self, mock_exchange):
        """Test that live trading uses actual exchange."""
        config = TradingConfig(
            trading_pairs=["BTC-USD"],
            paper_trading=False,  # Live trading
            position_size_percent=10.0,
            ml_enabled=False  # Disable ML for simpler test
        )
        
        bot = TradingBot(mock_exchange, config)
        bot.initialize()
        
        # Mock strategy
        bot.strategy.generate_signal = Mock(return_value=(Signal.BUY, 0.8))
        bot.strategy.should_enter_position = Mock(return_value=True)
        
        # Mock market data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'close': [50000.0] * 100
        })
        bot.market_data_cache['BTC-USD'] = df
        
        # Scan for opportunities
        bot.scan_for_opportunities()
        
        # Verify real order WAS placed
        mock_exchange.place_market_order.assert_called_once()
        
        # Verify paper balance not used
        assert bot.paper_balance == 0.0
        assert bot.paper_equity == 0.0
    
    def test_paper_position_closing(self, mock_exchange):
        """Test closing positions in paper trading mode."""
        config = TradingConfig(
            trading_pairs=["BTC-USD"],
            paper_trading=True,
            paper_trading_balance=10000.0,
            position_size_percent=10.0,
            ml_enabled=False  # Disable ML for simpler test
        )
        
        bot = TradingBot(mock_exchange, config)
        bot.initialize()
        
        # Open a position
        from trading_bot.bot import Position
        position = Position(
            symbol="BTC-USD",
            side="buy",
            quantity=0.02,
            entry_price=50000.0,
            entry_time=datetime.now()
        )
        bot.positions["BTC-USD"] = position
        
        # Track initial balances
        initial_balance = bot.paper_balance
        initial_equity = position.quantity * position.entry_price
        bot.paper_equity = initial_equity
        
        # Close position at higher price (profit)
        mock_exchange.get_current_price.return_value = 55000.0
        
        bot._close_position("BTC-USD", "test_close")
        
        # Balance should increase (initial + position value at new price)
        expected_balance = initial_balance + (position.quantity * 55000.0)
        assert bot.paper_balance == pytest.approx(expected_balance, rel=0.01)
        assert bot.paper_equity == pytest.approx(0.0, rel=0.01)
        
        # Verify real order was NOT placed
        mock_exchange.place_market_order.assert_not_called()
