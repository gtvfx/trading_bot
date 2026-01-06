"""Simplified tests for trade_history.py - matching actual API."""

import pytest
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from trading_bot import TradeHistory


def test_trade_history_initialization(temp_db_path):
    """Test TradeHistory initializes database correctly."""
    history = TradeHistory(temp_db_path)
    
    # Check tables exist
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    assert 'market_data' in tables
    assert 'predictions' in tables
    assert 'trades' in tables
    conn.close()


def test_store_market_data(temp_db_path, sample_ohlcv_data):
    """Test storing market data."""
    history = TradeHistory(temp_db_path)
    
    # Store data
    history.store_market_data('BTC-USD', sample_ohlcv_data)
    
    # Verify data was stored
    conn = sqlite3.connect(temp_db_path)
    query = "SELECT COUNT(*) FROM market_data WHERE symbol = 'BTC-USD'"
    count = pd.read_sql_query(query, conn).iloc[0, 0]
    conn.close()
    
    assert count == len(sample_ohlcv_data)


def test_store_prediction(temp_db_path):
    """Test storing predictions."""
    history = TradeHistory(temp_db_path)
    
    history.store_prediction(
        symbol='BTC-USD',
        prediction=1,
        confidence=0.75,
        current_price=50000.0
    )
    
    # Verify prediction was stored
    conn = sqlite3.connect(temp_db_path)
    query = "SELECT * FROM predictions WHERE symbol = 'BTC-USD'"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    assert len(df) == 1
    assert df.iloc[0]['symbol'] == 'BTC-USD'
    assert df.iloc[0]['prediction'] == 1
    assert df.iloc[0]['confidence'] == 0.75


def test_store_trade(temp_db_path):
    """Test storing trade information."""
    history = TradeHistory(temp_db_path)
    
    entry_time = datetime.now()
    history.store_trade(
        symbol='BTC-USD',
        side='buy',
        entry_time=entry_time,
        entry_price=50000.0,
        quantity=0.1,
        ml_confidence=0.75,
        technical_signals='RSI_OVERSOLD,MACD_BULLISH'
    )
    
    # Verify trade was stored
    conn = sqlite3.connect(temp_db_path)
    query = "SELECT * FROM trades WHERE symbol = 'BTC-USD'"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    assert len(df) == 1
    assert df.iloc[0]['symbol'] == 'BTC-USD'
    assert df.iloc[0]['side'] == 'buy'
    assert df.iloc[0]['quantity'] == 0.1
    assert df.iloc[0]['entry_price'] == 50000.0


def test_update_trade_exit(temp_db_path):
    """Test updating trade exit information."""
    history = TradeHistory(temp_db_path)
    
    # Create a trade
    entry_time = datetime.now()
    history.store_trade(
        symbol='BTC-USD',
        side='buy',
        entry_time=entry_time,
        entry_price=50000.0,
        quantity=0.1,
        ml_confidence=0.75,
        technical_signals='RSI_OVERSOLD'
    )
    
    # Update with exit
    exit_time = entry_time + timedelta(hours=2)
    history.update_trade_exit(
        symbol='BTC-USD',
        entry_time=entry_time,
        exit_time=exit_time,
        exit_price=52000.0,
        pnl=200.0,
        pnl_percent=4.0,
        take_profit_triggered=True
    )
    
    # Verify update
    conn = sqlite3.connect(temp_db_path)
    query = "SELECT exit_price, pnl, pnl_percent FROM trades WHERE symbol = 'BTC-USD'"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    assert df.iloc[0]['exit_price'] == 52000.0
    assert df.iloc[0]['pnl'] == pytest.approx(200.0, rel=1e-2)
    assert df.iloc[0]['pnl_percent'] == pytest.approx(4.0, rel=1e-2)


def test_get_training_data(temp_db_path, sample_ohlcv_data):
    """Test retrieving training data."""
    history = TradeHistory(temp_db_path)
    
    # Store market data
    history.store_market_data('BTC-USD', sample_ohlcv_data)
    
    # Retrieve training data
    df = history.get_training_data(symbol='BTC-USD')
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_get_trade_stats(temp_db_path):
    """Test calculating trade statistics."""
    history = TradeHistory(temp_db_path)
    
    # Create some trades
    entry1 = datetime.now()
    history.store_trade(
        'BTC-USD', 'buy', entry1, 50000.0, 0.1, 0.75, 'RSI',
        exit_time=entry1 + timedelta(hours=1),
        exit_price=52000.0,
        pnl=200.0,
        pnl_percent=4.0
    )
    
    entry2 = datetime.now() + timedelta(hours=3)
    history.store_trade(
        'ETH-USD', 'buy', entry2, 3000.0, 1.0, 0.60, 'MACD',
        exit_time=entry2 + timedelta(hours=1),
        exit_price=2900.0,
        pnl=-100.0,
        pnl_percent=-3.33
    )
    
    # Get stats
    stats = history.get_trade_stats(days=30)
    
    assert stats is not None
    assert stats['total_trades'] == 2
    assert stats['winning_trades'] == 1
    assert stats['losing_trades'] == 1


def test_database_persistence(temp_db_path, sample_ohlcv_data):
    """Test that data persists across TradeHistory instances."""
    # Create first instance and store data
    history1 = TradeHistory(temp_db_path)
    history1.store_market_data('BTC-USD', sample_ohlcv_data)
    del history1
    
    # Create second instance and verify data exists
    history2 = TradeHistory(temp_db_path)
    df = history2.get_training_data(symbol='BTC-USD')
    
    assert len(df) == len(sample_ohlcv_data)
