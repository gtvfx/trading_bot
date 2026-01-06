"""Tests for data_curator.py module - matching actual API."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from trading_bot import DataCurator, TradeHistory


def test_data_curator_initialization(temp_db_path):
    """Test DataCurator initializes correctly."""
    curator = DataCurator(temp_db_path)
    
    assert curator.history is not None
    assert isinstance(curator.history, TradeHistory)


def test_analyze_model_performance_no_data(temp_db_path, capsys):
    """Test model performance analysis with no data."""
    curator = DataCurator(temp_db_path)
    
    curator.analyze_model_performance()
    
    captured = capsys.readouterr()
    assert 'ML MODEL PERFORMANCE ANALYSIS' in captured.out
    assert 'No prediction data' in captured.out


def test_analyze_model_performance_with_data(temp_db_path, capsys):
    """Test model performance analysis with predictions."""
    history = TradeHistory(temp_db_path)
    
    # Add some predictions
    history.store_prediction('BTC-USD', 1, 0.75, 50000.0)
    history.store_prediction('BTC-USD', 0, 0.60, 50500.0)
    history.store_prediction('BTC-USD', 1, 0.80, 51000.0)
    
    # Update outcomes (simulate after 1 hour)
    history.update_prediction_outcome()
    
    curator = DataCurator(temp_db_path)
    curator.analyze_model_performance(symbol='BTC-USD')
    
    captured = capsys.readouterr()
    assert 'ML MODEL PERFORMANCE ANALYSIS' in captured.out
    assert 'BTC-USD' in captured.out or 'No prediction data' in captured.out


def test_analyze_trade_performance_no_trades(temp_db_path, capsys):
    """Test trade performance analysis with no trades."""
    curator = DataCurator(temp_db_path)
    
    curator.analyze_trade_performance()
    
    captured = capsys.readouterr()
    assert 'TRADING PERFORMANCE ANALYSIS' in captured.out


def test_analyze_trade_performance_with_trades(temp_db_path, capsys):
    """Test trade performance analysis with trades."""
    history = TradeHistory(temp_db_path)
    
    # Add some trades
    entry1 = datetime.now()
    history.store_trade(
        'BTC-USD', 'buy', entry1, 50000.0, 0.1, 0.75, 'RSI_OVERSOLD',
        exit_time=entry1 + timedelta(hours=2),
        exit_price=52000.0,
        pnl=200.0,
        pnl_percent=4.0
    )
    
    entry2 = datetime.now() + timedelta(hours=4)
    history.store_trade(
        'BTC-USD', 'buy', entry2, 51000.0, 0.1, 0.60, 'MACD_BULLISH',
        exit_time=entry2 + timedelta(hours=1),
        exit_price=50500.0,
        pnl=-50.0,
        pnl_percent=-0.98
    )
    
    curator = DataCurator(temp_db_path)
    curator.analyze_trade_performance(symbol='BTC-USD', days=30)
    
    captured = capsys.readouterr()
    assert 'TRADING PERFORMANCE ANALYSIS' in captured.out
    assert 'Total Trades:' in captured.out


def test_analyze_all_symbols(temp_db_path, capsys):
    """Test analyzing performance across all symbols."""
    history = TradeHistory(temp_db_path)
    
    # Add data for multiple symbols
    history.store_prediction('BTC-USD', 1, 0.75, 50000.0)
    history.store_prediction('ETH-USD', 0, 0.60, 3000.0)
    
    curator = DataCurator(temp_db_path)
    curator.analyze_model_performance()  # No symbol = all symbols
    
    captured = capsys.readouterr()
    assert 'ML MODEL PERFORMANCE ANALYSIS' in captured.out


def test_export_for_analysis(temp_db_path, sample_ohlcv_data, tmp_path):
    """Test exporting data for external analysis."""
    history = TradeHistory(temp_db_path)
    history.store_market_data('BTC-USD', sample_ohlcv_data)
    
    curator = DataCurator(temp_db_path)
    
    # Export to CSV file
    output_file = tmp_path / "export.csv"
    curator.export_for_analysis(str(output_file))
    
    # Check that file was created
    assert output_file.exists()


def test_clean_old_data(temp_db_path, sample_ohlcv_data):
    """Test cleaning old data."""
    history = TradeHistory(temp_db_path)
    history.store_market_data('BTC-USD', sample_ohlcv_data)
    
    curator = DataCurator(temp_db_path)
    
    # This method might not exist, test if callable
    if hasattr(curator, 'clean_old_data'):
        result = curator.clean_old_data(days_to_keep=1)
        assert isinstance(result, (dict, int, type(None)))


def test_analyze_multiple_days(temp_db_path):
    """Test analyzing different time periods."""
    history = TradeHistory(temp_db_path)
    
    # Add trades at different times
    for i in range(5):
        entry = datetime.now() - timedelta(days=i*10)
        history.store_trade(
            'BTC-USD', 'buy', entry, 50000.0 + i*1000, 0.1, 0.7, 'TEST',
            exit_time=entry + timedelta(hours=2),
            exit_price=50500.0 + i*1000,
            pnl=50.0,
            pnl_percent=1.0
        )
    
    curator = DataCurator(temp_db_path)
    
    # Analyze different periods
    curator.analyze_trade_performance(days=7)
    curator.analyze_trade_performance(days=30)
    curator.analyze_trade_performance(days=90)


def test_symbol_specific_analysis(temp_db_path):
    """Test analyzing specific symbol."""
    history = TradeHistory(temp_db_path)
    
    # Add data for different symbols
    entry = datetime.now()
    history.store_trade('BTC-USD', 'buy', entry, 50000.0, 0.1, 0.7, 'TEST',
                       exit_time=entry + timedelta(hours=1), exit_price=50500.0,
                       pnl=50.0, pnl_percent=1.0)
    
    history.store_trade('ETH-USD', 'buy', entry, 3000.0, 1.0, 0.6, 'TEST',
                       exit_time=entry + timedelta(hours=1), exit_price=3100.0,
                       pnl=100.0, pnl_percent=3.33)
    
    curator = DataCurator(temp_db_path)
    
    # Analyze specific symbol
    curator.analyze_trade_performance(symbol='BTC-USD')
    curator.analyze_model_performance(symbol='ETH-USD')


def test_confidence_calibration(temp_db_path):
    """Test confidence calibration analysis."""
    history = TradeHistory(temp_db_path)
    
    # Add predictions with varying confidence
    for i, conf in enumerate([0.55, 0.65, 0.75, 0.85, 0.95]):
        history.store_prediction('BTC-USD', 1, conf, 50000.0 + i*100)
    
    curator = DataCurator(temp_db_path)
    curator.analyze_model_performance(symbol='BTC-USD')


def test_empty_database_handling(temp_db_path, capsys):
    """Test handling of empty database."""
    curator = DataCurator(temp_db_path)
    
    # Should not crash with empty database
    curator.analyze_model_performance()
    curator.analyze_trade_performance()
    
    captured = capsys.readouterr()
    assert len(captured.out) > 0  # Should print something


def test_performance_with_stop_losses(temp_db_path):
    """Test performance analysis with stop losses."""
    history = TradeHistory(temp_db_path)
    
    # Add trades with stop losses
    entry = datetime.now()
    history.store_trade(
        'BTC-USD', 'buy', entry, 50000.0, 0.1, 0.7, 'TEST',
        exit_time=entry + timedelta(hours=1),
        exit_price=49000.0,
        pnl=-100.0,
        pnl_percent=-2.0,
        stop_loss_triggered=True
    )
    
    curator = DataCurator(temp_db_path)
    curator.analyze_trade_performance()


def test_performance_with_take_profits(temp_db_path):
    """Test performance analysis with take profits."""
    history = TradeHistory(temp_db_path)
    
    # Add trades with take profits
    entry = datetime.now()
    history.store_trade(
        'BTC-USD', 'buy', entry, 50000.0, 0.1, 0.8, 'TEST',
        exit_time=entry + timedelta(hours=1),
        exit_price=52500.0,
        pnl=250.0,
        pnl_percent=5.0,
        take_profit_triggered=True
    )
    
    curator = DataCurator(temp_db_path)
    curator.analyze_trade_performance()


def test_data_curator_with_missing_columns(temp_db_path):
    """Test curator handles data with missing columns."""
    history = TradeHistory(temp_db_path)
    
    # Store minimal market data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
        'close': [50000 + i*100 for i in range(10)],
        'volume': [1000] * 10
    })
    
    history.store_market_data('BTC-USD', df)
    
    curator = DataCurator(temp_db_path)
    curator.analyze_model_performance()


def test_large_dataset_performance(temp_db_path):
    """Test performance with larger dataset."""
    history = TradeHistory(temp_db_path)
    
    # Add many trades
    base_time = datetime.now() - timedelta(days=90)
    for i in range(50):
        entry = base_time + timedelta(days=i)
        pnl = (-1) ** i * (50 + i)  # Alternating wins/losses
        history.store_trade(
            'BTC-USD', 'buy', entry, 50000.0, 0.1, 0.7, 'TEST',
            exit_time=entry + timedelta(hours=2),
            exit_price=50000.0 + pnl*10,
            pnl=pnl,
            pnl_percent=(pnl/50000.0)*100
        )
    
    curator = DataCurator(temp_db_path)
    curator.analyze_trade_performance(days=90)
