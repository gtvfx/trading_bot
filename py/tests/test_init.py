"""Test package initialization and exports."""

import pytest


def test_package_imports():
    """Test that all main classes can be imported from package."""
    from trading_bot import (
        TradingBot,
        TradingConfig,
        TechnicalIndicators,
        TradingStrategy,
        MLPredictor,
        TradeHistory,
        DataCurator
    )
    
    # All imports should succeed
    assert TradingBot is not None
    assert TradingConfig is not None
    assert TechnicalIndicators is not None
    assert TradingStrategy is not None
    assert MLPredictor is not None
    assert TradeHistory is not None
    assert DataCurator is not None


def test_constants_import():
    """Test that constants can be imported from package."""
    from trading_bot import DB_PATH, MODEL_PATH
    
    assert isinstance(DB_PATH, str)
    assert isinstance(MODEL_PATH, str)


def test_version_exists():
    """Test that package version is defined."""
    import trading_bot
    
    assert hasattr(trading_bot, '__version__')
    assert isinstance(trading_bot.__version__, str)
    assert len(trading_bot.__version__) > 0


def test_version_format():
    """Test that version follows semantic versioning."""
    import trading_bot
    import re
    
    version = trading_bot.__version__
    
    # Should match semantic versioning or dev version
    # Examples: "1.2.3", "0.0.0", "1.0.0.dev0+g1234567"
    pattern = r'^\d+\.\d+\.\d+'
    
    assert re.match(pattern, version), f"Version '{version}' doesn't match expected format"


def test_all_exports():
    """Test that __all__ contains expected exports."""
    import trading_bot
    
    if hasattr(trading_bot, '__all__'):
        expected_exports = [
            'TradingBot',
            'TradingConfig',
            'TechnicalIndicators',
            'TradingStrategy',
            'MLPredictor',
            'TradeHistory',
            'DataCurator',
            'DB_PATH',
            'MODEL_PATH'
        ]
        
        for export in expected_exports:
            assert export in trading_bot.__all__, f"{export} not in __all__"


def test_no_import_errors():
    """Test that importing package doesn't raise errors."""
    try:
        import trading_bot
        # If we get here, import succeeded
        assert True
    except Exception as e:
        pytest.fail(f"Failed to import trading_bot: {e}")


def test_package_structure():
    """Test that package has expected structure."""
    import trading_bot
    import os
    from pathlib import Path
    
    # Get package directory
    package_dir = Path(trading_bot.__file__).parent
    
    # Check for expected files
    expected_files = [
        '__init__.py',
        '_constants.py',
        'bot.py',
        'config.py',
        'indicators.py',
        'strategy.py',
        'ml_predictor.py',
        'trade_history.py',
        'data_curator.py'
    ]
    
    for filename in expected_files:
        file_path = package_dir / filename
        assert file_path.exists(), f"Expected file not found: {filename}"


def test_submodule_imports():
    """Test that submodules can be imported."""
    from trading_bot import config
    from trading_bot import indicators
    from trading_bot import strategy
    from trading_bot import ml_predictor
    from trading_bot import trade_history
    from trading_bot import data_curator
    
    assert config is not None
    assert indicators is not None
    assert strategy is not None
    assert ml_predictor is not None
    assert trade_history is not None
    assert data_curator is not None
