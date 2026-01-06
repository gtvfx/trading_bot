# Trading Bot Test Suite

Comprehensive test suite for the trading bot using pytest. **47 tests passing** ✅

## Running Tests

### Run all tests:
```powershell
cd r:\repo\trading_bot\py
pytest tests/
```

### Run with verbose output:
```powershell
pytest tests/ -v
```

### Run specific test file:
```powershell
pytest tests/test_trade_history.py
```

### Run specific test:
```powershell
pytest tests/test_trade_history.py::test_store_prediction
```

### Run with coverage:
```powershell
pytest tests/ --cov=trading_bot --cov-report=html
```

## Test Structure

- **conftest.py**: Shared fixtures and pytest configuration
- **test_init.py**: Package initialization and exports (8 tests)
- **test_config.py**: Trading configuration (18 tests)
- **test_trade_history.py**: Database persistence and data storage (8 tests)
- **test_indicators.py**: Technical indicator calculations (13 tests)

## Test Coverage

✅ **47 passing tests** covering:
- Package initialization and imports
- Configuration management (all dataclass fields)
- Environment variable handling (DB_PATH, MODEL_PATH)
- Database operations (market_data, predictions, trades)
- Technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic, ADX, VWAP, EMA)
- Data persistence across sessions

## Fixtures

Key fixtures available in conftest.py:

- `temp_db_path`: Temporary database for testing
- `temp_model_dir`: Temporary directory for model files
- `sample_ohlcv_data`: Sample OHLCV market data
- `sample_config`: Default trading configuration
- `mock_exchange_client`: Mock exchange for testing without API
- `sample_df_with_indicators`: Sample data with technical indicators
- `set_test_env`: Auto-configured test environment variables

## Test Results

```
tests/test_config.py::18 PASSED
tests/test_indicators.py::13 PASSED
tests/test_init.py::8 PASSED
tests/test_trade_history.py::8 PASSED
```

## Adding New Tests

1. Create test file: `test_<module_name>.py`
2. Import module: `from trading_bot import ModuleName`
3. Use fixtures: `def test_something(fixture_name):`
4. Assert behavior: `assert result == expected`
5. Run tests: `pytest tests/`

## CI/CD Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ --cov=trading_bot
```

## Test Requirements

Install test dependencies:
```powershell
pip install pytest pytest-cov
```

Dependencies from requirements.txt are automatically used:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- joblib >= 1.3.0

