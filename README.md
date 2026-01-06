# trading_bot
API agnostic trading bot that provides an abstract interface to integrate with exchange-specific trading interfaces.

## Features

- **ML-Powered Predictions**: Random Forest and Gradient Boosting models for price movement prediction
- **Technical Analysis**: RSI, MACD, Bollinger Bands, EMA, Stochastic, ADX, VWAP, ATR, OBV
- **Risk Management**: Stop loss, take profit, trailing stops, position sizing
- **Trade History**: SQLite database tracking predictions, trades, and performance
- **Performance Analysis**: Model performance metrics, trade statistics, confidence calibration
- **Exchange Agnostic**: Abstract interface for multiple exchange integrations

## Testing

The project includes a comprehensive test suite covering all components. To run tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_config.py -v

# Run with coverage
pytest tests/ --cov=trading_bot --cov-report=html
```

**Current test status: 103 tests passing** ✅

### Test Coverage by Component

- **Configuration** (`test_config.py`): 18 tests - Config validation, defaults, environment variables
- **Indicators** (`test_indicators.py`): 13 tests - RSI, MACD, Bollinger Bands, EMA, ATR, Stochastic, ADX, VWAP
- **Trade History** (`test_trade_history.py`): 8 tests - Database operations, persistence, trade tracking
- **ML Predictor** (`test_ml_predictor.py`): 17 tests - Model training, prediction, save/load, feature engineering
- **Trading Strategy** (`test_strategy.py`): 23 tests - Signal generation, risk management, entry/exit logic
- **Data Curator** (`test_data_curator.py`): 16 tests - Performance analysis, reporting, data export
- **Package Structure** (`test_init.py`): 8 tests - Imports, versioning, module structure
