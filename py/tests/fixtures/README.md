# Test Fixtures

This directory contains static test data files used by the test suite.

## Files

- **BTC-USD.csv** - Sample historical OHLCV data for Bitcoin
  - 100 hours of data starting from 2024-01-01
  - Contains columns: id, symbol, timestamp, open, high, low, close, volume
  - Technical indicator columns (rsi, macd, ema, etc.) are present but empty
  - Used for testing data processing and technical analysis functions

## Usage

These fixtures can be loaded in tests when you need realistic market data without generating it dynamically:

```python
import pandas as pd
from pathlib import Path

fixtures_dir = Path(__file__).parent / 'fixtures'
btc_data = pd.read_csv(fixtures_dir / 'BTC-USD.csv')
```

However, most tests should use the dynamic fixtures defined in `conftest.py` (like `sample_ohlcv_data`) instead of these static files, as they're faster and more flexible.

## Note

These are **safe to commit** - they contain only synthetic/test data with no real trading information.
