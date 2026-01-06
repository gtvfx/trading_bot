# Exchange-Agnostic Trading Bot

A flexible, ML-powered algorithmic trading bot that works with any cryptocurrency exchange through adapter interfaces.

## Features

- **Exchange Agnostic**: Works with any exchange via adapter pattern (Coinbase, Robinhood, etc.)
- **Technical Indicators**: RSI, MACD, EMA, Bollinger Bands, Stochastic, ADX, OBV, VWAP
- **Machine Learning**: Ensemble models (Random Forest, Gradient Boosting) for price prediction
- **Persistent Learning**: Models and predictions saved to SQLite for continuous improvement
- **Data Curation**: Track prediction accuracy and refine training data over time
- **Risk Management**: Stop-loss, take-profit, position sizing, daily loss limits
- **Signal Generation**: Majority voting across multiple indicators with confidence scores
- **Scalable Architecture**: Easy to add new exchanges and indicators

## Architecture

```
trading_bot/
├── exchange_client.py    # Abstract interface for exchanges
├── config.py            # Trading configuration
├── indicators.py        # Technical indicators
├── ml_predictor.py      # ML models for prediction
├── strategy.py          # Signal generation logic
├── bot.py               # Main trading bot
├── trade_history.py     # SQLite persistence
└── data_curator.py      # Analysis & curation tools
```

## Quick Start

### 1. Create an Exchange Adapter

Implement the `ExchangeClient` interface for your exchange:

```python
from trading_bot.exchange_client import ExchangeClient
from datetime import datetime
import pandas as pd

class MyExchangeAdapter(ExchangeClient):
    def __init__(self, api_key, api_secret):
        self.client = MyExchangeAPI(api_key, api_secret)
    
    def get_account_balance(self):
        # Implement using your exchange's API
        return {'USD': 10000.0, 'BTC': 0.5}
    
    def get_buying_power(self):
        return self.client.get_cash_balance()
    
    def get_current_price(self, symbol):
        return self.client.get_ticker(symbol)['price']
    
    def get_historical_candles(self, symbol, start, end, granularity):
        # Return DataFrame with: [timestamp, open, high, low, close, volume]
        return self.client.get_candles(symbol, start, end)
    
    def place_market_order(self, symbol, side, quantity):
        return self.client.create_order(symbol, 'market', side, quantity)
    
    # ... implement other required methods
```

### 2. Configure the Bot

```python
from trading_bot import TradingConfig

config = TradingConfig(
    trading_pairs=["BTC-USD", "ETH-USD"],
    position_size_percent=10.0,
    max_positions=3,
    stop_loss_percent=2.0,
    take_profit_percent=5.0,
    ml_enabled=True,
    ml_confidence_threshold=0.65,
    candle_granularity="1h",
    refresh_interval=60
)
```

### 3. Run the Bot

```python
from trading_bot import TradingBot

# Initialize your exchange adapter
exchange = MyExchangeAdapter(api_key, api_secret)

# Create and run bot
bot = TradingBot(exchange, config)
bot.run()  # Run indefinitely, or bot.run(iterations=10) for fixed runs
```

## Configuration Options

### Position Management
- `position_size_percent`: % of buying power per trade (default: 10%)
- `max_positions`: Maximum concurrent positions (default: 5)

### Risk Management
- `stop_loss_percent`: % loss to trigger stop (default: 2%)
- `take_profit_percent`: % gain to trigger take profit (default: 5%)
- `max_daily_loss_percent`: Max daily loss before stopping (default: 5%)

### Indicators
- `rsi_period`, `rsi_oversold`, `rsi_overbought`: RSI settings
- `macd_fast`, `macd_slow`, `macd_signal`: MACD settings
- `ema_short`, `ema_long`: EMA crossover settings
- `bb_period`, `bb_std_dev`: Bollinger Bands settings

### Machine Learning
- `ml_enabled`: Enable ML predictions (default: True)
- `ml_model_type`: "random_forest" or "gradient_boosting"
- `ml_confidence_threshold`: Min confidence to act (default: 0.65)
- `ml_retrain_interval`: Retrain every N candles (default: 100)
- `ml_training_days`: Days of historical data for training (default: 90)

### Persistence
- `db_path`: SQLite database path (default: "data/trades.db", env: `TRADING_BOT_DB`)
- `model_path`: Directory for saved models (default: "models", env: `TRADING_BOT_MODELS`)

**Environment Variables:**
```bash
# Override database location
export TRADING_BOT_DB="/path/to/my/trades.db"

# Override model storage directory
export TRADING_BOT_MODELS="/path/to/my/models"

# Then run bot (will use environment paths)
python my_bot.py
```

### Strategy
- `min_signal_agreement`: Minimum indicators that must agree (default: 3)
- `lookback_window`: Historical candles to analyze (default: 100)

## Data Persistence & Continuous Learning

The bot stores all market data, predictions, and trades in SQLite for continuous learning and performance tracking.

### What Gets Stored

1. **Market Data**: All candles with calculated technical indicators
2. **ML Predictions**: Every prediction with confidence score and actual outcomes
3. **Trade History**: Complete record of entries, exits, P&L, and exit reasons

### Model Persistence

Models are automatically saved after training and loaded on startup:

```python
# Models load automatically on initialization
bot = TradingBot(exchange, config)
bot.initialize()  # Loads existing models if available

# Models improve over time as more data is collected
# Automatic retraining every ml_retrain_interval iterations
```

### Performance Analysis

Use the `DataCurator` tool to analyze performance:

```python
from trading_bot import DataCurator

curator = DataCurator()

# View data summary
curator.get_data_summary()

# Analyze ML model accuracy
curator.analyze_model_performance(symbol="BTC-USD")

# Analyze trading performance
curator.analyze_trade_performance(days=30)

# Export data for external analysis
curator.export_for_analysis("my_data.csv")
```

### Command Line Analysis

```bash
# Show data summary
python -m trading_bot.data_curator --summary

# Analyze model performance
python -m trading_bot.data_curator --model --symbol BTC-USD

# Analyze trades
python -m trading_bot.data_curator --trades --days 30

# Export data
python -m trading_bot.data_curator --export data.csv

# Clean old data (keep last 180 days)
python -m trading_bot.data_curator --clean 180
```

### Database Structure

**Location**: `data/trades.db` (SQLite)

Tables:
- `market_data`: OHLCV candles with all technical indicators
- `predictions`: ML predictions with outcomes and accuracy
- `trades`: Complete trade history with P&L

**Models**: Saved to `models/*.joblib` (per-symbol models)

### Continuous Improvement

The bot learns continuously:

1. **Initial Training**: Uses historical data from database if available
2. **Live Trading**: Stores all new market data and predictions
3. **Validation**: Updates prediction outcomes after 1 hour
4. **Retraining**: Periodically retrains on growing dataset
5. **Analysis**: Track accuracy trends and adjust strategy

**Recommendation**: Run the bot for at least 2-4 weeks to accumulate meaningful data before relying heavily on ML predictions. Initial models train on limited data and improve significantly with real market exposure.

## Signal Generation

The bot uses majority voting across multiple indicators:

1. **RSI**: Oversold/overbought conditions
2. **MACD**: Trend following and momentum
3. **EMA Crossover**: Short-term vs long-term trends
4. **Bollinger Bands**: Volatility and mean reversion
5. **Stochastic**: Momentum and reversal points
6. **ML Prediction**: Ensemble model predictions (weighted 2x if high confidence)

A position is entered when:
- Minimum `min_signal_agreement` indicators agree
- Confidence score > 60%
- Trend strength (ADX) > 0.5

## Exchange Adapters

### Coinbase Adapter Example

```python
from trading_bot.exchange_client import ExchangeClient
from cb.trading import CoinbaseClient

class CoinbaseAdapter(ExchangeClient):
    def __init__(self, coinbase_client):
        self.client = coinbase_client
    
    def get_account_balance(self):
        holdings = self.client.get_holdings()
        return {h['asset_code']: float(h['total_quantity']) 
                for h in holdings.get('results', [])}
    
    def get_buying_power(self):
        account = self.client.get_account()
        return float(account.get('buying_power', 0))
    
    # ... implement other methods
```

### Robinhood Adapter Example

```python
from trading_bot.exchange_client import ExchangeClient
from robinhood.trading import RobinhoodClient

class RobinhoodAdapter(ExchangeClient):
    def __init__(self, robinhood_client):
        self.client = robinhood_client
    
    def get_account_balance(self):
        holdings = self.client.get_holdings()
        return {h['asset_code']: float(h['total_quantity'])
                for h in holdings.get('results', [])}
    
    def get_buying_power(self):
        account = self.client.get_account()
        return float(account.get('buying_power', 0))
    
    # ... implement other methods
```

## Installation

```bash
cd trading_bot/py
pip install -r trading_bot/requirements.txt
```

## ⚠️ Security Warning

**NEVER commit these to version control:**
- `data/trades.db` - Contains your actual trading history and P&L
- `models/*.joblib` - Contains your trained models (proprietary)
- `*.csv` exports - May contain sensitive financial data

These are already in [.gitignore](../../.gitignore). Always verify before pushing:

```bash
git status  # Check what will be committed
```

If you accidentally commit sensitive data:
```bash
# Remove from Git history (careful!)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch data/trades.db" \
  --prune-empty --tag-name-filter cat -- --all
```

## Testing

The bot includes safety features:
- Paper trading mode (implement in your adapter)
- Daily loss limits
- Position size limits
- Risk management rules

Always test with small amounts first!

## License

MIT License - See LICENSE file for details
