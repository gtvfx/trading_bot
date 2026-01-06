# Models Directory - PRIVATE

⚠️ **WARNING: These models are trained on your proprietary trading data!**

## What's Stored Here

Trained machine learning models saved as `.joblib` files:
- `random_forest_BTC_USD.joblib`
- `gradient_boosting_ETH_USD.joblib`
- etc.

## Why This is Private

These models contain:
- **Your training data patterns** - learned from your specific market data
- **Your competitive edge** - how your strategy identifies opportunities
- **Feature importance** - which indicators your strategy values most

If shared publicly, others could:
- Reverse-engineer your strategy
- Understand your decision-making process
- Potentially front-run your trades

## Security

**Never commit models to version control!**

This directory is ignored by `.gitignore`.

## Backup Recommendation

While models shouldn't be in your repo, you should back them up:

```bash
# Backup all models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Or copy to secure location
cp -r models/ ../backups/models_$(date +%Y%m%d)/
```

## Re-generating Models

If you lose your models, don't worry! As long as you have your `trades.db`:

```python
from trading_bot import TradingBot, TradingConfig

# Models will retrain from historical data in database
bot = TradingBot(exchange, config)
bot.initialize()  # Retrains if models not found
```

## What IS Safe to Commit

- Empty directory structure
- README documentation
- Model architecture code (in `.py` files)

## What is NOT Safe to Commit

- `*.joblib` files (trained models)
- `*.pkl` files (pickled models)
- `*.h5` files (Keras/TensorFlow models)
- Any serialized model weights
