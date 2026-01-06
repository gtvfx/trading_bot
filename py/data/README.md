# Data Directory - PRIVATE

⚠️ **WARNING: This directory contains sensitive financial data!**

## What's Stored Here

- `trades.db` - SQLite database with all your trading data:
  - Complete trade history with P&L
  - ML predictions and their outcomes  
  - Market data with technical indicators
  - Position sizes and entry/exit prices

## Security

**Never commit this data to version control!**

This directory is ignored by `.gitignore` because it contains:
- Your actual trading performance (P&L)
- Your position sizes (reveals your capital)
- Your strategy's effectiveness metrics
- Personally identifiable financial information

## Backup Recommendation

While this data shouldn't be in your repo, you should back it up:

```bash
# Create backup
cp trades.db trades_backup_$(date +%Y%m%d).db

# Or use SQLite backup
sqlite3 trades.db ".backup ../backups/trades_backup.db"
```

## What IS Safe to Commit

- Code files (`.py`)
- Configuration templates (without API keys)
- Documentation
- Empty directory structure

## What is NOT Safe to Commit  

- `*.db` files
- `*.csv` exports
- Database backups
- Any file containing actual trading data
