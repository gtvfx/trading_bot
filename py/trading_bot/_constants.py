"""Package-level constants for trading bot.

These constants can be overridden using environment variables.

"""

import os
from pathlib import Path

# Package directory (py/trading_bot/..)
PACKAGE_DIR = Path(__file__).parent.parent

# Database path - can be overridden with TRADING_BOT_DB environment variable
# Defaults to absolute path: <package_dir>/data/trades.db
DB_PATH = os.getenv("TRADING_BOT_DB", str(PACKAGE_DIR / "data" / "trades.db"))

# Model storage path - can be overridden with TRADING_BOT_MODELS environment variable
# Defaults to absolute path: <package_dir>/models
MODEL_PATH = os.getenv("TRADING_BOT_MODELS", str(PACKAGE_DIR / "models"))

# Default configuration values
DEFAULT_POSITION_SIZE_PERCENT = 10.0
DEFAULT_MAX_POSITIONS = 5
DEFAULT_STOP_LOSS_PERCENT = 2.0
DEFAULT_TAKE_PROFIT_PERCENT = 5.0
DEFAULT_MAX_DAILY_LOSS_PERCENT = 5.0
