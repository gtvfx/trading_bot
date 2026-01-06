"""Exchange-agnostic algorithmic trading bot framework.

This package provides a flexible trading bot that works with multiple
cryptocurrency exchanges through adapter interfaces.

Architecture:
    - Abstract exchange interface for API-agnostic trading
    - Technical indicators and ML-based signal generation
    - Position and risk management
    - Backtesting capabilities

"""

from .bot import TradingBot
from .config import TradingConfig
from .indicators import TechnicalIndicators
from .ml_predictor import MLPredictor
from .strategy import TradingStrategy, Signal
from .trade_history import TradeHistory
from .data_curator import DataCurator
from ._constants import DB_PATH, MODEL_PATH

# Version is automatically managed by setuptools-scm from git tags
try:
    from ._version import version as __version__
except ImportError:
    # Fallback if not installed via setuptools
    __version__ = "0.0.0.dev0+unknown"

__all__ = [
    'TradingBot',
    'TradingConfig',
    'TechnicalIndicators',
    'MLPredictor',
    'TradingStrategy',
    'Signal',
    'TradeHistory',
    'DataCurator',
    'DB_PATH',
    'MODEL_PATH',
]
