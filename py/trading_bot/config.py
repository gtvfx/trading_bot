"""Trading bot configuration.

Centralized configuration for trading parameters, risk management,
and strategy settings.

"""

from dataclasses import dataclass, field
from typing import List, Optional
from ._constants import DB_PATH, MODEL_PATH


@dataclass
class TradingConfig:
    """Configuration for trading bot behavior and risk management."""
    
    # Trading pairs
    trading_pairs: List[str] = field(default_factory=lambda: ["BTC-USD"])
    
    # Position sizing
    position_size_percent: float = 10.0  # % of buying power per trade
    max_positions: int = 5  # Maximum concurrent positions
    
    # Risk management
    stop_loss_percent: float = 2.0  # % loss to trigger stop
    take_profit_percent: float = 5.0  # % gain to trigger take profit
    trailing_stop_percent: Optional[float] = None  # Optional trailing stop
    max_daily_loss_percent: float = 5.0  # Max daily loss before stopping
    
    # Technical indicators
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    ema_short: int = 9
    ema_long: int = 21
    
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    atr_period: int = 14
    
    # Volume indicators
    obv_enabled: bool = True
    vwap_enabled: bool = True
    
    # Machine Learning
    ml_enabled: bool = True
    ml_model_type: str = "random_forest"  # random_forest, gradient_boosting, lstm
    ml_lookback_periods: int = 50  # Historical periods for training
    ml_confidence_threshold: float = 0.65  # Min confidence to act on ML signals
    ml_retrain_interval: int = 100  # Retrain every N candles
    ml_training_days: int = 90  # Days of historical data for training
    
    # Persistence (can be overridden with environment variables)
    db_path: str = DB_PATH  # SQLite database path
    model_path: str = MODEL_PATH  # Directory for saved models
    
    # Strategy parameters
    min_signal_agreement: int = 3  # Minimum indicators that must agree
    lookback_window: int = 100  # Historical candles to analyze
    
    # Data collection
    candle_granularity: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d
    refresh_interval: int = 60  # Seconds between bot iterations
    
    # Backtesting
    backtest_start_balance: float = 10000.0
    backtest_commission_percent: float = 0.1  # Trading fee %
    
    # Paper Trading
    paper_trading: bool = False  # Simulate trades without executing real orders
    paper_trading_balance: float = 10000.0  # Starting balance for paper trading
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.position_size_percent <= 0 or self.position_size_percent > 100:
            raise ValueError("position_size_percent must be between 0 and 100")
        
        if self.stop_loss_percent < 0:
            raise ValueError("stop_loss_percent must be non-negative")
        
        if self.take_profit_percent < 0:
            raise ValueError("take_profit_percent must be non-negative")
        
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")
        
        if self.ml_confidence_threshold < 0 or self.ml_confidence_threshold > 1:
            raise ValueError("ml_confidence_threshold must be between 0 and 1")
