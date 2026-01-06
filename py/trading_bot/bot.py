"""Main trading bot implementation.

Coordinates data collection, signal generation, and order execution
across any exchange through the ExchangeClient interface.

"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd

from .exchange_client import ExchangeClient
from .config import TradingConfig
from .indicators import TechnicalIndicators
from .strategy import TradingStrategy, Signal
from .ml_predictor import MLPredictor
from .trade_history import TradeHistory


class Position:
    """Represents an open trading position."""
    
    def __init__(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        entry_time: datetime
    ):
        """Initialize position.
        
        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            quantity: Position size
            entry_price: Entry price
            entry_time: Entry timestamp

        """
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.order_id = None
    
    def get_pnl(self, current_price: float) -> float:
        """Calculate current P&L.
        
        Args:
            current_price: Current market price
            
        Returns:
            P&L in USD

        """
        if self.side == "buy":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity
    
    def get_pnl_percent(self, current_price: float) -> float:
        """Calculate P&L percentage.
        
        Args:
            current_price: Current market price
            
        Returns:
            P&L as percentage

        """
        if self.side == "buy":
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100


class TradingBot:
    """Exchange-agnostic algorithmic trading bot."""
    
    def __init__(
        self,
        exchange_client: ExchangeClient,
        config: TradingConfig
    ):
        """Initialize trading bot.
        
        Args:
            exchange_client: Exchange client implementing ExchangeClient interface
            config: Trading configuration

        """
        self.client = exchange_client
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.ml_predictor = None
        self.strategy = None
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        self.daily_pnl = 0.0
        self.daily_start_balance = 0.0
        self.iteration_count = 0
        self.trade_history = TradeHistory(config.db_path)
        
        # Initialize ML if enabled
        if config.ml_enabled:
            self.ml_predictor = MLPredictor(
                config.ml_model_type,
                model_path=config.model_path
            )
        
        # Initialize strategy
        self.strategy = TradingStrategy(config, self.ml_predictor)
    
    def initialize(self):
        """Initialize bot and collect initial data."""
        print("Initializing trading bot...")
        
        # Store starting balance
        self.daily_start_balance = self.client.get_buying_power()
        print(f"Starting balance: ${self.daily_start_balance:,.2f}")
        
        # Try to load existing ML models
        if self.config.ml_enabled and self.ml_predictor:
            print("Loading ML models...")
            for symbol in self.config.trading_pairs:
                if self.ml_predictor.load_model(symbol=symbol):
                    print(f"  ✓ {symbol} model loaded")
                else:
                    print(f"  ℹ {symbol} model not found, will train from scratch")
        
        # Collect initial market data
        print("Collecting initial market data...")
        for symbol in self.config.trading_pairs:
            self._update_market_data(symbol)
        
        # Train ML model if enabled and not already loaded
        if self.config.ml_enabled and self.ml_predictor:
            print("Training ML models...")
            for symbol in self.config.trading_pairs:
                # Try to load historical data from database first
                historical_df = self.trade_history.get_training_data(
                    symbol=symbol,
                    min_date=datetime.now() - timedelta(days=self.config.ml_training_days)
                )
                
                if not historical_df.empty and len(historical_df) >= self.config.ml_lookback_periods:
                    print(f"  Training {symbol} on {len(historical_df)} historical candles...")
                    self.ml_predictor.train(historical_df, symbol=symbol)
                else:
                    # Use current market data cache
                    df = self.market_data_cache.get(symbol)
                    if df is not None and len(df) >= self.config.ml_lookback_periods:
                        print(f"  Training {symbol} on {len(df)} recent candles...")
                        self.ml_predictor.train(df, symbol=symbol)
        
        print("✓ Bot initialized")
    
    def _update_market_data(self, symbol: str) -> pd.DataFrame:
        """Update market data for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Updated DataFrame with indicators

        """
        # Calculate time range
        end = datetime.now()
        
        # Get appropriate lookback based on granularity
        granularity_hours = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 1, "4h": 4, "1d": 24
        }
        
        hours = granularity_hours.get(self.config.candle_granularity, 1)
        start = end - timedelta(hours=hours * self.config.lookback_window)
        
        # Fetch candles
        df = self.client.get_historical_candles(
            symbol,
            start,
            end,
            self.config.candle_granularity
        )
        
        if df.empty:
            return df
        
        # Add technical indicators
        df = TechnicalIndicators.add_all_indicators(df, self.config)
        
        # Cache the data
        self.market_data_cache[symbol] = df
        
        # Store to database for future training
        self.trade_history.store_market_data(symbol, df)
        
        return df
    
    def scan_for_opportunities(self):
        """Scan all trading pairs for opportunities."""
        for symbol in self.config.trading_pairs:
            # Skip if already have position
            if symbol in self.positions:
                continue
            
            # Skip if max positions reached
            if len(self.positions) >= self.config.max_positions:
                break
            
            # Update market data
            df = self._update_market_data(symbol)
            if df.empty:
                continue
            
            # Generate signal
            signal, confidence = self.strategy.generate_signal(df)
            
            # Store ML prediction if enabled
            if self.config.ml_enabled and self.ml_predictor and self.ml_predictor.is_trained:
                try:
                    prediction, ml_confidence = self.ml_predictor.predict_proba(df)
                    current_price = df['close'].iloc[-1]
                    self.trade_history.store_prediction(
                        symbol=symbol,
                        prediction=prediction,
                        confidence=ml_confidence,
                        current_price=current_price
                    )
                except Exception as e:
                    print(f"Failed to store prediction for {symbol}: {e}")
            
            # Check if should enter
            if self.strategy.should_enter_position(df, signal, confidence):
                self._open_position(symbol, signal, confidence, df)
    
    def _open_position(
        self,
        symbol: str,
        signal: Signal,
        confidence: float,
        df: pd.DataFrame
    ):
        """Open a new trading position.
        
        Args:
            symbol: Trading pair
            signal: Trading signal
            confidence: Signal confidence
            df: Market data

        """
        try:
            # Calculate position size
            buying_power = self.client.get_buying_power()
            position_value = buying_power * (self.config.position_size_percent / 100)
            
            current_price = df['close'].iloc[-1]
            quantity = position_value / current_price
            
            # Determine side
            side = "buy" if signal == Signal.BUY else "sell"
            
            # Place market order
            print(f"\n📊 Opening {side.upper()} position: {symbol}")
            print(f"   Price: ${current_price:,.2f}")
            print(f"   Quantity: {quantity:.8f}")
            print(f"   Confidence: {confidence:.2%}")
            
            order = self.client.place_market_order(symbol, side, quantity)
            
            if order and order.get('status') in ['filled', 'open']:
                # Create position object
                position = Position(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=current_price,
                    entry_time=datetime.now()
                )
                position.order_id = order.get('id')
                
                self.positions[symbol] = position
                
                # Store trade in database
                ml_conf = confidence if self.config.ml_enabled else 0.0
                signals_str = f"rsi,macd,ema,bb,stoch,ml" if self.config.ml_enabled else "rsi,macd,ema,bb,stoch"
                
                self.trade_history.store_trade(
                    symbol=symbol,
                    side=side,
                    entry_time=position.entry_time,
                    entry_price=current_price,
                    quantity=quantity,
                    ml_confidence=ml_conf,
                    technical_signals=signals_str
                )
                
                print(f"✓ Position opened: {symbol}")
            else:
                print(f"❌ Failed to open position: {symbol}")
        
        except Exception as e:
            print(f"Error opening position for {symbol}: {e}")
    
    def evaluate_positions(self):
        """Evaluate all open positions for exit signals."""
        symbols_to_close = []
        
        for symbol, position in self.positions.items():
            # Update market data
            df = self._update_market_data(symbol)
            if df.empty:
                continue
            
            current_price = df['close'].iloc[-1]
            pnl = position.get_pnl(current_price)
            pnl_pct = position.get_pnl_percent(current_price)
            
            # Check if should exit
            should_exit, reason = self.strategy.should_exit_position(
                df, position.entry_price, position.side
            )
            
            if should_exit:
                print(f"\n📉 Closing position: {symbol}")
                print(f"   Reason: {reason}")
                print(f"   P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
                
                self._close_position(symbol, reason)
                symbols_to_close.append(symbol)
        
        # Remove closed positions
        for symbol in symbols_to_close:
            del self.positions[symbol]
    
    def _close_position(self, symbol: str, reason: str):
        """Close a trading position.
        
        Args:
            symbol: Trading pair
            reason: Reason for closing

        """
        try:
            position = self.positions[symbol]
            
            # Determine opposite side for closing
            close_side = "sell" if position.side == "buy" else "buy"
            
            # Place market order to close
            order = self.client.place_market_order(
                symbol,
                close_side,
                position.quantity
            )
            
            if order and order.get('status') in ['filled', 'open']:
                # Update daily P&L
                current_price = self.client.get_current_price(symbol)
                if current_price:
                    pnl = position.get_pnl(current_price)
                    pnl_pct = position.get_pnl_percent(current_price)
                    self.daily_pnl += pnl
                    
                    # Update trade in database with exit info
                    stop_loss = "stop_loss" in reason.lower() or "stop" in reason.lower()
                    take_profit = "take_profit" in reason.lower() or "profit" in reason.lower()
                    
                    self.trade_history.update_trade_exit(
                        symbol=symbol,
                        entry_time=position.entry_time,
                        exit_time=datetime.now(),
                        exit_price=current_price,
                        pnl=pnl,
                        pnl_percent=pnl_pct,
                        stop_loss_triggered=stop_loss,
                        take_profit_triggered=take_profit
                    )
                
                print(f"✓ Position closed: {symbol}")
            else:
                print(f"❌ Failed to close position: {symbol}")
        
        except Exception as e:
            print(f"Error closing position for {symbol}: {e}")
    
    def check_risk_limits(self) -> bool:
        """Check if daily risk limits are breached.
        
        Returns:
            True if should continue trading, False if should stop

        """
        if self.daily_start_balance == 0:
            return True
        
        daily_loss_pct = (self.daily_pnl / self.daily_start_balance) * 100
        
        if daily_loss_pct <= -self.config.max_daily_loss_percent:
            print(f"\n⚠️  Daily loss limit reached: {daily_loss_pct:.2f}%")
            print("Stopping trading for today")
            return False
        
        return True
    
    def retrain_ml_models(self):
        """Retrain ML models periodically."""
        if not self.config.ml_enabled or not self.ml_predictor:
            return
        
        if self.iteration_count % self.config.ml_retrain_interval == 0:
            print("\n🔄 Retraining ML models...")
            
            for symbol in self.config.trading_pairs:
                # Use historical data from database for better training
                historical_df = self.trade_history.get_training_data(
                    symbol=symbol,
                    min_date=datetime.now() - timedelta(days=self.config.ml_training_days),
                    limit=self.config.ml_lookback_periods * 2
                )
                
                if not historical_df.empty and len(historical_df) >= self.config.ml_lookback_periods:
                    print(f"  Training {symbol} on {len(historical_df)} candles from database...")
                    self.ml_predictor.train(historical_df, symbol=symbol)
                else:
                    # Fall back to cache
                    df = self.market_data_cache.get(symbol)
                    if df is not None and len(df) >= self.config.ml_lookback_periods:
                        print(f"  Training {symbol} on {len(df)} cached candles...")
                        self.ml_predictor.train(df, symbol=symbol)
    
    def print_status(self):
        """Print current bot status."""
        print("\n" + "="*70)
        print(f"Trading Bot Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Account info
        buying_power = self.client.get_buying_power()
        print(f"\nBuying Power: ${buying_power:,.2f}")
        print(f"Daily P&L: ${self.daily_pnl:+,.2f}")
        
        # ML Performance (if enabled)
        if self.config.ml_enabled and self.iteration_count % 10 == 0:
            for symbol in self.config.trading_pairs:
                accuracy = self.trade_history.get_model_accuracy(symbol=symbol, days=7)
                if accuracy:
                    print(f"\n{symbol} ML Accuracy (7d): {accuracy['accuracy']:.1%} "
                          f"({accuracy['total_predictions']} predictions)")
        
        # Positions
        print(f"\nOpen Positions: {len(self.positions)}/{self.config.max_positions}")
        
        if self.positions:
            for symbol, pos in self.positions.items():
                current_price = self.client.get_current_price(symbol)
                if current_price:
                    pnl = pos.get_pnl(current_price)
                    pnl_pct = pos.get_pnl_percent(current_price)
                    print(f"  {symbol}: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        
        print("="*70)
    
    def run_iteration(self):
        """Run one iteration of the trading bot."""
        self.iteration_count += 1
        
        # Print status
        self.print_status()
        
        # Update prediction outcomes periodically
        if self.iteration_count % 10 == 0:
            self.trade_history.update_prediction_outcome()
        
        # Check risk limits
        if not self.check_risk_limits():
            return False
        
        # Evaluate existing positions
        self.evaluate_positions()
        
        # Scan for new opportunities
        self.scan_for_opportunities()
        
        # Retrain ML models if needed
        self.retrain_ml_models()
        
        return True
    
    def run(self, iterations: Optional[int] = None):
        """Run the trading bot.
        
        Args:
            iterations: Number of iterations to run (None for infinite)

        """
        self.initialize()
        
        iteration = 0
        while True:
            if iterations and iteration >= iterations:
                break
            
            # Run iteration
            should_continue = self.run_iteration()
            
            if not should_continue:
                print("\n🛑 Bot stopped")
                break
            
            # Wait before next iteration
            print(f"\n⏳ Waiting {self.config.refresh_interval} seconds...")
            time.sleep(self.config.refresh_interval)
            
            iteration += 1
        
        # Close any remaining positions
        if self.positions:
            print("\n🔒 Closing remaining positions...")
            for symbol in list(self.positions.keys()):
                self._close_position(symbol, "bot_stopped")
