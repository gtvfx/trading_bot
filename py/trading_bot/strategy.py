"""Trading strategy and signal generation.

Combines technical indicators and ML predictions to generate trading signals.
Uses majority voting and confidence thresholds for robust decision making.

"""

import pandas as pd
from enum import Enum
from typing import Tuple, Optional
from .ml_predictor import MLPredictor


class Signal(Enum):
    """Trading signal enum."""
    
    BUY = 1
    SELL = -1
    HOLD = 0


class TradingStrategy:
    """Generate trading signals using indicators and ML."""
    
    def __init__(self, config, ml_predictor: Optional[MLPredictor] = None):
        """Initialize trading strategy.
        
        Args:
            config: TradingConfig instance
            ml_predictor: Optional ML predictor instance

        """
        self.config = config
        self.ml_predictor = ml_predictor
    
    def analyze_rsi(self, df: pd.DataFrame) -> Signal:
        """Analyze RSI for signals.
        
        Args:
            df: DataFrame with RSI indicator
            
        Returns:
            Trading signal based on RSI

        """
        if 'rsi' not in df.columns or df.empty:
            return Signal.HOLD
        
        current_rsi = df['rsi'].iloc[-1]
        
        if current_rsi < self.config.rsi_oversold:
            return Signal.BUY  # Oversold - potential buy
        elif current_rsi > self.config.rsi_overbought:
            return Signal.SELL  # Overbought - potential sell
        
        return Signal.HOLD
    
    def analyze_macd(self, df: pd.DataFrame) -> Signal:
        """Analyze MACD for signals.
        
        Args:
            df: DataFrame with MACD indicators
            
        Returns:
            Trading signal based on MACD

        """
        if 'macd' not in df.columns or len(df) < 2:
            return Signal.HOLD
        
        current_macd = df['macd'].iloc[-1]
        current_signal = df['macd_signal'].iloc[-1]
        prev_macd = df['macd'].iloc[-2]
        prev_signal = df['macd_signal'].iloc[-2]
        
        # Bullish crossover
        if prev_macd <= prev_signal and current_macd > current_signal:
            return Signal.BUY
        
        # Bearish crossover
        if prev_macd >= prev_signal and current_macd < current_signal:
            return Signal.SELL
        
        return Signal.HOLD
    
    def analyze_ema(self, df: pd.DataFrame) -> Signal:
        """Analyze EMA crossover for signals.
        
        Args:
            df: DataFrame with EMA indicators
            
        Returns:
            Trading signal based on EMA crossover

        """
        if 'ema_short' not in df.columns or len(df) < 2:
            return Signal.HOLD
        
        current_short = df['ema_short'].iloc[-1]
        current_long = df['ema_long'].iloc[-1]
        prev_short = df['ema_short'].iloc[-2]
        prev_long = df['ema_long'].iloc[-2]
        
        # Bullish crossover
        if prev_short <= prev_long and current_short > current_long:
            return Signal.BUY
        
        # Bearish crossover
        if prev_short >= prev_long and current_short < current_long:
            return Signal.SELL
        
        return Signal.HOLD
    
    def analyze_bollinger_bands(self, df: pd.DataFrame) -> Signal:
        """Analyze Bollinger Bands for signals.
        
        Args:
            df: DataFrame with Bollinger Band indicators
            
        Returns:
            Trading signal based on Bollinger Bands

        """
        if 'bb_upper' not in df.columns or df.empty:
            return Signal.HOLD
        
        current_price = df['close'].iloc[-1]
        upper_band = df['bb_upper'].iloc[-1]
        lower_band = df['bb_lower'].iloc[-1]
        
        # Price near lower band - potential buy
        if current_price <= lower_band * 1.01:
            return Signal.BUY
        
        # Price near upper band - potential sell
        if current_price >= upper_band * 0.99:
            return Signal.SELL
        
        return Signal.HOLD
    
    def analyze_stochastic(self, df: pd.DataFrame) -> Signal:
        """Analyze Stochastic Oscillator for signals.
        
        Args:
            df: DataFrame with Stochastic indicators
            
        Returns:
            Trading signal based on Stochastic

        """
        if 'stoch_k' not in df.columns or len(df) < 2:
            return Signal.HOLD
        
        current_k = df['stoch_k'].iloc[-1]
        current_d = df['stoch_d'].iloc[-1]
        prev_k = df['stoch_k'].iloc[-2]
        prev_d = df['stoch_d'].iloc[-2]
        
        # Bullish signal: oversold and K crosses above D
        if current_k < 20 and prev_k <= prev_d and current_k > current_d:
            return Signal.BUY
        
        # Bearish signal: overbought and K crosses below D
        if current_k > 80 and prev_k >= prev_d and current_k < current_d:
            return Signal.SELL
        
        return Signal.HOLD
    
    def analyze_trend_strength(self, df: pd.DataFrame) -> float:
        """Analyze trend strength using ADX.
        
        Args:
            df: DataFrame with ADX indicator
            
        Returns:
            Trend strength score (0-1)

        """
        if 'adx' not in df.columns or df.empty:
            return 0.5
        
        adx = df['adx'].iloc[-1]
        
        # ADX > 25 indicates strong trend
        # ADX < 20 indicates weak/no trend
        if adx > 25:
            return 1.0
        elif adx > 20:
            return 0.75
        else:
            return 0.5
    
    def get_ml_signal(self, df: pd.DataFrame) -> Tuple[Signal, float]:
        """Get ML prediction signal with confidence.
        
        Args:
            df: DataFrame with market data and indicators
            
        Returns:
            Tuple of (signal, confidence)

        """
        if not self.ml_predictor or not self.ml_predictor.is_trained:
            return Signal.HOLD, 0.0
        
        try:
            prediction, confidence = self.ml_predictor.predict_proba(df)
            
            # Convert prediction to signal
            if prediction == 1:
                return Signal.BUY, confidence
            else:
                return Signal.SELL, confidence
        
        except Exception as e:
            print(f"ML prediction error: {e}")
            return Signal.HOLD, 0.0
    
    def generate_signal(self, df: pd.DataFrame) -> Tuple[Signal, float]:
        """Generate trading signal using multiple indicators.
        
        Combines multiple indicators with majority voting for robust signals.
        
        Args:
            df: DataFrame with market data and indicators
            
        Returns:
            Tuple of (signal, confidence)

        """
        if df.empty or len(df) < self.config.lookback_window:
            return Signal.HOLD, 0.0
        
        signals = []
        
        # Collect signals from technical indicators
        signals.append(self.analyze_rsi(df))
        signals.append(self.analyze_macd(df))
        signals.append(self.analyze_ema(df))
        signals.append(self.analyze_bollinger_bands(df))
        signals.append(self.analyze_stochastic(df))
        
        # Count votes
        buy_votes = sum(1 for s in signals if s == Signal.BUY)
        sell_votes = sum(1 for s in signals if s == Signal.SELL)
        
        # Get ML signal if enabled
        ml_signal = Signal.HOLD
        ml_confidence = 0.0
        
        if self.config.ml_enabled and self.ml_predictor:
            ml_signal, ml_confidence = self.get_ml_signal(df)
            
            # ML signal counts as 2 votes if confidence is high
            if ml_confidence >= self.config.ml_confidence_threshold:
                if ml_signal == Signal.BUY:
                    buy_votes += 2
                elif ml_signal == Signal.SELL:
                    sell_votes += 2
        
        # Get trend strength
        trend_strength = self.analyze_trend_strength(df)
        
        # Determine final signal with majority voting
        total_votes = buy_votes + sell_votes
        
        if total_votes == 0:
            return Signal.HOLD, 0.0
        
        # Need minimum agreement threshold
        if buy_votes >= self.config.min_signal_agreement:
            confidence = (buy_votes / (len(signals) + 2)) * trend_strength
            return Signal.BUY, confidence
        
        elif sell_votes >= self.config.min_signal_agreement:
            confidence = (sell_votes / (len(signals) + 2)) * trend_strength
            return Signal.SELL, confidence
        
        return Signal.HOLD, 0.0
    
    def should_enter_position(self, df: pd.DataFrame, signal: Signal, confidence: float) -> bool:
        """Determine if should enter a new position.
        
        Args:
            df: DataFrame with market data
            signal: Trading signal
            confidence: Signal confidence score
            
        Returns:
            True if should enter position

        """
        if signal == Signal.HOLD:
            return False
        
        # Check minimum confidence
        if confidence < 0.6:
            return False
        
        # Check trend strength
        trend_strength = self.analyze_trend_strength(df)
        if trend_strength < 0.5:
            return False
        
        return True
    
    def should_exit_position(
        self,
        df: pd.DataFrame,
        entry_price: float,
        position_side: str
    ) -> Tuple[bool, str]:
        """Determine if should exit current position.
        
        Args:
            df: DataFrame with market data
            entry_price: Entry price of position
            position_side: "buy" or "sell"
            
        Returns:
            Tuple of (should_exit, reason)

        """
        current_price = df['close'].iloc[-1]
        
        if position_side == "buy":
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_percent = ((entry_price - current_price) / entry_price) * 100
        
        # Stop loss
        if pnl_percent <= -self.config.stop_loss_percent:
            return True, "stop_loss"
        
        # Take profit
        if pnl_percent >= self.config.take_profit_percent:
            return True, "take_profit"
        
        # Signal reversal
        signal, confidence = self.generate_signal(df)
        
        if position_side == "buy" and signal == Signal.SELL and confidence > 0.7:
            return True, "signal_reversal"
        
        if position_side == "sell" and signal == Signal.BUY and confidence > 0.7:
            return True, "signal_reversal"
        
        return False, ""
