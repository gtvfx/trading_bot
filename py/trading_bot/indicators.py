"""Technical indicators for trading analysis.

Implements proven technical indicators used for market analysis and signal generation.
All indicators follow standard industry calculations.

"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Calculate technical indicators for trading analysis."""
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI).
        
        RSI measures momentum and overbought/oversold conditions.
        
        Args:
            df: DataFrame with 'close' column
            period: RSI period (default: 14)
            
        Returns:
            Series with RSI values (0-100)

        """
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        MACD shows trend direction and momentum.
        
        Args:
            df: DataFrame with 'close' column
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)

        """
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA).
        
        Args:
            df: DataFrame with 'close' column
            period: EMA period
            
        Returns:
            Series with EMA values

        """
        return df['close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA).
        
        Args:
            df: DataFrame with 'close' column
            period: SMA period
            
        Returns:
            Series with SMA values

        """
        return df['close'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple:
        """Calculate Bollinger Bands.
        
        Bollinger Bands show volatility and potential reversal points.
        
        Args:
            df: DataFrame with 'close' column
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)

        """
        middle = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR).
        
        ATR measures market volatility.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ATR period
            
        Returns:
            Series with ATR values

        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> tuple:
        """Calculate Stochastic Oscillator.
        
        Stochastic measures momentum and overbought/oversold conditions.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (k_line, d_line)

        """
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume (OBV).
        
        OBV uses volume to predict price movements.
        
        Args:
            df: DataFrame with 'close' and 'volume' columns
            
        Returns:
            Series with OBV values

        """
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP).
        
        VWAP shows the average price weighted by volume.
        
        Args:
            df: DataFrame with 'high', 'low', 'close', 'volume' columns
            
        Returns:
            Series with VWAP values

        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX).
        
        ADX measures trend strength.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ADX period
            
        Returns:
            Series with ADX values

        """
        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Calculate ATR
        atr = TechnicalIndicators.calculate_atr(df, period)
        
        # Calculate +DI and -DI
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame, config) -> pd.DataFrame:
        """Add all configured indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            config: TradingConfig instance
            
        Returns:
            DataFrame with all indicator columns added

        """
        # RSI
        df['rsi'] = TechnicalIndicators.calculate_rsi(df, config.rsi_period)
        
        # MACD
        macd, signal, hist = TechnicalIndicators.calculate_macd(
            df, config.macd_fast, config.macd_slow, config.macd_signal
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # EMAs
        df['ema_short'] = TechnicalIndicators.calculate_ema(df, config.ema_short)
        df['ema_long'] = TechnicalIndicators.calculate_ema(df, config.ema_long)
        
        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(
            df, config.bb_period, config.bb_std_dev
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # ATR
        df['atr'] = TechnicalIndicators.calculate_atr(df, config.atr_period)
        
        # Stochastic
        k, d = TechnicalIndicators.calculate_stochastic(df)
        df['stoch_k'] = k
        df['stoch_d'] = d
        
        # Volume indicators (if enabled)
        if config.obv_enabled:
            df['obv'] = TechnicalIndicators.calculate_obv(df)
        
        if config.vwap_enabled:
            df['vwap'] = TechnicalIndicators.calculate_vwap(df)
        
        # ADX for trend strength
        df['adx'] = TechnicalIndicators.calculate_adx(df)
        
        return df
