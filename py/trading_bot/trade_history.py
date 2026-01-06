"""Trade history and market data persistence.

Stores market data, predictions, and trades in SQLite database
for continuous learning and performance analysis.

"""

import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from ._constants import DB_PATH

# Register datetime adapters for Python 3.12+ compatibility
# Converts datetime objects to ISO format strings for storage
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())

# Converts ISO format strings back to datetime objects when reading
sqlite3.register_converter("DATETIME", lambda b: datetime.fromisoformat(b.decode()))


class TradeHistory:
    """Store and retrieve historical trade data for ML training."""
    
    def __init__(self, db_path: str = DB_PATH):
        """Initialize trade history database.
        
        Args:
            db_path: Path to SQLite database file
            
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_hist REAL,
                ema_short REAL,
                ema_long REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                bb_width REAL,
                stoch_k REAL,
                stoch_d REAL,
                adx REAL,
                obv REAL,
                vwap REAL,
                UNIQUE(symbol, timestamp)
            )
        """)
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                prediction INTEGER,
                confidence REAL,
                actual_outcome INTEGER,
                price_at_prediction REAL,
                price_after_1h REAL,
                correct BOOLEAN
            )
        """)
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_time DATETIME NOT NULL,
                exit_time DATETIME,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                pnl REAL,
                pnl_percent REAL,
                ml_confidence REAL,
                technical_signals TEXT,
                stop_loss_triggered BOOLEAN,
                take_profit_triggered BOOLEAN
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"✓ Database initialized: {self.db_path}")
    
    def store_market_data(self, symbol: str, df: pd.DataFrame):
        """Store market data with indicators.
        
        Args:
            symbol: Trading pair
            df: DataFrame with OHLCV and indicator data
            
        """
        if df.empty:
            return
        
        conn = sqlite3.connect(self.db_path)
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        
        # Only keep relevant columns
        cols_to_keep = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
                       'rsi', 'macd', 'macd_signal', 'macd_hist', 'ema_short', 'ema_long',
                       'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                       'stoch_k', 'stoch_d', 'adx', 'obv', 'vwap']
        
        df_copy = df_copy[[col for col in cols_to_keep if col in df_copy.columns]]
        
        try:
            df_copy.to_sql('market_data', conn, if_exists='append', index=False)
        except sqlite3.IntegrityError:
            # Data already exists, skip
            pass
        finally:
            conn.close()
    
    def store_prediction(
        self,
        symbol: str,
        prediction: int,
        confidence: float,
        current_price: float
    ):
        """Store ML prediction for later validation.
        
        Args:
            symbol: Trading pair
            prediction: 1 (buy) or 0 (sell)
            confidence: Prediction confidence (0-1)
            current_price: Current market price
            
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions 
            (symbol, timestamp, prediction, confidence, price_at_prediction)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, datetime.now(), prediction, confidence, current_price))
        
        conn.commit()
        conn.close()
    
    def store_trade(
        self,
        symbol: str,
        side: str,
        entry_time: datetime,
        entry_price: float,
        quantity: float,
        ml_confidence: float,
        technical_signals: str,
        exit_time: Optional[datetime] = None,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        pnl_percent: Optional[float] = None,
        stop_loss_triggered: bool = False,
        take_profit_triggered: bool = False
    ):
        """Store trade information.
        
        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            entry_time: Entry timestamp
            entry_price: Entry price
            quantity: Position size
            ml_confidence: ML prediction confidence
            technical_signals: Comma-separated list of signals
            exit_time: Exit timestamp (optional)
            exit_price: Exit price (optional)
            pnl: Profit/loss in USD (optional)
            pnl_percent: P&L percentage (optional)
            stop_loss_triggered: Whether stop loss was hit
            take_profit_triggered: Whether take profit was hit
            
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades 
            (symbol, entry_time, exit_time, side, entry_price, exit_price,
             quantity, pnl, pnl_percent, ml_confidence, technical_signals,
             stop_loss_triggered, take_profit_triggered)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, entry_time, exit_time, side, entry_price, exit_price,
              quantity, pnl, pnl_percent, ml_confidence, technical_signals,
              stop_loss_triggered, take_profit_triggered))
        
        conn.commit()
        conn.close()
    
    def update_trade_exit(
        self,
        symbol: str,
        entry_time: datetime,
        exit_time: datetime,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        stop_loss_triggered: bool = False,
        take_profit_triggered: bool = False
    ):
        """Update trade with exit information.
        
        Args:
            symbol: Trading pair
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            exit_price: Exit price
            pnl: Profit/loss in USD
            pnl_percent: P&L percentage
            stop_loss_triggered: Whether stop loss was hit
            take_profit_triggered: Whether take profit was hit
            
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE trades
            SET exit_time = ?, exit_price = ?, pnl = ?, pnl_percent = ?,
                stop_loss_triggered = ?, take_profit_triggered = ?
            WHERE symbol = ? AND entry_time = ? AND exit_time IS NULL
        """, (exit_time, exit_price, pnl, pnl_percent,
              stop_loss_triggered, take_profit_triggered, symbol, entry_time))
        
        conn.commit()
        conn.close()
    
    def update_prediction_outcome(self):
        """Update predictions with actual outcomes after 1 hour."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get unvalidated predictions older than 1 hour
        cursor.execute("""
            SELECT id, symbol, price_at_prediction, prediction, timestamp
            FROM predictions
            WHERE actual_outcome IS NULL
            AND timestamp < datetime('now', '-1 hour')
        """)
        
        updated = 0
        for pred_id, symbol, entry_price, prediction, timestamp in cursor.fetchall():
            # Get actual price 1 hour later
            cursor.execute("""
                SELECT close FROM market_data
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp ASC LIMIT 1
            """, (symbol, timestamp))
            
            result = cursor.fetchone()
            if result:
                actual_price = result[0]
                price_change = (actual_price - entry_price) / entry_price
                actual_outcome = 1 if price_change > 0 else 0
                correct = (prediction == actual_outcome)
                
                cursor.execute("""
                    UPDATE predictions
                    SET actual_outcome = ?, price_after_1h = ?, correct = ?
                    WHERE id = ?
                """, (actual_outcome, actual_price, correct, pred_id))
                
                updated += 1
        
        conn.commit()
        conn.close()
        
        if updated > 0:
            print(f"✓ Updated {updated} prediction outcomes")
    
    def get_training_data(
        self,
        symbol: Optional[str] = None,
        min_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Retrieve historical data for training.
        
        Args:
            symbol: Trading pair (optional)
            min_date: Minimum date (optional)
            limit: Maximum rows to return (optional)
            
        Returns:
            DataFrame with historical market data
            
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM market_data WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if min_date:
            query += " AND timestamp >= ?"
            params.append(min_date)
        
        query += " ORDER BY timestamp ASC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    
    def get_model_accuracy(
        self,
        symbol: Optional[str] = None,
        days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Calculate model accuracy over recent period.
        
        Args:
            symbol: Trading pair (optional)
            days: Number of days to look back
            
        Returns:
            Dictionary with accuracy metrics or None
            
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
                AVG(confidence) as avg_confidence
            FROM predictions
            WHERE actual_outcome IS NOT NULL
            AND timestamp >= datetime('now', '-' || ? || ' days')
        """
        params: list = [days]
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        result = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if result['total'].iloc[0] > 0:
            total = int(result['total'].iloc[0])
            correct = int(result['correct_predictions'].iloc[0])
            accuracy = correct / total
            return {
                'accuracy': accuracy,
                'total_predictions': total,
                'correct_predictions': correct,
                'avg_confidence': float(result['avg_confidence'].iloc[0])
            }
        return None
    
    def get_trade_stats(
        self,
        symbol: Optional[str] = None,
        days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Get trading statistics.
        
        Args:
            symbol: Trading pair (optional)
            days: Number of days to look back
            
        Returns:
            Dictionary with trade statistics or None
            
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                AVG(pnl) as avg_pnl,
                SUM(pnl) as total_pnl,
                AVG(pnl_percent) as avg_pnl_percent,
                MAX(pnl) as max_profit,
                MIN(pnl) as max_loss,
                SUM(CASE WHEN stop_loss_triggered = 1 THEN 1 ELSE 0 END) as stop_losses,
                SUM(CASE WHEN take_profit_triggered = 1 THEN 1 ELSE 0 END) as take_profits
            FROM trades
            WHERE exit_time IS NOT NULL
            AND entry_time >= datetime('now', '-' || ? || ' days')
        """
        params: list = [days]
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        result = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if result['total_trades'].iloc[0] > 0:
            total = int(result['total_trades'].iloc[0])
            wins = int(result['winning_trades'].iloc[0])
            return {
                'total_trades': total,
                'winning_trades': wins,
                'losing_trades': total - wins,
                'win_rate': wins / total,
                'avg_pnl': float(result['avg_pnl'].iloc[0]),
                'total_pnl': float(result['total_pnl'].iloc[0]),
                'avg_pnl_percent': float(result['avg_pnl_percent'].iloc[0]),
                'max_profit': float(result['max_profit'].iloc[0]),
                'max_loss': float(result['max_loss'].iloc[0]),
                'stop_losses': int(result['stop_losses'].iloc[0]),
                'take_profits': int(result['take_profits'].iloc[0])
            }
        return None
