"""Data curation and analysis tools.

Tools for reviewing model performance, analyzing trades,
and curating training data.

"""

import sqlite3
import pandas as pd
from typing import Optional
from .trade_history import TradeHistory
from ._constants import DB_PATH


class DataCurator:
    """Tools for reviewing and cleaning training data."""
    
    def __init__(self, db_path: str = DB_PATH):
        """Initialize data curator.
        
        Args:
            db_path: Path to SQLite database
            
        """
        self.history = TradeHistory(db_path)
    
    def analyze_model_performance(self, symbol: Optional[str] = None):
        """Generate comprehensive model performance report.
        
        Args:
            symbol: Trading pair (optional, analyzes all if not specified)
            
        """
        conn = sqlite3.connect(self.history.db_path)
        
        print("\n" + "="*70)
        print("ML MODEL PERFORMANCE ANALYSIS")
        print("="*70)
        
        # Accuracy by symbol
        print("\n📊 Prediction Accuracy by Symbol")
        print("-" * 70)
        
        df = pd.read_sql_query("""
            SELECT 
                symbol,
                COUNT(*) as predictions,
                AVG(CASE WHEN correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
                AVG(confidence) as avg_confidence,
                AVG(CASE WHEN correct = 1 THEN confidence ELSE NULL END) as confidence_when_correct,
                AVG(CASE WHEN correct = 0 THEN confidence ELSE NULL END) as confidence_when_wrong
            FROM predictions
            WHERE actual_outcome IS NOT NULL
            """ + ("AND symbol = ?" if symbol else "") + """
            GROUP BY symbol
        """, conn, params=[symbol] if symbol else None)
        
        if not df.empty:
            for _, row in df.iterrows():
                print(f"\n{row['symbol']}:")
                print(f"  Total Predictions: {int(row['predictions'])}")
                print(f"  Accuracy: {row['accuracy']:.1%}")
                print(f"  Avg Confidence: {row['avg_confidence']:.1%}")
                print(f"  Confidence (Correct): {row['confidence_when_correct']:.1%}")
                print(f"  Confidence (Wrong): {row['confidence_when_wrong']:.1%}")
        else:
            print("  No prediction data available yet")
        
        # Performance over time
        print("\n📈 Accuracy Trend (Last 30 Days)")
        print("-" * 70)
        
        df = pd.read_sql_query("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as predictions,
                AVG(CASE WHEN correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy
            FROM predictions
            WHERE actual_outcome IS NOT NULL
            AND timestamp >= datetime('now', '-30 days')
            """ + ("AND symbol = ?" if symbol else "") + """
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 10
        """, conn, params=[symbol] if symbol else None)
        
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("  No recent prediction data")
        
        # Confidence calibration
        print("\n🎯 Confidence Calibration")
        print("-" * 70)
        
        df = pd.read_sql_query("""
            SELECT 
                CASE 
                    WHEN confidence < 0.6 THEN '<60%'
                    WHEN confidence < 0.7 THEN '60-70%'
                    WHEN confidence < 0.8 THEN '70-80%'
                    WHEN confidence < 0.9 THEN '80-90%'
                    ELSE '90%+'
                END as confidence_bucket,
                COUNT(*) as predictions,
                AVG(CASE WHEN correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy
            FROM predictions
            WHERE actual_outcome IS NOT NULL
            """ + ("AND symbol = ?" if symbol else "") + """
            GROUP BY confidence_bucket
            ORDER BY confidence_bucket
        """, conn, params=[symbol] if symbol else None)
        
        if not df.empty:
            print(df.to_string(index=False))
            print("\nℹ Well-calibrated model: accuracy should match confidence bucket")
        else:
            print("  No prediction data")
        
        conn.close()
        print("="*70)
    
    def analyze_trade_performance(self, symbol: Optional[str] = None, days: int = 30):
        """Analyze trading performance.
        
        Args:
            symbol: Trading pair (optional)
            days: Number of days to analyze
            
        """
        print("\n" + "="*70)
        print("TRADING PERFORMANCE ANALYSIS")
        print("="*70)
        
        stats = self.history.get_trade_stats(symbol=symbol, days=days)
        
        if stats:
            print(f"\n📊 Summary (Last {days} Days)")
            print("-" * 70)
            print(f"Total Trades: {stats['total_trades']}")
            print(f"Winning Trades: {stats['winning_trades']}")
            print(f"Losing Trades: {stats['losing_trades']}")
            print(f"Win Rate: {stats['win_rate']:.1%}")
            print(f"\nTotal P&L: ${stats['total_pnl']:+,.2f}")
            print(f"Average P&L: ${stats['avg_pnl']:+,.2f}")
            print(f"Average P&L %: {stats['avg_pnl_percent']:+.2f}%")
            print(f"\nMax Profit: ${stats['max_profit']:,.2f}")
            print(f"Max Loss: ${stats['max_loss']:,.2f}")
            print(f"\nStop Losses Hit: {stats['stop_losses']}")
            print(f"Take Profits Hit: {stats['take_profits']}")
        else:
            print("\n  No trade data available yet")
        
        # Trade distribution
        conn = sqlite3.connect(self.history.db_path)
        
        print("\n💰 P&L Distribution")
        print("-" * 70)
        
        df = pd.read_sql_query("""
            SELECT 
                CASE 
                    WHEN pnl_percent < -5 THEN '<-5%'
                    WHEN pnl_percent < -2 THEN '-5% to -2%'
                    WHEN pnl_percent < 0 THEN '-2% to 0%'
                    WHEN pnl_percent < 2 THEN '0% to 2%'
                    WHEN pnl_percent < 5 THEN '2% to 5%'
                    ELSE '>5%'
                END as pnl_bucket,
                COUNT(*) as trades
            FROM trades
            WHERE exit_time IS NOT NULL
            AND entry_time >= datetime('now', '-' || ? || ' days')
            """ + ("AND symbol = ?" if symbol else "") + """
            GROUP BY pnl_bucket
            ORDER BY 
                CASE 
                    WHEN pnl_bucket = '<-5%' THEN 1
                    WHEN pnl_bucket = '-5% to -2%' THEN 2
                    WHEN pnl_bucket = '-2% to 0%' THEN 3
                    WHEN pnl_bucket = '0% to 2%' THEN 4
                    WHEN pnl_bucket = '2% to 5%' THEN 5
                    ELSE 6
                END
        """, conn, params=[days, symbol] if symbol else [days])
        
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("  No completed trades")
        
        conn.close()
        print("="*70)
    
    def export_for_analysis(self, output_path: str = "data/ml_data.csv"):
        """Export data for external analysis.
        
        Args:
            output_path: Path to save CSV file
            
        """
        df = self.history.get_training_data()
        
        if not df.empty:
            df.to_csv(output_path, index=False)
            print(f"✓ Exported {len(df)} rows to {output_path}")
        else:
            print("⚠ No data to export")
    
    def get_data_summary(self):
        """Get summary of available data.
        
        Returns:
            Dictionary with data counts
            
        """
        conn = sqlite3.connect(self.history.db_path)
        cursor = conn.cursor()
        
        print("\n" + "="*70)
        print("DATA SUMMARY")
        print("="*70)
        
        # Market data
        cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM market_data")
        count, min_date, max_date = cursor.fetchone()
        print(f"\n📊 Market Data Candles: {count:,}")
        if count > 0:
            print(f"  Date Range: {min_date} to {max_date}")
        
        # Symbols
        cursor.execute("SELECT symbol, COUNT(*) FROM market_data GROUP BY symbol")
        symbols = cursor.fetchall()
        if symbols:
            print(f"\n  Symbols:")
            for symbol, cnt in symbols:
                print(f"    {symbol}: {cnt:,} candles")
        
        # Predictions
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN actual_outcome IS NOT NULL THEN 1 ELSE 0 END) as validated
            FROM predictions
        """)
        total_pred, validated_pred = cursor.fetchone()
        print(f"\n🤖 ML Predictions: {total_pred:,}")
        print(f"  Validated: {validated_pred:,}")
        print(f"  Pending: {total_pred - validated_pred:,}")
        
        # Trades
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN exit_time IS NOT NULL THEN 1 ELSE 0 END) as closed
            FROM trades
        """)
        total_trades, closed_trades = cursor.fetchone()
        print(f"\n💼 Trades: {total_trades:,}")
        print(f"  Closed: {closed_trades:,}")
        print(f"  Open: {total_trades - closed_trades:,}")
        
        conn.close()
        print("="*70)
    
    def clean_old_data(self, days_to_keep: int = 180):
        """Remove old data to keep database manageable.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        """
        conn = sqlite3.connect(self.history.db_path)
        cursor = conn.cursor()
        
        # Delete old market data
        cursor.execute("""
            DELETE FROM market_data
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """, (days_to_keep,))
        market_deleted = cursor.rowcount
        
        # Delete old predictions
        cursor.execute("""
            DELETE FROM predictions
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """, (days_to_keep,))
        pred_deleted = cursor.rowcount
        
        # Keep all completed trades (they're small)
        
        conn.commit()
        conn.close()
        
        print(f"✓ Cleaned old data:")
        print(f"  Market data: {market_deleted:,} rows deleted")
        print(f"  Predictions: {pred_deleted:,} rows deleted")
        print(f"  Keeping data from last {days_to_keep} days")


def main():
    """CLI tool for data analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading bot data analysis tool")
    parser.add_argument("--db", default=DB_PATH, help="Database path")
    parser.add_argument("--symbol", help="Filter by symbol")
    parser.add_argument("--days", type=int, default=30, help="Days to analyze")
    parser.add_argument("--summary", action="store_true", help="Show data summary")
    parser.add_argument("--model", action="store_true", help="Analyze model performance")
    parser.add_argument("--trades", action="store_true", help="Analyze trade performance")
    parser.add_argument("--export", help="Export data to CSV")
    parser.add_argument("--clean", type=int, help="Clean data older than N days")
    
    args = parser.parse_args()
    
    curator = DataCurator(args.db)
    
    if args.summary or not any([args.model, args.trades, args.export, args.clean]):
        curator.get_data_summary()
    
    if args.model:
        curator.analyze_model_performance(symbol=args.symbol)
    
    if args.trades:
        curator.analyze_trade_performance(symbol=args.symbol, days=args.days)
    
    if args.export:
        curator.export_for_analysis(args.export)
    
    if args.clean:
        curator.clean_old_data(days_to_keep=args.clean)


if __name__ == "__main__":
    main()
