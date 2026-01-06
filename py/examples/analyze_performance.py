"""Example: Analyzing trading bot performance.

This example shows how to use the DataCurator to analyze
your trading bot's historical performance.

"""

from trading_bot import DataCurator

def main():
    """Analyze trading bot data."""
    
    # Initialize curator
    curator = DataCurator(db_path="data/trades.db")
    
    # 1. Get overview of available data
    print("\n" + "="*70)
    print("STEP 1: Data Summary")
    print("="*70)
    curator.get_data_summary()
    
    # 2. Analyze ML model performance
    print("\n" + "="*70)
    print("STEP 2: ML Model Performance")
    print("="*70)
    curator.analyze_model_performance()
    
    # You can also analyze specific symbols
    # curator.analyze_model_performance(symbol="BTC-USD")
    
    # 3. Analyze trading performance
    print("\n" + "="*70)
    print("STEP 3: Trading Performance")
    print("="*70)
    curator.analyze_trade_performance(days=30)
    
    # 4. Export data for further analysis
    print("\n" + "="*70)
    print("STEP 4: Export Data")
    print("="*70)
    curator.export_for_analysis("exported_data.csv")
    
    print("\n✓ Analysis complete!")
    print("\nNext steps:")
    print("  1. Review accuracy trends - are predictions improving?")
    print("  2. Check confidence calibration - do high-confidence predictions perform better?")
    print("  3. Analyze trade distribution - are stop-losses/take-profits working?")
    print("  4. Use exported CSV for deeper analysis in Excel/Python")


if __name__ == "__main__":
    main()
