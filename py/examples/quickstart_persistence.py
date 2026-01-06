"""Quick start example for trading bot with persistence.

This example demonstrates the complete workflow with persistent learning.

"""

from datetime import datetime, timedelta
from trading_bot import TradingBot, TradingConfig, DataCurator


def setup_bot_with_persistence(exchange_adapter):
    """Initialize bot with persistence enabled.
    
    Args:
        exchange_adapter: Your exchange adapter instance
        
    Returns:
        Configured TradingBot instance
        
    """
    config = TradingConfig(
        # Trading settings
        trading_pairs=["BTC-USD", "ETH-USD"],
        position_size_percent=5.0,  # Start small!
        max_positions=2,
        
        # Risk management
        stop_loss_percent=2.0,
        take_profit_percent=5.0,
        max_daily_loss_percent=5.0,
        
        # Machine learning
        ml_enabled=True,
        ml_model_type="gradient_boosting",
        ml_confidence_threshold=0.70,  # Higher threshold = more conservative
        ml_retrain_interval=100,
        ml_training_days=90,
        
        # Persistence (NEW!)
        db_path="data/trades.db",
        model_path="models",
        
        # Strategy
        min_signal_agreement=3,
        lookback_window=100,
        candle_granularity="1h",
        refresh_interval=300  # 5 minutes
    )
    
    # Create bot - will automatically:
    # 1. Load existing models if available
    # 2. Load historical data from database
    # 3. Train on accumulated data
    bot = TradingBot(exchange_adapter, config)
    
    return bot


def run_with_monitoring(bot, iterations=None):
    """Run bot with periodic performance monitoring.
    
    Args:
        bot: TradingBot instance
        iterations: Number of iterations (None = infinite)
        
    """
    print("\n" + "="*70)
    print("STARTING TRADING BOT WITH PERSISTENCE")
    print("="*70)
    print("\n✓ Models will be saved automatically")
    print("✓ All data will be stored in database")
    print("✓ Predictions will be validated over time")
    print("\nMonitor performance with:")
    print("  python -m trading_bot.data_curator --summary")
    print("="*70 + "\n")
    
    try:
        # Run bot
        bot.run(iterations=iterations)
        
    finally:
        # Show final performance
        print("\n" + "="*70)
        print("BOT STOPPED - FINAL PERFORMANCE")
        print("="*70)
        
        curator = DataCurator(bot.config.db_path)
        
        # Show data collected
        curator.get_data_summary()
        
        # Show ML performance if data available
        accuracy = bot.trade_history.get_model_accuracy(days=7)
        if accuracy:
            print(f"\n7-day ML Accuracy: {accuracy['accuracy']:.1%}")
            print(f"Total Predictions: {accuracy['total_predictions']}")
        
        # Show trade performance
        stats = bot.trade_history.get_trade_stats(days=30)
        if stats:
            print(f"\n30-day Trade Performance:")
            print(f"  Win Rate: {stats['win_rate']:.1%}")
            print(f"  Total P&L: ${stats['total_pnl']:+,.2f}")
            print(f"  Avg P&L: ${stats['avg_pnl']:+,.2f}")


def analyze_performance():
    """Run performance analysis."""
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70 + "\n")
    
    curator = DataCurator("data/trades.db")
    
    # Get overview
    curator.get_data_summary()
    
    # Analyze ML performance
    curator.analyze_model_performance()
    
    # Analyze trades
    curator.analyze_trade_performance(days=30)
    
    print("\n✓ Analysis complete!")


def main():
    """Main example."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        # Just run analysis
        analyze_performance()
        return
    
    # Normal bot operation
    print("="*70)
    print("TRADING BOT WITH PERSISTENT LEARNING")
    print("="*70)
    print("\nTo use this example:")
    print("1. Create your exchange adapter")
    print("2. Replace 'exchange_adapter' below with your adapter")
    print("3. Run the bot")
    print("\nTo analyze existing data:")
    print("  python quickstart_persistence.py analyze")
    print("="*70)
    
    # Example (replace with your exchange adapter):
    # from my_exchange import MyExchangeAdapter
    # exchange = MyExchangeAdapter(api_key, api_secret)
    # bot = setup_bot_with_persistence(exchange)
    # run_with_monitoring(bot, iterations=10)  # Run 10 iterations for testing
    
    print("\nℹ Uncomment the code above and add your exchange adapter to run.")


if __name__ == "__main__":
    main()
