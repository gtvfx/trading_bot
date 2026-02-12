"""Paper Trading Example with Robinhood.

This example shows how to run the trading_bot in paper trading mode
using the Robinhood exchange adapter.

Requirements:
    - Robinhood API credentials (API key and private key)
    - robinhood package installed
    - trading_bot package installed

Setup:
    1. Set up your Robinhood API credentials:
       export ROBINHOOD_API_KEY="your_api_key"
       export ROBINHOOD_PRIVATE_KEY="your_private_key"
    
    2. Run this script:
       python paper_trading_robinhood.py

"""

import os
import sys
from robinhood.trading import RobinhoodClient, RobinhoodExchangeAdapter, get_keys
from trading_bot import TradingBot, TradingConfig


def setup_robinhood_adapter():
    """Set up Robinhood adapter with secure credentials.
    
    Returns:
        RobinhoodExchangeAdapter instance
        
    Raises:
        ValueError: If credentials are not configured
        
    """
    # Option 1: Use your existing encrypted key setup
    try:
        print("✓ Loading from encrypted key files...")
        private_key, public_key, api_key = get_keys()
        client = RobinhoodClient(api_key, private_key)
        print("✓ Successfully loaded credentials from SERVICE_ROOT/robinhood.json")
        return RobinhoodExchangeAdapter(client)
    except Exception as e:
        print(f"⚠️  Could not load from encrypted storage: {e}")
    
    # Option 2: Use environment variables (fallback)
    api_key = os.getenv("ROBINHOOD_API_KEY")
    private_key = os.getenv("ROBINHOOD_PRIVATE_KEY")
    
    if api_key and private_key:
        print("✓ Using environment variable credentials")
        client = RobinhoodClient(api_key, private_key)
        return RobinhoodExchangeAdapter(client)
    
    # No credentials found
    print("\n✗ Could not load credentials")
    print("\nYour setup:")
    print("  You have encrypted key files in robinhood/config/")
    print("  Make sure SERVICE_ROOT environment variable is set")
    print("\nAlternatively, use environment variables:")
    print("  $env:ROBINHOOD_API_KEY='your_key'")
    print("  $env:ROBINHOOD_PRIVATE_KEY='your_private_key'")
    raise ValueError("Robinhood credentials not configured")


def run_paper_trading():
    """Run trading bot in paper trading mode."""
    
    print("=" * 80)
    print(" ROBINHOOD PAPER TRADING EXAMPLE")
    print("=" * 80)
    print()
    
    # 1. Set up Robinhood adapter
    try:
        adapter = setup_robinhood_adapter()
    except ValueError as e:
        print(f"\n✗ Setup failed: {e}")
        return
    
    # 2. Configure trading bot for paper trading
    config = TradingConfig(
        # Trading pairs to monitor
        trading_pairs=["BTC-USD", "ETH-USD"],
        
        # Paper trading settings
        paper_trading=True,  # 🧪 Enable paper trading mode
        paper_trading_balance=10000.0,  # Start with $10,000 virtual money
        
        # Position sizing
        position_size_percent=5.0,  # Use 5% of balance per trade
        max_positions=2,  # Maximum 2 simultaneous positions
        
        # Risk management
        stop_loss_percent=2.0,  # Exit if down 2%
        take_profit_percent=5.0,  # Take profit at 5% gain
        max_daily_loss_percent=5.0,  # Stop trading if down 5% today
        
        # Machine learning - disabled for initial testing
        ml_enabled=False,  # Disable ML to test basic trading first
        # ml_model_type="gradient_boosting",  # or "random_forest"
        # ml_confidence_threshold=0.65,  # Require 65% confidence
        # ml_retrain_interval=100,  # Retrain every 100 iterations
        
        # Strategy settings
        min_signal_agreement=3,  # Need 3+ indicators to agree
        lookback_window=200,  # Use 200 candles for analysis (increased for ML)
        candle_granularity="1h",  # 1-hour candles
        refresh_interval=300  # Check every 5 minutes (300 seconds)
    )
    
    # 3. Create bot instance
    bot = TradingBot(adapter, config)
    
    # 4. Display configuration
    print("\n📋 Configuration:")
    print(f"   Mode: {'🧪 PAPER TRADING' if config.paper_trading else '💰 LIVE TRADING'}")
    print(f"   Virtual Balance: ${config.paper_trading_balance:,.2f}")
    print(f"   Trading Pairs: {', '.join(config.trading_pairs)}")
    print(f"   Position Size: {config.position_size_percent}%")
    print(f"   Max Positions: {config.max_positions}")
    print(f"   Stop Loss: {config.stop_loss_percent}%")
    print(f"   Take Profit: {config.take_profit_percent}%")
    print(f"   ML Enabled: {'Yes' if config.ml_enabled else 'No'}")
    print()
    
    print("⚠️  This is PAPER TRADING - No real money will be used!")
    print("   All trades are simulated for testing purposes.")
    print()
    
    # 5. Run bot with limited iterations for testing
    iterations = 20  # Run for 20 iterations (about 1.5 hours with 5min interval)
    
    print(f"🚀 Starting bot for {iterations} iterations...")
    print("   Press Ctrl+C to stop early\n")
    
    try:
        bot.run(iterations=iterations)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    
    except Exception as e:
        print(f"\n\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Display results
    print()
    print("=" * 80)
    print(" PAPER TRADING RESULTS")
    print("=" * 80)
    print()
    
    initial_balance = config.paper_trading_balance
    final_balance = bot.paper_balance
    final_equity = bot.paper_equity
    total_value = final_balance + final_equity
    
    print(f"💰 Final Balances:")
    print(f"   Cash: ${final_balance:,.2f}")
    print(f"   Equity (open positions): ${final_equity:,.2f}")
    print(f"   Total Portfolio: ${total_value:,.2f}")
    print()
    
    # Calculate performance
    pnl = total_value - initial_balance
    pnl_pct = (pnl / initial_balance) * 100
    
    print(f"📊 Performance:")
    print(f"   Initial Balance: ${initial_balance:,.2f}")
    print(f"   P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
    print()
    
    # Recommendation
    if pnl > 0:
        print("✅ Strategy was profitable in paper trading!")
        print("   Next steps:")
        print("   1. Review the trades and understand why it worked")
        print("   2. Test with different market conditions")
        print("   3. When confident, switch to live trading with:")
        print("      - Very small position sizes (1-2%)")
        print("      - Close monitoring")
        print("      - Clear exit plan")
    else:
        print("⚠️  Strategy lost money in paper trading")
        print("   Recommendations:")
        print("   1. Adjust strategy parameters")
        print("   2. Review indicator signals")
        print("   3. Consider different trading pairs")
        print("   4. Keep testing in paper mode until profitable")
    
    print()
    print("💡 To switch to LIVE trading:")
    print("   1. Set config.paper_trading = False")
    print("   2. START VERY SMALL (position_size_percent=1.0 or less)")
    print("   3. Monitor closely and be ready to stop")
    print("   4. Never risk more than you can afford to lose!")
    print()


if __name__ == "__main__":
    run_paper_trading()
