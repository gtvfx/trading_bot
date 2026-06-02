"""Paper Trading Example with Coinbase.

This example shows how to run the trading_bot in paper trading mode
using the Coinbase Advanced Trade API.

Requirements:
    - Coinbase Advanced Trade API credentials
    - coinbase package (coinbase-advanced-py)
    - trading_bot package installed

Setup:
    1. Set up your Coinbase API credentials:
       - Create API keys at https://www.coinbase.com/settings/api
       - Save them securely using the KeyManager
    
    2. Run this script:
       python paper_trading_coinbase.py

Note: This example creates a simple adapter for Coinbase. For production use,
      consider creating a full CoinbaseExchangeAdapter similar to the
      Robinhood adapter.

"""

import os
import sys
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

# Coinbase imports
try:
    from cb.trading import CoinbaseClient, KeyManager
    from cb.personal import get_keys, setup_keys
except ImportError:
    print("✗ Could not import Coinbase modules")
    print("  Make sure coinbase package is installed")
    sys.exit(1)

# Trading bot imports
from trading_bot import TradingBot, TradingConfig
from trading_bot.exchange_client import ExchangeClient


class CoinbaseExchangeAdapter(ExchangeClient):
    """Simple Coinbase adapter for trading_bot.
    
    This is a minimal adapter to demonstrate paper trading with Coinbase.
    For production use, create a full-featured adapter with proper error
    handling, rate limiting, and all ExchangeClient methods implemented.
    
    """
    
    def __init__(self, client: CoinbaseClient):
        """Initialize adapter.
        
        Args:
            client: Configured CoinbaseClient instance
            
        """
        self.client = client
    
    def get_historical_data(
        self,
        symbol: str,
        granularity: str = "1h",
        limit: int = 100
    ) -> pd.DataFrame:
        """Get historical price data.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            granularity: Candle size ("1h", "1d", etc.)
            limit: Number of candles
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            
        """
        # Map granularity to Coinbase format
        granularity_map = {
            "1m": "ONE_MINUTE",
            "5m": "FIVE_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "30m": "1800",
            "1h": "ONE_HOUR",
            "6h": "SIX_HOUR",
            "1d": "ONE_DAY"
        }
        coinbase_granularity = granularity_map.get(granularity, "ONE_HOUR")
        
        # Calculate time range based on granularity and limit
        granularity_seconds = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "1800": 1800,
            "ONE_HOUR": 3600,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400
        }
        
        seconds = granularity_seconds.get(coinbase_granularity, 3600)
        end_time = int(datetime.now().timestamp())
        start_time = end_time - (seconds * limit)
        
        # Get candles from Coinbase
        response = self.client.get_candles(
            product_id=symbol,
            start=str(start_time),
            end=str(end_time),
            granularity=coinbase_granularity
        )
        
        # Convert to DataFrame
        if response and 'candles' in response and response['candles']:
            candles = response['candles']
            df = pd.DataFrame(candles)
            
            # Check if we have data
            if df.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert columns to proper types
            df['timestamp'] = pd.to_numeric(df['start'], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Select only needed columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Sort by timestamp and remove duplicates
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            return df
        
        # Return empty DataFrame if no data
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Current price as float
            
        """
        product = self.client.get_product(symbol)
        if product and 'price' in product:
            return float(product['price'])
        return 0.0
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balances.
        
        Returns:
            Dictionary of currency -> available balance
            
        """
        balances = {}
        accounts = self.client.get_accounts()
        
        if accounts and 'accounts' in accounts:
            for account in accounts['accounts']:
                currency = account.get('currency', '')
                available = account.get('available_balance', {}).get('value', '0')
                balances[currency] = float(available)
        
        return balances
    
    def get_buying_power(self) -> float:
        """Get available buying power in USD."""
        balances = self.get_account_balance()
        return balances.get('USD', 0.0)
    
    def get_historical_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        granularity: str
    ) -> pd.DataFrame:
        """Get historical OHLCV candlestick data."""
        # Map granularity to Coinbase format
        granularity_map = {
            "1m": "ONE_MINUTE",
            "5m": "FIVE_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "30m": "THIRTY_MINUTE",
            "1h": "ONE_HOUR",
            "6h": "SIX_HOUR",
            "1d": "ONE_DAY"
        }
        coinbase_granularity = granularity_map.get(granularity, "ONE_HOUR")
        
        # Get candles from Coinbase
        try:
            response = self.client.get_candles(
                product_id=symbol,
                start=str(int(start.timestamp())),
                end=str(int(end.timestamp())),
                granularity=coinbase_granularity
            )
        except Exception as e:
            print(f"Error fetching candles for {symbol}: {e}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert to DataFrame
        if response and 'candles' in response and response['candles']:
            candles = response['candles']
            df = pd.DataFrame(candles)
            
            # Check if we have data
            if df.empty:
                print(f"Warning: Empty candles returned for {symbol}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert columns to proper types
            df['timestamp'] = pd.to_numeric(df['start'], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Select only needed columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Sort by timestamp and remove duplicates
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            print(f"Fetched {len(df)} candles for {symbol}")
            return df
        
        # Return empty DataFrame if no data
        print(f"Warning: No candle data in response for {symbol}")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Dict:
        """Place a market order."""
        # This would place a real order - only used in live trading
        # In paper trading mode, this method won't be called
        response = self.client.create_market_order(
            product_id=symbol,
            side=side,
            size=str(quantity)
        )
        return response or {}
    
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Dict:
        """Place a limit order."""
        response = self.client.create_limit_order(
            product_id=symbol,
            side=side,
            size=str(quantity),
            price=str(price)
        )
        return response or {}
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            self.client.cancel_orders(order_ids=[order_id])
            return True
        except:
            return False
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status."""
        response = self.client.get_order(order_id)
        return response or {}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders."""
        response = self.client.list_orders(product_id=symbol) if symbol else self.client.list_orders()
        if response and 'orders' in response:
            return response['orders']
        return []
    
    def get_holdings(self) -> Dict[str, float]:
        """Get current holdings/positions."""
        holdings = {}
        accounts = self.client.get_accounts()
        
        if accounts and 'accounts' in accounts:
            for account in accounts['accounts']:
                currency = account.get('currency', '')
                balance = account.get('available_balance', {}).get('value', '0')
                holdings[currency] = float(balance)
        
        return holdings


def setup_coinbase_adapter(read_only: bool = True):
    """Set up Coinbase adapter with secure credentials.
    
    Args:
        read_only: If True, use read-only credentials. False for trading credentials.
    
    Returns:
        CoinbaseExchangeAdapter instance
        
    Raises:
        ValueError: If credentials are not configured
        
    """
    # Option 1: Use your existing encrypted key setup
    try:
        print("✓ Loading from encrypted key files...")
        manager = setup_keys(read_only=read_only)
        cipher_key = manager.load_cipher_key()
        client = CoinbaseClient.from_key_manager(manager, cipher_key=cipher_key)
        print("✓ Successfully loaded credentials from encrypted storage")
        return CoinbaseExchangeAdapter(client)
    except Exception as e:
        print(f"⚠️  Could not load from encrypted storage: {e}")
    
    # Option 2: Use environment variables (fallback)
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")
    
    if api_key and api_secret:
        print("✓ Using environment variable credentials")
        client = CoinbaseClient(api_key, api_secret)
        return CoinbaseExchangeAdapter(client)
    
    # No credentials found
    print("\n✗ Could not load credentials")
    print("\nYour setup:")
    print("  You have encrypted key files in cb/personal/")
    print("  Make sure SERVICE_ROOT environment variable is set")
    print("\nAlternatively, use environment variables:")
    print("  $env:COINBASE_API_KEY='your_key_name'")
    print("  $env:COINBASE_API_SECRET='your_private_key_pem'")
    print("     python -m cb.trading.setup_keys")
    raise ValueError("Coinbase credentials not configured")


def run_paper_trading():
    """Run trading bot in paper trading mode with Coinbase."""
    
    print("=" * 80)
    print(" COINBASE PAPER TRADING EXAMPLE")
    print("=" * 80)
    print()
    
    # 1. Set up Coinbase adapter
    # Note: Use read_only=False if you want to test with trading credentials
    try:
        adapter = setup_coinbase_adapter(read_only=True)
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
        lookback_window=300,  # Use 200 candles for analysis (increased for ML)
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
