"""Example: Using environment variables for configuration.

Shows how to customize database and model paths using environment variables.

"""

import os
from trading_bot import TradingBot, TradingConfig, DB_PATH, MODEL_PATH

def example_with_environment_variables():
    """Example using environment variables."""
    
    print("Current configuration:")
    print(f"  Database: {DB_PATH}")
    print(f"  Models: {MODEL_PATH}")
    
    # These are set from environment variables:
    # - TRADING_BOT_DB (defaults to "data/trades.db")
    # - TRADING_BOT_MODELS (defaults to "models")
    
    # Option 1: Use defaults (from environment or hardcoded)
    config = TradingConfig(
        trading_pairs=["BTC-USD"],
        ml_enabled=True
        # db_path and model_path will use values from _constants
    )
    
    print(f"\nConfig uses:")
    print(f"  db_path: {config.db_path}")
    print(f"  model_path: {config.model_path}")


def example_with_custom_paths():
    """Example with custom paths (overriding environment)."""
    
    # Option 2: Override with custom paths
    config = TradingConfig(
        trading_pairs=["BTC-USD"],
        ml_enabled=True,
        db_path="custom/location/trades.db",  # Override default
        model_path="custom/models"  # Override default
    )
    
    print(f"\nCustom config uses:")
    print(f"  db_path: {config.db_path}")
    print(f"  model_path: {config.model_path}")


def example_set_environment():
    """Example: Setting environment variables in code."""
    
    # Set environment variables before importing
    os.environ["TRADING_BOT_DB"] = "/secure/storage/trades.db"
    os.environ["TRADING_BOT_MODELS"] = "/secure/storage/models"
    
    # Reload constants to pick up new environment variables
    from importlib import reload
    import trading_bot._constants as constants
    reload(constants)
    
    print(f"\nAfter setting environment:")
    print(f"  DB_PATH: {constants.DB_PATH}")
    print(f"  MODEL_PATH: {constants.MODEL_PATH}")


if __name__ == "__main__":
    print("="*70)
    print("ENVIRONMENT VARIABLE CONFIGURATION")
    print("="*70)
    
    example_with_environment_variables()
    example_with_custom_paths()
    
    print("\n" + "="*70)
    print("SETTING ENVIRONMENT VARIABLES")
    print("="*70)
    print("\nIn bash/zsh:")
    print("  export TRADING_BOT_DB=\"/path/to/trades.db\"")
    print("  export TRADING_BOT_MODELS=\"/path/to/models\"")
    print("  python my_bot.py")
    
    print("\nIn PowerShell:")
    print("  $env:TRADING_BOT_DB=\"C:\\path\\to\\trades.db\"")
    print("  $env:TRADING_BOT_MODELS=\"C:\\path\\to\\models\"")
    print("  python my_bot.py")
    
    print("\nIn Python script:")
    print("  import os")
    print("  os.environ['TRADING_BOT_DB'] = '/path/to/trades.db'")
    print("  from trading_bot import TradingBot, TradingConfig")
    
    print("\n" + "="*70)
