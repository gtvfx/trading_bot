"""Test that constants are properly accessible."""

import sys
import os

# Add parent directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_constants_import():
    """Test importing constants."""
    from trading_bot import DB_PATH, MODEL_PATH
    
    print("✓ Constants imported successfully")
    print(f"  DB_PATH: {DB_PATH}")
    print(f"  MODEL_PATH: {MODEL_PATH}")
    
    return True

def test_config_uses_constants():
    """Test that config uses constants."""
    from trading_bot import TradingConfig, DB_PATH, MODEL_PATH
    
    config = TradingConfig()
    
    assert config.db_path == DB_PATH, f"Expected {DB_PATH}, got {config.db_path}"
    assert config.model_path == MODEL_PATH, f"Expected {MODEL_PATH}, got {config.model_path}"
    
    print("✓ Config uses constants correctly")
    print(f"  config.db_path: {config.db_path}")
    print(f"  config.model_path: {config.model_path}")
    
    return True

def test_environment_override():
    """Test environment variable override."""
    # Set environment variables
    os.environ["TRADING_BOT_DB"] = "test/db.db"
    os.environ["TRADING_BOT_MODELS"] = "test/models"
    
    # Reload module to pick up new environment
    from importlib import reload
    import trading_bot._constants as constants
    reload(constants)
    
    print("✓ Environment variables work")
    print(f"  DB_PATH after env: {constants.DB_PATH}")
    print(f"  MODEL_PATH after env: {constants.MODEL_PATH}")
    
    # Cleanup
    del os.environ["TRADING_BOT_DB"]
    del os.environ["TRADING_BOT_MODELS"]
    
    return True

if __name__ == "__main__":
    print("="*70)
    print("TESTING CONSTANTS MODULE")
    print("="*70)
    
    try:
        test_constants_import()
        print()
        test_config_uses_constants()
        print()
        test_environment_override()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
