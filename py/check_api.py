"""Check actual API signatures."""
import inspect
import tempfile
from trading_bot.trade_history import TradeHistory

# Create temporary instance
with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    h = TradeHistory(f.name)
    
print("TradeHistory public methods:")
for name in dir(h):
    if not name.startswith('_'):
        attr = getattr(h, name)
        if callable(attr):
            sig = inspect.signature(attr)
            print(f"  {name}{sig}")
