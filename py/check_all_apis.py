"""Check APIs for all modules."""
import inspect
from trading_bot import TradingConfig, TechnicalIndicators, MLPredictor, TradingStrategy

print("="*70)
print("TradingConfig fields:")
config = TradingConfig()
for field in config.__dataclass_fields__:
    value = getattr(config, field)
    print(f"  {field}: {type(value).__name__} = {value if not isinstance(value, list) else '...'}")

print("\n" + "="*70)
print("TechnicalIndicators methods:")
for name in dir(TechnicalIndicators):
    if not name.startswith('_') and callable(getattr(TechnicalIndicators, name)):
        method = getattr(TechnicalIndicators, name)
        sig = inspect.signature(method)
        print(f"  {name}{sig}")

print("\n" + "="*70)
print("MLPredictor methods:")
predictor = MLPredictor()
for name in dir(predictor):
    if not name.startswith('_') and callable(getattr(predictor, name)):
        method = getattr(predictor, name)
        sig = inspect.signature(method)
        print(f"  {name}{sig}")

print("\n" + "="*70)
print("TradingStrategy methods:")
strategy = TradingStrategy(config)
for name in dir(strategy):
    if not name.startswith('_') and callable(getattr(strategy, name)):
        method = getattr(strategy, name)
        sig = inspect.signature(method)
        print(f"  {name}{sig}")
