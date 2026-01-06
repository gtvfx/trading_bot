"""Abstract base class for exchange clients.

This module defines the interface that all exchange adapters must implement.
This allows the trading bot to work with any exchange (Coinbase, Robinhood, etc.)
without changing the core logic.

"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd


class ExchangeClient(ABC):
    """Abstract base class for cryptocurrency exchange clients.
    
    All exchange adapters must implement these methods to work with the trading bot.
    
    """
    
    @abstractmethod
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balances for all assets.
        
        Returns:
            Dictionary mapping asset code to available balance
            Example: {'USD': 10000.0, 'BTC': 0.5, 'ETH': 2.0}

        """
        pass
    
    @abstractmethod
    def get_buying_power(self) -> float:
        """Get available buying power in USD.
        
        Returns:
            Available USD balance for trading

        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a trading pair.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            
        Returns:
            Current price or None if unavailable

        """
        pass
    
    @abstractmethod
    def get_historical_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        granularity: str
    ) -> pd.DataFrame:
        """Get historical OHLCV candlestick data.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            start: Start datetime
            end: End datetime
            granularity: Time interval (e.g., "1h", "1d")
            
        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]

        """
        pass
    
    @abstractmethod
    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Dict[str, Any]:
        """Place a market order.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            side: "buy" or "sell"
            quantity: Amount of asset to trade
            
        Returns:
            Order details dictionary with at least {'id', 'status', 'filled_quantity'}

        """
        pass
    
    @abstractmethod
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Dict[str, Any]:
        """Place a limit order.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            side: "buy" or "sell"
            quantity: Amount of asset to trade
            price: Limit price
            
        Returns:
            Order details dictionary with at least {'id', 'status'}

        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successfully cancelled, False otherwise

        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status and details.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order details dictionary with at least {'id', 'status', 'filled_quantity'}

        """
        pass
    
    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders.
        
        Args:
            symbol: Optional trading pair to filter by
            
        Returns:
            List of order dictionaries

        """
        pass
    
    @abstractmethod
    def get_holdings(self) -> Dict[str, float]:
        """Get current holdings/positions.
        
        Returns:
            Dictionary mapping asset code to total quantity held

        """
        pass
    
    # Optional methods with default implementations
    
    def supports_stop_loss(self) -> bool:
        """Check if exchange supports stop-loss orders.
        
        Returns:
            True if stop-loss orders are supported

        """
        return False
    
    def place_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float
    ) -> Dict[str, Any]:
        """Place a stop-loss order (if supported).
        
        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            quantity: Amount of asset
            stop_price: Stop price trigger
            
        Returns:
            Order details dictionary
            
        Raises:
            NotImplementedError: If exchange doesn't support stop-loss

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support stop-loss orders"
        )
