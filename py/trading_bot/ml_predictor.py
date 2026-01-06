"""Machine learning predictor for price movement forecasting.

Implements ensemble ML models for predicting price direction and confidence scores.
Uses multiple proven algorithms for robust predictions.

"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from ._constants import MODEL_PATH


class MLPredictor:
    """Machine learning predictor for trading signals.
    
    Uses ensemble methods to predict price direction with confidence scores.
    
    """
    
    def __init__(self, model_type: str = "random_forest", model_path: str = MODEL_PATH):
        """Initialize ML predictor.
        
        Args:
            model_type: Type of model ("random_forest" or "gradient_boosting")
            model_path: Directory to save/load models

        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        # Create models directory
        os.makedirs(model_path, exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML training/prediction.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            DataFrame with engineered features

        """
        features = df.copy()
        
        # Price-based features
        features['price_change'] = features['close'].pct_change()
        features['price_momentum'] = features['close'].diff(5)
        
        # Volatility features
        features['price_std'] = features['close'].rolling(20).std()
        features['returns_std'] = features['price_change'].rolling(20).std()
        
        # Trend features
        features['trend_5'] = features['close'].rolling(5).mean() / features['close']
        features['trend_10'] = features['close'].rolling(10).mean() / features['close']
        features['trend_20'] = features['close'].rolling(20).mean() / features['close']
        
        # Volume features
        features['volume_change'] = features['volume'].pct_change()
        features['volume_ratio'] = features['volume'] / features['volume'].rolling(20).mean()
        
        # Indicator features (if available)
        indicator_cols = ['rsi', 'macd', 'macd_signal', 'macd_hist', 
                         'ema_short', 'ema_long', 'bb_upper', 'bb_middle', 'bb_lower',
                         'atr', 'stoch_k', 'stoch_d', 'adx']
        
        for col in indicator_cols:
            if col in df.columns:
                features[f'{col}_value'] = features[col]
        
        # Relative position indicators
        if 'bb_upper' in features.columns and 'bb_lower' in features.columns:
            bb_range = features['bb_upper'] - features['bb_lower']
            features['bb_position'] = (features['close'] - features['bb_lower']) / bb_range
        
        if 'ema_short' in features.columns and 'ema_long' in features.columns:
            features['ema_cross'] = (features['ema_short'] - features['ema_long']) / features['close']
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def create_target(self, df: pd.DataFrame, forward_periods: int = 5) -> pd.Series:
        """Create target variable (future price movement).
        
        Args:
            df: DataFrame with price data
            forward_periods: Periods ahead to predict
            
        Returns:
            Series with target values (1 for up, 0 for down)

        """
        future_returns = df['close'].shift(-forward_periods) / df['close'] - 1
        target = (future_returns > 0).astype(int)
        return target
    
    def train(self, df: pd.DataFrame, forward_periods: int = 5, symbol: Optional[str] = None) -> float:
        """Train the ML model.
        
        Args:
            df: DataFrame with OHLCV and indicator data
            forward_periods: Periods ahead to predict
            symbol: Trading pair symbol (for saving model)
            
        Returns:
            Model accuracy score

        """
        # Prepare features and target
        features = self.prepare_features(df)
        target = self.create_target(df, forward_periods)
        
        # Align features and target
        valid_idx = features.index.intersection(target.index)
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]
        
        # Select feature columns (exclude OHLCV and some raw indicators)
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.feature_columns = [col for col in features.columns if col not in exclude_cols]
        
        X = features[self.feature_columns]
        y = target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        self.is_trained = True
        
        print(f"Model trained - Train accuracy: {train_score:.2%}, Test accuracy: {test_score:.2%}")
        
        # Auto-save after training
        if symbol:
            self.save_model(symbol=symbol)
        
        return float(test_score)
    
    def predict(self, df: pd.DataFrame) -> int:
        """Predict price direction.
        
        Args:
            df: DataFrame with current market data
            
        Returns:
            Prediction: 1 for up, 0 for down

        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self.prepare_features(df)
        X = features[self.feature_columns].iloc[[-1]]
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        return int(prediction)
    
    def predict_proba(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Predict price direction with confidence.
        
        Args:
            df: DataFrame with current market data
            
        Returns:
            Tuple of (prediction, confidence_score)

        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self.prepare_features(df)
        X = features[self.feature_columns].iloc[[-1]]
        X_scaled = self.scaler.transform(X)
        
        probas = self.model.predict_proba(X_scaled)[0]
        prediction = int(np.argmax(probas))
        confidence = probas[prediction]
        
        return prediction, confidence
    
    def save_model(self, symbol: Optional[str] = None):
        """Save trained model to disk.
        
        Args:
            symbol: Trading pair symbol (optional, for per-symbol models)

        """
        if not self.is_trained:
            print("⚠ No trained model to save")
            return
        
        filename = f"{self.model_path}/{self.model_type}"
        if symbol:
            # Remove special characters from symbol for filename
            clean_symbol = symbol.replace('-', '_').replace('/', '_')
            filename += f"_{clean_symbol}"
        filename += ".joblib"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filename)
        print(f"✓ Model saved to {filename}")
    
    def load_model(self, symbol: Optional[str] = None) -> bool:
        """Load trained model from disk if it exists.
        
        Args:
            symbol: Trading pair symbol (optional, for per-symbol models)
            
        Returns:
            True if model loaded successfully, False otherwise

        """
        filename = f"{self.model_path}/{self.model_type}"
        if symbol:
            clean_symbol = symbol.replace('-', '_').replace('/', '_')
            filename += f"_{clean_symbol}"
        filename += ".joblib"
        
        if not os.path.exists(filename):
            return False
        
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_type = model_data['model_type']
            self.is_trained = True
            print(f"✓ Loaded existing model from {filename}")
            return True
        except Exception as e:
            print(f"⚠ Failed to load model from {filename}: {e}")
            return False
