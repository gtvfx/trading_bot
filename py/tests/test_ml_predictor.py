"""Tests for ml_predictor.py module - matching actual API."""

import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from trading_bot import MLPredictor, TradingConfig, TechnicalIndicators


def test_ml_predictor_initialization(temp_model_dir):
    """Test MLPredictor initializes correctly."""
    predictor = MLPredictor(model_type="random_forest", model_path=temp_model_dir)
    
    assert predictor.model_type == "random_forest"
    assert predictor.model_path == temp_model_dir
    assert predictor.model is None
    assert predictor.is_trained is False


def test_prepare_features(sample_df_with_indicators):
    """Test feature preparation from DataFrame."""
    predictor = MLPredictor()
    
    features = predictor.prepare_features(sample_df_with_indicators)
    
    # Check that features were created
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0
    
    # Check for engineered features
    assert 'price_change' in features.columns
    assert 'price_momentum' in features.columns
    assert 'price_std' in features.columns


def test_create_target(sample_df_with_indicators):
    """Test target variable creation."""
    predictor = MLPredictor()
    
    target = predictor.create_target(sample_df_with_indicators, forward_periods=5)
    
    assert isinstance(target, pd.Series)
    assert len(target) > 0
    # Target should be binary (0 or 1)
    assert target.isin([0, 1]).all()


def test_train_model_random_forest(sample_df_with_indicators, temp_model_dir):
    """Test training with Random Forest."""
    predictor = MLPredictor(model_type="random_forest", model_path=temp_model_dir)
    
    # Train model
    accuracy = predictor.train(sample_df_with_indicators, forward_periods=5, symbol='BTC-USD')
    
    assert predictor.is_trained is True
    assert predictor.model is not None
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1


def test_train_model_gradient_boosting(sample_df_with_indicators, temp_model_dir):
    """Test training with Gradient Boosting."""
    predictor = MLPredictor(model_type="gradient_boosting", model_path=temp_model_dir)
    
    # Train model
    accuracy = predictor.train(sample_df_with_indicators, forward_periods=5)
    
    assert predictor.is_trained is True
    assert predictor.model is not None
    assert isinstance(accuracy, float)


def test_predict_without_training(sample_df_with_indicators):
    """Test that prediction fails without training."""
    predictor = MLPredictor()
    
    with pytest.raises(ValueError, match="not trained"):
        predictor.predict(sample_df_with_indicators)


def test_predict_after_training(sample_df_with_indicators, temp_model_dir):
    """Test prediction after training."""
    predictor = MLPredictor(model_path=temp_model_dir)
    
    # Train first
    predictor.train(sample_df_with_indicators, forward_periods=5)
    
    # Then predict
    prediction = predictor.predict(sample_df_with_indicators)
    
    assert prediction in [0, 1]


def test_predict_proba(sample_df_with_indicators, temp_model_dir):
    """Test prediction with probability."""
    predictor = MLPredictor(model_path=temp_model_dir)
    
    # Train first
    predictor.train(sample_df_with_indicators, forward_periods=5)
    
    # Get prediction with probability
    prediction, confidence = predictor.predict_proba(sample_df_with_indicators)
    
    assert prediction in [0, 1]
    assert 0 <= confidence <= 1


def test_save_model(sample_df_with_indicators, temp_model_dir):
    """Test saving trained model."""
    predictor = MLPredictor(model_path=temp_model_dir)
    
    # Train model
    predictor.train(sample_df_with_indicators, forward_periods=5)
    
    # Save model
    predictor.save_model(symbol='BTC-USD')
    
    # Check that model files were created (format: modeltype_symbol.joblib)
    model_dir = Path(temp_model_dir)
    model_files = list(model_dir.glob('*.joblib'))
    
    assert len(model_files) > 0, "No model files were saved"


def test_load_model(sample_df_with_indicators, temp_model_dir):
    """Test loading saved model."""
    # Train and save
    predictor1 = MLPredictor(model_path=temp_model_dir)
    predictor1.train(sample_df_with_indicators, forward_periods=5)
    predictor1.save_model(symbol='BTC-USD')
    
    # Create new predictor and load
    predictor2 = MLPredictor(model_path=temp_model_dir)
    success = predictor2.load_model(symbol='BTC-USD')
    
    assert success is True
    assert predictor2.is_trained is True
    assert predictor2.model is not None
    
    # Verify predictions work
    prediction = predictor2.predict(sample_df_with_indicators)
    assert prediction in [0, 1]


def test_load_nonexistent_model(temp_model_dir):
    """Test loading a model that doesn't exist."""
    predictor = MLPredictor(model_path=temp_model_dir)
    
    success = predictor.load_model(symbol='NONEXISTENT-USD')
    
    assert success is False
    assert predictor.is_trained is False


def test_model_persistence(sample_df_with_indicators, temp_model_dir):
    """Test that saved model produces consistent predictions."""
    # Train and predict
    predictor1 = MLPredictor(model_path=temp_model_dir)
    predictor1.train(sample_df_with_indicators, forward_periods=5)
    pred1, conf1 = predictor1.predict_proba(sample_df_with_indicators)
    predictor1.save_model(symbol='BTC-USD')
    
    # Load and predict
    predictor2 = MLPredictor(model_path=temp_model_dir)
    predictor2.load_model(symbol='BTC-USD')
    pred2, conf2 = predictor2.predict_proba(sample_df_with_indicators)
    
    # Predictions should be identical
    assert pred1 == pred2
    assert abs(conf1 - conf2) < 1e-6


def test_insufficient_data_handling():
    """Test handling of insufficient training data."""
    predictor = MLPredictor()
    
    # Create tiny dataset (too small to train)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
        'open': [100 + i for i in range(10)],
        'high': [101 + i for i in range(10)],
        'low': [99 + i for i in range(10)],
        'close': [100 + i for i in range(10)],
        'volume': [1000] * 10
    })
    
    # Should handle gracefully
    with pytest.raises((ValueError, Exception)):
        predictor.train(df, forward_periods=5)


def test_feature_importance(sample_df_with_indicators, temp_model_dir):
    """Test that trained models have feature importance."""
    predictor = MLPredictor(model_type="random_forest", model_path=temp_model_dir)
    
    # Train model
    predictor.train(sample_df_with_indicators, forward_periods=5)
    
    # Random Forest should have feature_importances_
    assert hasattr(predictor.model, 'feature_importances_')
    assert len(predictor.model.feature_importances_) > 0


def test_prediction_confidence_range(sample_df_with_indicators, temp_model_dir):
    """Test that confidence values are in valid range."""
    predictor = MLPredictor(model_path=temp_model_dir)
    predictor.train(sample_df_with_indicators, forward_periods=5)
    
    # Get multiple predictions
    for _ in range(5):
        _, confidence = predictor.predict_proba(sample_df_with_indicators)
        assert 0 <= confidence <= 1


def test_different_forward_periods(sample_df_with_indicators, temp_model_dir):
    """Test training with different forward periods."""
    predictor = MLPredictor(model_path=temp_model_dir)
    
    # Train with 3 periods
    acc1 = predictor.train(sample_df_with_indicators, forward_periods=3)
    
    # Train with 10 periods
    predictor2 = MLPredictor(model_path=temp_model_dir)
    acc2 = predictor2.train(sample_df_with_indicators, forward_periods=10)
    
    # Both should succeed
    assert isinstance(acc1, float)
    assert isinstance(acc2, float)


def test_feature_columns_saved(sample_df_with_indicators, temp_model_dir):
    """Test that feature columns are saved with model."""
    predictor = MLPredictor(model_path=temp_model_dir)
    predictor.train(sample_df_with_indicators, forward_periods=5)
    
    assert len(predictor.feature_columns) > 0
    assert isinstance(predictor.feature_columns, list)
