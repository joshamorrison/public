"""
Tests for MMM model functionality
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from models.mmm.econometric_mmm import EconometricMMM

class TestEconometricMMM:
    """Test cases for EconometricMMM model"""
    
    def test_init_default(self):
        """Test default initialization"""
        model = EconometricMMM()
        
        assert model.adstock_rate == 0.5
        assert model.saturation_param == 0.6
        assert model.regularization_alpha == 0.1
        assert model.include_baseline is True
        assert model.mlflow_tracker is None
    
    def test_init_with_params(self):
        """Test initialization with custom parameters"""
        model = EconometricMMM(
            adstock_rate=0.7,
            saturation_param=0.8,
            regularization_alpha=0.05,
            include_baseline=False
        )
        
        assert model.adstock_rate == 0.7
        assert model.saturation_param == 0.8
        assert model.regularization_alpha == 0.05
        assert model.include_baseline is False
    
    def test_adstock_transformation(self, sample_marketing_data):
        """Test adstock transformation"""
        model = EconometricMMM()
        
        # Test with sample data
        spend_data = sample_marketing_data['tv_spend'].values
        adstocked = model._apply_adstock(spend_data, 0.5)
        
        # Should be same length
        assert len(adstocked) == len(spend_data)
        
        # Should be non-negative
        assert (adstocked >= 0).all()
        
        # Should have carryover effect (later values influenced by earlier)
        assert adstocked[1] >= spend_data[1]  # Should include carryover
    
    def test_saturation_transformation(self, sample_marketing_data):
        """Test saturation transformation"""
        model = EconometricMMM()
        
        spend_data = sample_marketing_data['tv_spend'].values
        saturated = model._apply_saturation(spend_data, 0.6)
        
        # Should be same length
        assert len(saturated) == len(spend_data)
        
        # Should be between 0 and 1 (normalized)
        assert (saturated >= 0).all()
        assert (saturated <= 1).all()
    
    def test_fit_basic(self, sample_marketing_data):
        """Test basic model fitting"""
        model = EconometricMMM()
        
        spend_columns = ['tv_spend', 'digital_spend']
        
        results = model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'performance' in results
        assert 'coefficients' in results
        assert 'transformations' in results
        
        # Check performance metrics
        performance = results['performance']
        assert 'r2_score' in performance
        assert 'mape' in performance
        assert isinstance(performance['r2_score'], (int, float))
        assert isinstance(performance['mape'], (int, float))
        
        # R² should be reasonable
        assert -1 <= performance['r2_score'] <= 1
        
        # MAPE should be positive
        assert performance['mape'] >= 0
    
    def test_fit_with_synergies(self, sample_marketing_data):
        """Test model fitting with synergies"""
        model = EconometricMMM()
        
        spend_columns = ['tv_spend', 'digital_spend', 'radio_spend']
        
        results = model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=True
        )
        
        # Should have synergy features
        transformations = results['transformations']
        feature_names = transformations['feature_names']
        
        # Should have interaction terms when synergies enabled
        synergy_features = [name for name in feature_names if '_x_' in name]
        assert len(synergy_features) > 0, "Should have synergy features when include_synergies=True"
    
    def test_feature_engineering(self, sample_marketing_data):
        """Test feature engineering pipeline"""
        model = EconometricMMM()
        
        spend_columns = ['tv_spend', 'digital_spend']
        
        # Test the internal feature engineering
        features = model._engineer_features(
            data=sample_marketing_data,
            spend_columns=spend_columns,
            include_synergies=True
        )
        
        # Should be a DataFrame
        assert isinstance(features, pd.DataFrame)
        
        # Should have transformed columns
        assert len(features.columns) > len(spend_columns)
        
        # Should have adstocked and saturated features
        adstock_features = [col for col in features.columns if 'adstock' in col]
        saturation_features = [col for col in features.columns if 'saturated' in col]
        
        assert len(adstock_features) > 0
        assert len(saturation_features) > 0
    
    def test_predict(self, sample_marketing_data):
        """Test model prediction"""
        model = EconometricMMM()
        
        spend_columns = ['tv_spend', 'digital_spend']
        
        # First fit the model
        model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        # Test prediction
        predictions = model.predict(sample_marketing_data[spend_columns])
        
        # Should return array-like
        assert hasattr(predictions, '__len__')
        assert len(predictions) == len(sample_marketing_data)
        
        # Should be numeric
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    
    def test_attribution_analysis(self, sample_marketing_data):
        """Test attribution analysis"""
        model = EconometricMMM()
        
        spend_columns = ['tv_spend', 'digital_spend', 'radio_spend']
        
        # Fit model first
        model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        # Test attribution
        attribution = model.get_attribution_analysis(sample_marketing_data[spend_columns])
        
        # Should return dict
        assert isinstance(attribution, dict)
        
        # Should have channel attributions
        for channel in spend_columns:
            channel_key = channel.replace('_spend', '')
            assert channel_key in attribution or channel in attribution
    
    def test_incremental_analysis(self, sample_marketing_data):
        """Test incremental analysis"""
        model = EconometricMMM()
        
        spend_columns = ['tv_spend', 'digital_spend']
        
        # Fit model first
        model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        # Test incremental analysis
        current_spend = {
            'tv_spend': 50000,
            'digital_spend': 30000
        }
        
        incremental_spend = {
            'tv_spend': 55000,  # +10% increase
            'digital_spend': 33000  # +10% increase  
        }
        
        incremental = model.calculate_incremental_impact(current_spend, incremental_spend)
        
        # Should return dict with impact metrics
        assert isinstance(incremental, dict)
        assert 'total_incremental_revenue' in incremental
        assert 'channel_contributions' in incremental
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        model = EconometricMMM()
        
        # Test with empty data
        empty_data = pd.DataFrame()
        
        with pytest.raises((ValueError, KeyError)):
            model.fit(
                data=empty_data,
                target_column='revenue',
                spend_columns=['tv_spend'],
                include_synergies=False
            )
        
        # Test with missing columns
        small_data = pd.DataFrame({'other_col': [1, 2, 3]})
        
        with pytest.raises((ValueError, KeyError)):
            model.fit(
                data=small_data,
                target_column='revenue',
                spend_columns=['tv_spend'],
                include_synergies=False
            )
    
    @patch('models.mmm.econometric_mmm.setup_mlflow_tracking')
    def test_mlflow_integration(self, mock_mlflow, sample_marketing_data):
        """Test MLflow integration"""
        # Mock MLflow tracker
        mock_tracker = MagicMock()
        mock_mlflow.return_value = mock_tracker
        
        # Test with MLflow tracker
        model = EconometricMMM(mlflow_tracker=mock_tracker)
        
        spend_columns = ['tv_spend', 'digital_spend']
        
        results = model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        # Should complete without error
        assert isinstance(results, dict)

class TestModelValidation:
    """Test model validation and quality checks"""
    
    def test_model_performance_bounds(self, sample_marketing_data):
        """Test that model performance is within reasonable bounds"""
        model = EconometricMMM()
        
        spend_columns = ['tv_spend', 'digital_spend', 'radio_spend']
        
        results = model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        performance = results['performance']
        
        # R² should be reasonable for synthetic data
        assert performance['r2_score'] >= -1  # Can be negative for bad models
        assert performance['r2_score'] <= 1   # Cannot exceed 1
        
        # MAPE should be reasonable
        assert performance['mape'] >= 0
        assert performance['mape'] <= 1000  # Very high but not infinite
    
    def test_feature_importance(self, sample_marketing_data):
        """Test that model produces reasonable feature importance"""
        model = EconometricMMM()
        
        spend_columns = ['tv_spend', 'digital_spend', 'radio_spend']
        
        results = model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        coefficients = results['coefficients']
        
        # Should have coefficients for each channel
        assert len(coefficients) > 0
        
        # Coefficients should be numeric
        for coef in coefficients.values():
            assert isinstance(coef, (int, float, np.number))
    
    def test_prediction_consistency(self, sample_marketing_data):
        """Test that predictions are consistent"""
        model = EconometricMMM()
        
        spend_columns = ['tv_spend', 'digital_spend']
        
        # Fit model
        model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        # Make predictions twice with same data
        pred1 = model.predict(sample_marketing_data[spend_columns])
        pred2 = model.predict(sample_marketing_data[spend_columns])
        
        # Should be identical
        np.testing.assert_array_equal(pred1, pred2)

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_single_channel(self, sample_marketing_data):
        """Test model with single channel"""
        model = EconometricMMM()
        
        results = model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=['tv_spend'],  # Single channel
            include_synergies=False
        )
        
        # Should work with single channel
        assert isinstance(results, dict)
        assert 'performance' in results
    
    def test_zero_spend(self):
        """Test model with zero spend scenarios"""
        model = EconometricMMM()
        
        # Create data with some zero spend
        data = pd.DataFrame({
            'tv_spend': [0, 10000, 20000, 0, 15000],
            'digital_spend': [5000, 0, 15000, 10000, 0],
            'revenue': [20000, 30000, 50000, 25000, 35000]
        })
        
        results = model.fit(
            data=data,
            target_column='revenue',
            spend_columns=['tv_spend', 'digital_spend'],
            include_synergies=False
        )
        
        # Should handle zero spend gracefully
        assert isinstance(results, dict)
    
    def test_small_dataset(self, small_marketing_data):
        """Test model with small dataset"""
        model = EconometricMMM()
        
        results = model.fit(
            data=small_marketing_data,
            target_column='revenue',
            spend_columns=['tv_spend', 'digital_spend'],
            include_synergies=False
        )
        
        # Should work with small dataset
        assert isinstance(results, dict)
        assert 'performance' in results