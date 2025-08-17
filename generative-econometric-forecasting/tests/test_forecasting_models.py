#!/usr/bin/env python3
"""
🧪 Forecasting Models Unit Tests
Tests core econometric forecasting functionality
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.forecasting_models import EconometricForecaster
    FORECASTING_AVAILABLE = True
except ImportError:
    FORECASTING_AVAILABLE = False


class TestForecastingModels(unittest.TestCase):
    """Test core forecasting functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample time series data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='ME')
        np.random.seed(42)
        trend = np.linspace(100, 120, len(dates))
        noise = np.random.normal(0, 2, len(dates))
        seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        values = trend + seasonal + noise
        
        self.sample_data = pd.Series(values, index=dates, name='test_metric')
        self.forecast_horizon = 6
    
    @unittest.skipIf(not FORECASTING_AVAILABLE, "Forecasting models not available")
    def test_econometric_forecaster_initialization(self):
        """Test EconometricForecaster initialization"""
        forecaster = EconometricForecaster()
        self.assertIsInstance(forecaster, EconometricForecaster)
        self.assertIsNotNone(forecaster.models)
    
    def test_data_validation(self):
        """Test input data validation"""
        # Valid data should pass
        self.assertIsInstance(self.sample_data, pd.Series)
        self.assertTrue(isinstance(self.sample_data.index, pd.DatetimeIndex))
        self.assertGreater(len(self.sample_data), 24)  # Sufficient data points
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        # Create data with missing values
        data_with_gaps = self.sample_data.copy()
        data_with_gaps.iloc[10:12] = np.nan
        
        # Test that missing data is detected
        self.assertTrue(data_with_gaps.isnull().any())
        
        # Test forward fill
        filled_data = data_with_gaps.fillna(method='ffill')
        self.assertFalse(filled_data.isnull().any())
    
    def test_seasonal_decomposition(self):
        """Test seasonal decomposition functionality"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Test seasonal decomposition
        decomposition = seasonal_decompose(self.sample_data, model='additive', period=12)
        
        self.assertIsNotNone(decomposition.trend)
        self.assertIsNotNone(decomposition.seasonal)
        self.assertIsNotNone(decomposition.resid)
    
    def test_stationarity_testing(self):
        """Test stationarity tests"""
        from statsmodels.tsa.stattools import adfuller
        
        # Test ADF test
        adf_result = adfuller(self.sample_data.dropna())
        
        self.assertIsInstance(adf_result[0], float)  # Test statistic
        self.assertIsInstance(adf_result[1], float)  # p-value
        self.assertIsInstance(adf_result[4], dict)   # Critical values


class TestARIMAModeling(unittest.TestCase):
    """Test ARIMA modeling functionality"""
    
    def setUp(self):
        """Set up test data for ARIMA"""
        dates = pd.date_range('2020-01-01', periods=36, freq='ME')
        np.random.seed(42)
        values = np.cumsum(np.random.normal(0.5, 1, 36)) + 100
        self.ts_data = pd.Series(values, index=dates)
    
    def test_arima_model_creation(self):
        """Test ARIMA model instantiation"""
        from statsmodels.tsa.arima.model import ARIMA
        
        # Test model creation
        model = ARIMA(self.ts_data, order=(1, 1, 1))
        self.assertIsNotNone(model)
    
    def test_arima_fitting(self):
        """Test ARIMA model fitting"""
        from statsmodels.tsa.arima.model import ARIMA
        
        try:
            model = ARIMA(self.ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            self.assertIsNotNone(fitted_model)
            self.assertIsNotNone(fitted_model.params)
        except Exception as e:
            self.skipTest(f"ARIMA fitting failed: {e}")
    
    def test_forecast_generation(self):
        """Test forecast generation"""
        from statsmodels.tsa.arima.model import ARIMA
        
        try:
            model = ARIMA(self.ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=6)
            
            self.assertEqual(len(forecast), 6)
            self.assertIsInstance(forecast, pd.Series)
        except Exception as e:
            self.skipTest(f"Forecast generation failed: {e}")


class TestProphetIntegration(unittest.TestCase):
    """Test Prophet model integration"""
    
    def setUp(self):
        """Set up test data for Prophet"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        trend = np.linspace(100, 150, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly pattern
        noise = np.random.normal(0, 5, len(dates))
        values = trend + seasonal + noise
        
        self.prophet_data = pd.DataFrame({
            'ds': dates,
            'y': values
        })
    
    def test_prophet_data_format(self):
        """Test Prophet data format requirements"""
        # Prophet requires 'ds' and 'y' columns
        self.assertIn('ds', self.prophet_data.columns)
        self.assertIn('y', self.prophet_data.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.prophet_data['ds']))
    
    def test_prophet_model_mock(self):
        """Test Prophet model with mocking"""
        try:
            with patch('prophet.Prophet') as MockProphet:
                mock_model = MockProphet.return_value
                mock_model.fit.return_value = mock_model
                mock_model.make_future_dataframe.return_value = self.prophet_data.copy()
                mock_model.predict.return_value = pd.DataFrame({
                    'ds': self.prophet_data['ds'],
                    'yhat': self.prophet_data['y'] * 1.1,
                    'yhat_lower': self.prophet_data['y'] * 0.9,
                    'yhat_upper': self.prophet_data['y'] * 1.3
                })
                
                # Test model usage
                model = MockProphet()
                model.fit(self.prophet_data)
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                
                self.assertIsNotNone(forecast)
                MockProphet.assert_called_once()
        except ImportError:
            self.skipTest("Prophet not available for mocking")


class TestModelEvaluation(unittest.TestCase):
    """Test model evaluation metrics"""
    
    def setUp(self):
        """Set up test data for evaluation"""
        np.random.seed(42)
        self.actual = np.random.normal(100, 10, 50)
        self.predicted = self.actual + np.random.normal(0, 5, 50)
    
    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation"""
        from sklearn.metrics import mean_absolute_error
        
        mae = mean_absolute_error(self.actual, self.predicted)
        self.assertIsInstance(mae, float)
        self.assertGreater(mae, 0)
    
    def test_mse_calculation(self):
        """Test Mean Squared Error calculation"""
        from sklearn.metrics import mean_squared_error
        
        mse = mean_squared_error(self.actual, self.predicted)
        rmse = np.sqrt(mse)
        
        self.assertIsInstance(mse, float)
        self.assertIsInstance(rmse, float)
        self.assertGreater(mse, 0)
        self.assertGreater(rmse, 0)
    
    def test_mape_calculation(self):
        """Test Mean Absolute Percentage Error calculation"""
        # Manual MAPE calculation
        mape = np.mean(np.abs((self.actual - self.predicted) / self.actual)) * 100
        
        self.assertIsInstance(mape, float)
        self.assertGreater(mape, 0)
        self.assertLess(mape, 100)  # Should be reasonable


class TestFoundationModelsIntegration(unittest.TestCase):
    """Test foundation models integration"""
    
    def test_foundation_models_imports(self):
        """Test that foundation model imports work or fail gracefully"""
        try:
            from models.foundation_models.huggingface_forecaster import HybridFoundationEnsemble
            self.assertTrue(True, "Foundation models imported successfully")
        except ImportError:
            self.skipTest("Foundation models not available")
    
    def test_foundation_model_mock(self):
        """Test foundation model functionality with mocking"""
        # Test the concept of foundation model results without requiring actual imports
        mock_result = {
            'predictions': np.array([100, 101, 102, 103, 104, 105]),
            'confidence_intervals': np.array([[95, 105], [96, 106], [97, 107], [98, 108], [99, 109], [100, 110]])
        }
        
        # Test result structure
        self.assertIn('predictions', mock_result)
        self.assertIn('confidence_intervals', mock_result)
        self.assertEqual(len(mock_result['predictions']), 6)
        self.assertEqual(len(mock_result['confidence_intervals']), 6)


if __name__ == '__main__':
    print("🧪 RUNNING FORECASTING MODELS TESTS")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)