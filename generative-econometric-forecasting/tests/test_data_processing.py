#!/usr/bin/env python3
"""
🧪 Data Processing Unit Tests
Tests data fetching, processing, and validation functionality
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data.fred_client import FredDataClient
    FRED_CLIENT_AVAILABLE = True
except ImportError:
    FRED_CLIENT_AVAILABLE = False


class TestFredDataClient(unittest.TestCase):
    """Test FRED API client functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_fred_data = pd.Series(
            [100.0, 101.5, 99.8, 102.1, 103.0],
            index=pd.date_range('2023-01-01', periods=5, freq='ME'),
            name='GDP'
        )
    
    @unittest.skipIf(not FRED_CLIENT_AVAILABLE, "FRED client not available")
    def test_fred_client_initialization(self):
        """Test FRED client initialization"""
        client = FredDataClient(api_key='test_key')
        self.assertIsInstance(client, FredDataClient)
        self.assertEqual(client.api_key, 'test_key')
    
    @unittest.skipIf(not FRED_CLIENT_AVAILABLE, "FRED client not available")
    def test_fred_client_without_api_key(self):
        """Test FRED client initialization without API key"""
        client = FredDataClient()
        self.assertIsInstance(client, FredDataClient)
        # Should use environment variable or None
        self.assertTrue(client.api_key is None or isinstance(client.api_key, str))
    
    @unittest.skipIf(not FRED_CLIENT_AVAILABLE, "FRED client not available")
    @patch('data.fred_client.Fred')
    def test_fred_api_mock(self, mock_fred):
        """Test FRED API with mocking"""
        # Mock FRED API response
        mock_instance = mock_fred.return_value
        mock_instance.get_series.return_value = self.sample_fred_data
        
        client = FredDataClient(api_key='test_key')
        client.fred = mock_instance
        
        # Test data retrieval
        data = client.fred.get_series('GDP')
        
        self.assertIsInstance(data, pd.Series)
        self.assertEqual(len(data), 5)
        mock_instance.get_series.assert_called_once_with('GDP')
    
    def test_data_validation(self):
        """Test data validation functionality"""
        # Test valid data
        valid_data = self.sample_fred_data
        self.assertIsInstance(valid_data, pd.Series)
        self.assertTrue(isinstance(valid_data.index, pd.DatetimeIndex))
        
        # Test data with missing values
        data_with_gaps = valid_data.copy()
        data_with_gaps.iloc[2] = np.nan
        self.assertTrue(data_with_gaps.isnull().any())
        
        # Test data cleaning
        cleaned_data = data_with_gaps.fillna(method='ffill')
        self.assertFalse(cleaned_data.isnull().any())


class TestDataTransformation(unittest.TestCase):
    """Test data transformation and preprocessing"""
    
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range('2020-01-01', periods=24, freq='ME')
        np.random.seed(42)
        self.raw_data = pd.Series(
            100 + np.cumsum(np.random.normal(0.5, 2, 24)),
            index=dates,
            name='economic_indicator'
        )
    
    def test_log_transformation(self):
        """Test logarithmic transformation"""
        log_data = np.log(self.raw_data)
        
        self.assertEqual(len(log_data), len(self.raw_data))
        self.assertFalse(log_data.isnull().any())
        self.assertTrue(all(log_data < self.raw_data))  # Log values should be smaller
    
    def test_difference_transformation(self):
        """Test differencing transformation"""
        diff_data = self.raw_data.diff()
        
        self.assertEqual(len(diff_data), len(self.raw_data))
        self.assertTrue(diff_data.iloc[0] == diff_data.iloc[0] or pd.isna(diff_data.iloc[0]))  # First value NaN
        
        # Remove NaN and test
        diff_clean = diff_data.dropna()
        self.assertEqual(len(diff_clean), len(self.raw_data) - 1)
    
    def test_seasonal_difference(self):
        """Test seasonal differencing"""
        seasonal_diff = self.raw_data.diff(12)  # Annual seasonality
        
        self.assertEqual(len(seasonal_diff), len(self.raw_data))
        # First 12 values should be NaN
        self.assertTrue(seasonal_diff.iloc[:12].isnull().all())
        
        # Remaining values should not be NaN
        self.assertFalse(seasonal_diff.iloc[12:].isnull().any())
    
    def test_standardization(self):
        """Test data standardization"""
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        standardized = scaler.fit_transform(self.raw_data.values.reshape(-1, 1)).flatten()
        
        # Check standardization properties
        self.assertAlmostEqual(np.mean(standardized), 0, places=10)
        self.assertAlmostEqual(np.std(standardized), 1, places=10)
    
    def test_outlier_detection(self):
        """Test outlier detection"""
        # Add outliers to data
        data_with_outliers = self.raw_data.copy()
        data_with_outliers.iloc[10] = 1000  # Clear outlier
        
        # Simple outlier detection using IQR
        Q1 = data_with_outliers.quantile(0.25)
        Q3 = data_with_outliers.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data_with_outliers[(data_with_outliers < lower_bound) | 
                                    (data_with_outliers > upper_bound)]
        
        self.assertGreater(len(outliers), 0)  # Should detect the outlier
        self.assertIn(1000, outliers.values)  # Should include our injected outlier


class TestMultivariateData(unittest.TestCase):
    """Test multivariate data handling"""
    
    def setUp(self):
        """Set up multivariate test data"""
        dates = pd.date_range('2020-01-01', periods=36, freq='ME')
        np.random.seed(42)
        
        # Create correlated economic indicators
        gdp = 100 + np.cumsum(np.random.normal(0.2, 1, 36))
        unemployment = 5 + np.cumsum(np.random.normal(-0.05, 0.2, 36))
        inflation = 2 + np.cumsum(np.random.normal(0.01, 0.1, 36))
        
        self.multivariate_data = pd.DataFrame({
            'GDP': gdp,
            'Unemployment': unemployment,
            'Inflation': inflation
        }, index=dates)
    
    def test_multivariate_data_structure(self):
        """Test multivariate data structure"""
        self.assertIsInstance(self.multivariate_data, pd.DataFrame)
        self.assertEqual(self.multivariate_data.shape[1], 3)  # 3 variables
        self.assertEqual(len(self.multivariate_data), 36)     # 36 time periods
        
        # Check column names
        expected_columns = ['GDP', 'Unemployment', 'Inflation']
        self.assertListEqual(list(self.multivariate_data.columns), expected_columns)
    
    def test_correlation_analysis(self):
        """Test correlation analysis"""
        correlation_matrix = self.multivariate_data.corr()
        
        self.assertEqual(correlation_matrix.shape, (3, 3))
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(correlation_matrix), [1.0, 1.0, 1.0])
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(correlation_matrix.values, 
                                           correlation_matrix.T.values)
    
    def test_missing_data_handling_multivariate(self):
        """Test handling missing data in multivariate series"""
        # Inject missing values using valid dates from the index
        data_with_missing = self.multivariate_data.copy()
        valid_dates = data_with_missing.index
        data_with_missing.loc[valid_dates[6], 'GDP'] = np.nan  # 6th month
        data_with_missing.loc[valid_dates[15], 'Unemployment'] = np.nan  # 15th month
        
        # Test detection
        missing_by_column = data_with_missing.isnull().sum()
        self.assertEqual(missing_by_column['GDP'], 1)
        self.assertEqual(missing_by_column['Unemployment'], 1)
        self.assertEqual(missing_by_column['Inflation'], 0)
        
        # Test forward fill
        filled_data = data_with_missing.fillna(method='ffill')
        self.assertFalse(filled_data.isnull().any().any())


class TestEconomicIndicators(unittest.TestCase):
    """Test economic indicator specific processing"""
    
    def test_gdp_processing(self):
        """Test GDP data processing"""
        # Quarterly GDP data
        dates = pd.date_range('2020-01-01', periods=12, freq='Q')
        gdp_data = pd.Series([20000, 20500, 19800, 21000, 21200, 21500, 
                             21800, 22000, 22200, 22500, 22800, 23000], 
                            index=dates, name='GDP')
        
        # Calculate QoQ growth rate
        qoq_growth = gdp_data.pct_change() * 100
        
        self.assertEqual(len(qoq_growth), len(gdp_data))
        self.assertTrue(pd.isna(qoq_growth.iloc[0]))  # First value should be NaN
        
        # Calculate YoY growth rate
        yoy_growth = gdp_data.pct_change(4) * 100  # 4 quarters = 1 year
        
        self.assertEqual(len(yoy_growth), len(gdp_data))
        self.assertTrue(yoy_growth.iloc[:4].isnull().all())  # First 4 values should be NaN
    
    def test_unemployment_rate_processing(self):
        """Test unemployment rate data processing"""
        dates = pd.date_range('2020-01-01', periods=24, freq='ME')
        unemployment = pd.Series(np.random.uniform(3, 8, 24), index=dates, name='Unemployment')
        
        # Test moving average
        ma_3 = unemployment.rolling(window=3).mean()
        
        self.assertEqual(len(ma_3), len(unemployment))
        self.assertTrue(ma_3.iloc[:2].isnull().all())  # First 2 values should be NaN
        self.assertFalse(ma_3.iloc[2:].isnull().any())  # Rest should not be NaN
    
    def test_inflation_rate_processing(self):
        """Test inflation rate data processing"""
        dates = pd.date_range('2020-01-01', periods=24, freq='ME')
        cpi = pd.Series(200 + np.cumsum(np.random.normal(0.2, 0.5, 24)), 
                       index=dates, name='CPI')
        
        # Calculate monthly inflation
        monthly_inflation = cpi.pct_change() * 100
        
        # Calculate annualized inflation
        annual_inflation = cpi.pct_change(12) * 100
        
        self.assertEqual(len(monthly_inflation), len(cpi))
        self.assertEqual(len(annual_inflation), len(cpi))
        
        # Test that inflation calculations are reasonable
        self.assertTrue(monthly_inflation.iloc[1:].abs().max() < 10)  # Monthly inflation < 10%


class TestDataQuality(unittest.TestCase):
    """Test data quality checks"""
    
    def setUp(self):
        """Set up test data with various quality issues"""
        dates = pd.date_range('2020-01-01', periods=50, freq='ME')
        np.random.seed(42)
        
        # Create data with various issues
        values = 100 + np.cumsum(np.random.normal(0.5, 2, 50))
        values[10] = np.nan  # Missing value
        values[25] = 1000    # Outlier
        values[30] = -50     # Negative value (might be invalid for some indicators)
        
        self.problematic_data = pd.Series(values, index=dates, name='test_indicator')
    
    def test_missing_value_detection(self):
        """Test missing value detection"""
        missing_count = self.problematic_data.isnull().sum()
        missing_percentage = (missing_count / len(self.problematic_data)) * 100
        
        self.assertEqual(missing_count, 1)
        self.assertAlmostEqual(missing_percentage, 2.0, places=1)  # 1/50 = 2%
    
    def test_outlier_detection_zscore(self):
        """Test outlier detection using Z-score"""
        from scipy import stats
        
        # Calculate Z-scores (excluding NaN)
        clean_data = self.problematic_data.dropna()
        z_scores = np.abs(stats.zscore(clean_data))
        outliers = clean_data[z_scores > 3]
        
        self.assertGreater(len(outliers), 0)  # Should detect outliers
    
    def test_data_completeness(self):
        """Test data completeness checks"""
        # Check for gaps in time series
        expected_freq = pd.infer_freq(self.problematic_data.index)
        self.assertEqual(expected_freq, 'ME')  # Should be monthly
        
        # Check date range completeness
        date_range = pd.date_range(start=self.problematic_data.index.min(),
                                 end=self.problematic_data.index.max(),
                                 freq='M')
        
        missing_dates = date_range.difference(self.problematic_data.index)
        self.assertEqual(len(missing_dates), 0)  # No missing dates in our test data
    
    def test_value_range_validation(self):
        """Test value range validation"""
        # Test for negative values (might be invalid for some indicators)
        negative_values = self.problematic_data[self.problematic_data < 0]
        self.assertGreater(len(negative_values), 0)  # Should find negative values
        
        # Test for extreme values (excluding NaN values)
        clean_data = self.problematic_data.dropna()
        median_val = clean_data.median()
        mad = np.median(np.abs(clean_data - median_val))  # Median Absolute Deviation
        
        # Values more than 2 MADs from median could be considered extreme
        extreme_threshold = 2 * mad
        extreme_values = clean_data[
            np.abs(clean_data - median_val) > extreme_threshold
        ]
        
        # Should find at least the outlier we injected (value 1000)
        self.assertGreater(len(extreme_values), 0)  # Should find extreme values
        self.assertIn(1000, extreme_values.values)  # Should include our injected outlier


if __name__ == '__main__':
    print("🧪 RUNNING DATA PROCESSING TESTS")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)