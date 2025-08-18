"""
Tests for R integration functionality
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Try to import R integration
try:
    from models.r_integration.r_mmm_models import RMMEconometricModels
    R_MODELS_AVAILABLE = True
except ImportError:
    R_MODELS_AVAILABLE = False
    RMMEconometricModels = None

@pytest.mark.r_integration
class TestRIntegrationAvailability:
    """Test R integration availability and setup"""
    
    def test_r_integration_import(self):
        """Test that R integration module can be imported"""
        if R_MODELS_AVAILABLE:
            assert RMMEconometricModels is not None
        else:
            pytest.skip("R integration not available")
    
    def test_r_environment_detection(self):
        """Test R environment detection"""
        if not R_MODELS_AVAILABLE:
            pytest.skip("R integration not available")
        
        r_models = RMMEconometricModels()
        diagnostics = r_models.get_r_diagnostics()
        
        assert isinstance(diagnostics, dict)
        assert 'r_available' in diagnostics
        assert 'packages_requested' in diagnostics
        assert 'packages_loaded' in diagnostics
        assert 'packages_missing' in diagnostics

@pytest.mark.r_integration
class TestRMMModels:
    """Test R-based MMM model implementations"""
    
    @pytest.fixture
    def r_models(self):
        """Create R models instance"""
        if not R_MODELS_AVAILABLE:
            pytest.skip("R integration not available")
        return RMMEconometricModels()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for R testing"""
        np.random.seed(42)
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=26, freq='W'),
            'tv_spend': np.random.normal(50000, 10000, 26),
            'digital_spend': np.random.normal(30000, 8000, 26),
            'radio_spend': np.random.normal(20000, 5000, 26),
            'revenue': np.random.normal(100000, 15000, 26)
        })
        
        # Create correlation
        data['revenue'] = (
            data['tv_spend'] * 0.8 +
            data['digital_spend'] * 1.2 +
            data['radio_spend'] * 0.6 +
            np.random.normal(20000, 5000, 26)
        )
        
        # Ensure positive values
        for col in data.columns:
            if col != 'date':
                data[col] = np.maximum(data[col], 0)
        
        return data
    
    def test_advanced_adstock_model(self, r_models, sample_data):
        """Test advanced adstock modeling with R"""
        spend_columns = ['tv_spend', 'digital_spend', 'radio_spend']
        
        results = r_models.advanced_adstock_model(
            data=sample_data,
            spend_columns=spend_columns,
            target_column='revenue'
        )
        
        assert isinstance(results, dict)
        assert 'model_type' in results
        assert 'adstock_parameters' in results
        
        # Check if R was actually used or fell back to Python
        if r_models.r_available:
            assert 'r_squared' in results
            assert results['model_type'] == 'r_advanced_adstock'
        else:
            assert results['model_type'] == 'python_fallback_adstock'
    
    def test_vector_autoregression_analysis(self, r_models, sample_data):
        """Test VAR analysis with R"""
        variables = ['tv_spend', 'digital_spend', 'revenue']
        
        results = r_models.vector_autoregression_analysis(
            data=sample_data,
            variables=variables,
            max_lags=3
        )
        
        assert isinstance(results, dict)
        assert 'model_type' in results
        assert 'variables' in results
        assert results['variables'] == variables
        
        # Check if R VAR was used or fell back
        if r_models.r_available and 'vars' in r_models.r_loaded_packages:
            assert results['model_type'] == 'r_vector_autoregression'
            assert 'optimal_lags' in results
        else:
            assert results['model_type'] == 'python_fallback_var'
    
    def test_bayesian_mmm_model(self, r_models, sample_data):
        """Test Bayesian MMM with R"""
        spend_columns = ['tv_spend', 'digital_spend']
        
        results = r_models.bayesian_mmm_model(
            data=sample_data,
            spend_columns=spend_columns,
            target_column='revenue'
        )
        
        assert isinstance(results, dict)
        assert 'model_type' in results
        assert 'features_used' in results
        assert 'uncertainty_quantification' in results
        
        # Should always have uncertainty quantification
        assert results['uncertainty_quantification'] is True
        
        # Check model type based on R availability
        if r_models.r_available:
            assert results['model_type'] == 'r_bayesian_mmm'
        else:
            assert results['model_type'] == 'python_fallback_bayesian'
    
    def test_adstock_parameters_customization(self, r_models, sample_data):
        """Test custom adstock parameters"""
        spend_columns = ['tv_spend', 'digital_spend']
        custom_params = {'tv_spend': 0.7, 'digital_spend': 0.3}
        
        results = r_models.advanced_adstock_model(
            data=sample_data,
            spend_columns=spend_columns,
            target_column='revenue',
            adstock_params=custom_params
        )
        
        assert isinstance(results, dict)
        assert 'adstock_parameters' in results
        
        # Should use custom parameters
        for channel, param in custom_params.items():
            assert results['adstock_parameters'][channel] == param

@pytest.mark.r_integration
class TestFallbackMechanisms:
    """Test Python fallback mechanisms when R is unavailable"""
    
    def test_python_fallback_adstock(self):
        """Test Python fallback for adstock modeling"""
        if not R_MODELS_AVAILABLE:
            pytest.skip("R integration not available")
        
        # Force R unavailable
        r_models = RMMEconometricModels()
        r_models.r_available = False
        
        # Create test data
        data = pd.DataFrame({
            'tv_spend': [10000, 15000, 20000],
            'digital_spend': [8000, 12000, 16000],
            'revenue': [50000, 75000, 100000]
        })
        
        results = r_models.advanced_adstock_model(
            data=data,
            spend_columns=['tv_spend', 'digital_spend'],
            target_column='revenue'
        )
        
        # Should fall back to Python
        assert results['model_type'] == 'python_fallback_adstock'
        assert 'fallback_reason' in results
        assert results['fallback_reason'] == 'R not available'
    
    def test_python_fallback_var(self):
        """Test Python fallback for VAR analysis"""
        if not R_MODELS_AVAILABLE:
            pytest.skip("R integration not available")
        
        r_models = RMMEconometricModels()
        r_models.r_available = False
        
        data = pd.DataFrame({
            'tv_spend': [10000, 15000, 20000, 25000],
            'digital_spend': [8000, 12000, 16000, 20000],
            'revenue': [50000, 75000, 100000, 125000]
        })
        
        results = r_models.vector_autoregression_analysis(
            data=data,
            variables=['tv_spend', 'digital_spend', 'revenue']
        )
        
        assert results['model_type'] == 'python_fallback_var'
        assert 'correlation_matrix' in results
    
    def test_python_fallback_bayesian(self):
        """Test Python fallback for Bayesian modeling"""
        if not R_MODELS_AVAILABLE:
            pytest.skip("R integration not available")
        
        r_models = RMMEconometricModels()
        r_models.r_available = False
        
        data = pd.DataFrame({
            'tv_spend': [10000, 15000, 20000, 25000, 30000],
            'digital_spend': [8000, 12000, 16000, 20000, 24000],
            'revenue': [50000, 75000, 100000, 125000, 150000]
        })
        
        results = r_models.bayesian_mmm_model(
            data=data,
            spend_columns=['tv_spend', 'digital_spend'],
            target_column='revenue'
        )
        
        assert results['model_type'] == 'python_fallback_bayesian'
        assert 'alpha_' in results
        assert 'lambda_' in results

@pytest.mark.r_integration
class TestRPackageRequirements:
    """Test R package requirements and loading"""
    
    def test_required_packages_list(self):
        """Test that required R packages are properly specified"""
        if not R_MODELS_AVAILABLE:
            pytest.skip("R integration not available")
        
        r_models = RMMEconometricModels()
        
        expected_packages = [
            'forecast', 'vars', 'urca', 'tseries',
            'lmtest', 'car', 'mgcv', 'MASS'
        ]
        
        for package in expected_packages:
            assert package in r_models.r_packages
    
    def test_package_loading_status(self):
        """Test R package loading status reporting"""
        if not R_MODELS_AVAILABLE:
            pytest.skip("R integration not available")
        
        r_models = RMMEconometricModels()
        diagnostics = r_models.get_r_diagnostics()
        
        # Should report which packages loaded and which didn't
        assert isinstance(diagnostics['packages_loaded'], list)
        assert isinstance(diagnostics['packages_missing'], list)
        
        # Total should equal requested
        total_accounted = len(diagnostics['packages_loaded']) + len(diagnostics['packages_missing'])
        assert total_accounted == len(r_models.r_packages)
    
    def test_custom_package_list(self):
        """Test initialization with custom R package list"""
        if not R_MODELS_AVAILABLE:
            pytest.skip("R integration not available")
        
        custom_packages = ['MASS', 'forecast']
        r_models = RMMEconometricModels(r_packages=custom_packages)
        
        assert r_models.r_packages == custom_packages

@pytest.mark.r_integration 
class TestErrorHandling:
    """Test error handling in R integration"""
    
    def test_missing_r_installation(self):
        """Test behavior when R is not installed"""
        if not R_MODELS_AVAILABLE:
            pytest.skip("R integration not available")
        
        # Mock rpy2 import failure
        with patch('models.r_integration.r_mmm_models.R_AVAILABLE', False):
            r_models = RMMEconometricModels()
            
            assert r_models.r_available is False
            
            # All methods should fall back gracefully
            data = pd.DataFrame({
                'tv_spend': [10000, 15000],
                'revenue': [50000, 75000]
            })
            
            result = r_models.advanced_adstock_model(
                data=data,
                spend_columns=['tv_spend'],
                target_column='revenue'
            )
            
            assert 'fallback' in result['model_type']
    
    def test_empty_data_handling(self):
        """Test handling of empty or invalid data"""
        if not R_MODELS_AVAILABLE:
            pytest.skip("R integration not available")
        
        r_models = RMMEconometricModels()
        
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        
        # Should handle gracefully (may use fallback)
        try:
            result = r_models.advanced_adstock_model(
                data=empty_data,
                spend_columns=[],
                target_column='revenue'
            )
            # If it returns, should be a fallback
            assert 'fallback' in result.get('model_type', '')
        except (ValueError, KeyError):
            # Or it should raise an appropriate error
            pass
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns"""
        if not R_MODELS_AVAILABLE:
            pytest.skip("R integration not available")
        
        r_models = RMMEconometricModels()
        
        data = pd.DataFrame({
            'other_column': [1, 2, 3]
        })
        
        # Should handle missing columns gracefully
        try:
            result = r_models.advanced_adstock_model(
                data=data,
                spend_columns=['missing_column'],
                target_column='missing_target'
            )
            # If it returns, should indicate the issue
            assert isinstance(result, dict)
        except (ValueError, KeyError):
            # Or raise appropriate error
            pass

@pytest.mark.r_integration
@pytest.mark.integration
class TestRIntegrationWorkflow:
    """Test complete R integration workflow"""
    
    def test_complete_r_workflow(self):
        """Test complete workflow using R integration"""
        if not R_MODELS_AVAILABLE:
            pytest.skip("R integration not available")
        
        # Generate synthetic data
        np.random.seed(42)
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=52, freq='W'),
            'tv_spend': np.random.normal(50000, 10000, 52),
            'digital_spend': np.random.normal(30000, 8000, 52),
            'radio_spend': np.random.normal(20000, 5000, 52),
            'revenue': np.random.normal(100000, 15000, 52)
        })
        
        # Create realistic correlations
        data['revenue'] = (
            data['tv_spend'] * 0.8 +
            data['digital_spend'] * 1.2 +
            data['radio_spend'] * 0.6 +
            np.random.normal(20000, 5000, 52)
        )
        
        # Ensure positive values
        for col in data.columns:
            if col != 'date':
                data[col] = np.maximum(data[col], 0)
        
        r_models = RMMEconometricModels()
        spend_columns = ['tv_spend', 'digital_spend', 'radio_spend']
        
        print(f"\n[R-WORKFLOW] R available: {r_models.r_available}")
        
        # Test adstock modeling
        adstock_results = r_models.advanced_adstock_model(
            data=data,
            spend_columns=spend_columns,
            target_column='revenue'
        )
        
        assert isinstance(adstock_results, dict)
        print(f"[R-WORKFLOW] Adstock model: {adstock_results['model_type']}")
        
        # Test VAR analysis
        var_results = r_models.vector_autoregression_analysis(
            data=data,
            variables=spend_columns + ['revenue'],
            max_lags=4
        )
        
        assert isinstance(var_results, dict)
        print(f"[R-WORKFLOW] VAR analysis: {var_results['model_type']}")
        
        # Test Bayesian modeling
        bayesian_results = r_models.bayesian_mmm_model(
            data=data,
            spend_columns=spend_columns,
            target_column='revenue'
        )
        
        assert isinstance(bayesian_results, dict)
        print(f"[R-WORKFLOW] Bayesian model: {bayesian_results['model_type']}")
        
        # Get diagnostics
        diagnostics = r_models.get_r_diagnostics()
        print(f"[R-WORKFLOW] Packages loaded: {len(diagnostics['packages_loaded'])}")
        print(f"[R-WORKFLOW] Packages missing: {len(diagnostics['packages_missing'])}")
        
        return {
            'adstock_results': adstock_results,
            'var_results': var_results,
            'bayesian_results': bayesian_results,
            'diagnostics': diagnostics
        }

if __name__ == "__main__":
    # Run R integration tests
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        if not R_MODELS_AVAILABLE:
            print("❌ R integration not available - install rpy2 and R packages")
            sys.exit(1)
        
        test_class = TestRIntegrationWorkflow()
        
        try:
            results = test_class.test_complete_r_workflow()
            print("\n✅ R integration workflow test completed")
            
        except Exception as e:
            print(f"\n❌ R integration test failed: {e}")
            sys.exit(1)