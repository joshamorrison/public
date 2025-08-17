"""
R Statistical Models Integration
Bridges Python and R for advanced econometric modeling using rpy2
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
import os

logger = logging.getLogger(__name__)

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, Formula
    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri as numpy2ri
    
    # Activate automatic pandas conversion
    pandas2ri.activate()
    numpy2ri.activate()
    
    # Import R packages
    base = importr('base')
    stats = importr('stats')
    
    try:
        forecast = importr('forecast')
        R_FORECAST_AVAILABLE = True
    except:
        R_FORECAST_AVAILABLE = False
        logger.warning("R forecast package not available. Install with: install.packages('forecast')")
    
    try:
        vars_pkg = importr('vars')
        R_VARS_AVAILABLE = True
    except:
        R_VARS_AVAILABLE = False
        logger.warning("R vars package not available. Install with: install.packages('vars')")
    
    try:
        urca = importr('urca')
        R_URCA_AVAILABLE = True
    except:
        R_URCA_AVAILABLE = False
        logger.warning("R urca package not available. Install with: install.packages('urca')")
    
    RPY2_AVAILABLE = True
    
except ImportError:
    RPY2_AVAILABLE = False
    logger.warning("rpy2 not available. Install with: pip install rpy2")
    logger.warning("Also ensure R is installed on your system")


class RStatisticalModels:
    """
    Advanced statistical modeling using R through rpy2 interface.
    Provides access to R's comprehensive econometric packages.
    """
    
    def __init__(self):
        """Initialize R statistical models interface."""
        self.r_available = RPY2_AVAILABLE
        
        if not self.r_available:
            logger.warning("R integration not available - falling back to Python-only models")
            return
        
        # Set up R environment
        try:
            robjects.r('options(warn=-1)')  # Suppress R warnings
            logger.info("R statistical models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize R environment: {e}")
            self.r_available = False
    
    def fit_arima_r(self, data: pd.Series, order: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """
        Fit ARIMA model using R's forecast package.
        
        Args:
            data: Time series data
            order: ARIMA order (p, d, q). If None, auto.arima is used
            
        Returns:
            Dictionary with model results and forecasts
        """
        if not self.r_available or not R_FORECAST_AVAILABLE:
            raise RuntimeError("R forecast package not available")
        
        try:
            # Convert to R time series
            r_data = robjects.FloatVector(data.values)
            ts_data = stats.ts(r_data, frequency=12)  # Assuming monthly data
            
            if order is None:
                # Use auto.arima for automatic model selection
                model = forecast.auto_arima(ts_data)
            else:
                # Fit specified ARIMA model
                model = forecast.Arima(ts_data, order=robjects.IntVector(order))
            
            # Generate forecasts
            forecast_result = forecast.forecast(model, h=12)
            
            # Extract results
            results = {
                'model': model,
                'fitted_values': np.array(robjects.r['fitted'](model)),
                'residuals': np.array(robjects.r['residuals'](model)),
                'forecast_mean': np.array(forecast_result.rx2('mean')),
                'forecast_lower': np.array(forecast_result.rx2('lower')),
                'forecast_upper': np.array(forecast_result.rx2('upper')),
                'aic': robjects.r['AIC'](model)[0],
                'bic': robjects.r['BIC'](model)[0]
            }
            
            logger.info(f"R ARIMA model fitted successfully - AIC: {results['aic']:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"R ARIMA modeling failed: {e}")
            raise
    
    def fit_var_model_r(self, data: pd.DataFrame, lag_order: Optional[int] = None) -> Dict[str, Any]:
        """
        Fit Vector Autoregression (VAR) model using R's vars package.
        
        Args:
            data: Multivariate time series data
            lag_order: Number of lags. If None, optimal lag is selected
            
        Returns:
            Dictionary with VAR model results
        """
        if not self.r_available or not R_VARS_AVAILABLE:
            raise RuntimeError("R vars package not available")
        
        try:
            # Convert to R data frame
            r_data = pandas2ri.py2rpy(data)
            
            if lag_order is None:
                # Select optimal lag order
                lag_select = vars_pkg.VARselect(r_data, lag_max=8)
                lag_order = int(robjects.r['as.numeric'](lag_select.rx2('selection').rx2('AIC'))[0])
            
            # Fit VAR model
            var_model = vars_pkg.VAR(r_data, p=lag_order)
            
            # Generate forecasts
            var_forecast = robjects.r['predict'](var_model, n_ahead=12)
            
            results = {
                'model': var_model,
                'lag_order': lag_order,
                'forecasts': pandas2ri.rpy2py(var_forecast.rx2('fcst')),
                'summary': robjects.r['summary'](var_model)
            }
            
            logger.info(f"R VAR model fitted successfully - Lag order: {lag_order}")
            return results
            
        except Exception as e:
            logger.error(f"R VAR modeling failed: {e}")
            raise
    
    def cointegration_test_r(self, data: pd.DataFrame, test_type: str = 'johansen') -> Dict[str, Any]:
        """
        Perform cointegration tests using R's urca package.
        
        Args:
            data: Multivariate time series data
            test_type: Type of test ('johansen' or 'engle_granger')
            
        Returns:
            Dictionary with test results
        """
        if not self.r_available or not R_URCA_AVAILABLE:
            raise RuntimeError("R urca package not available")
        
        try:
            r_data = pandas2ri.py2rpy(data)
            
            if test_type == 'johansen':
                # Johansen cointegration test
                test_result = urca.ca_jo(r_data, type='trace', ecdet='const')
                
                results = {
                    'test_type': 'johansen',
                    'test_statistic': np.array(robjects.r['teststat'](test_result)),
                    'critical_values': np.array(robjects.r['cval'](test_result)),
                    'eigenvalues': np.array(robjects.r['lambda'](test_result)),
                    'summary': robjects.r['summary'](test_result)
                }
                
            elif test_type == 'engle_granger':
                # Engle-Granger test (for two variables)
                if data.shape[1] != 2:
                    raise ValueError("Engle-Granger test requires exactly 2 variables")
                
                test_result = urca.ca_eg(r_data)
                
                results = {
                    'test_type': 'engle_granger',
                    'test_statistic': robjects.r['teststat'](test_result),
                    'critical_values': robjects.r['cval'](test_result),
                    'summary': robjects.r['summary'](test_result)
                }
            
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            logger.info(f"R cointegration test ({test_type}) completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"R cointegration test failed: {e}")
            raise
    
    def garch_model_r(self, data: pd.Series) -> Dict[str, Any]:
        """
        Fit GARCH model for volatility modeling using R.
        
        Args:
            data: Time series data (returns)
            
        Returns:
            Dictionary with GARCH model results
        """
        if not self.r_available:
            raise RuntimeError("R not available")
        
        try:
            # Install rugarch if not available
            robjects.r('''
            if (!require(rugarch)) {
                install.packages("rugarch", repos="https://cran.r-project.org")
                library(rugarch)
            }
            ''')
            
            # Convert data to R
            r_data = robjects.FloatVector(data.values)
            
            # Specify GARCH model
            robjects.r('''
            spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                              mean.model = list(armaOrder = c(1, 1)))
            ''')
            
            # Fit model
            robjects.r['assign']('data', r_data)
            robjects.r('fit <- ugarchfit(spec, data)')
            
            # Extract results
            results = {
                'fitted_model': robjects.r('fit'),
                'volatility': np.array(robjects.r('sigma(fit)')),
                'fitted_values': np.array(robjects.r('fitted(fit)')),
                'residuals': np.array(robjects.r('residuals(fit)'))
            }
            
            logger.info("R GARCH model fitted successfully")
            return results
            
        except Exception as e:
            logger.error(f"R GARCH modeling failed: {e}")
            raise
    
    def structural_break_test_r(self, data: pd.Series) -> Dict[str, Any]:
        """
        Test for structural breaks using R's strucchange package.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with structural break test results
        """
        if not self.r_available:
            raise RuntimeError("R not available")
        
        try:
            # Install strucchange if not available
            robjects.r('''
            if (!require(strucchange)) {
                install.packages("strucchange", repos="https://cran.r-project.org")
                library(strucchange)
            }
            ''')
            
            # Convert data to R
            r_data = robjects.FloatVector(data.values)
            
            # Create time index
            n = len(data)
            time_index = robjects.IntVector(range(1, n + 1))
            
            # Perform Chow test for structural breaks
            robjects.r['assign']('y', r_data)
            robjects.r['assign']('time', time_index)
            robjects.r('df <- data.frame(y = y, time = time)')
            robjects.r('bp_test <- breakpoints(y ~ time, data = df)')
            
            results = {
                'breakpoints': robjects.r('bp_test$breakpoints'),
                'rss': robjects.r('bp_test$RSS'),
                'summary': robjects.r('summary(bp_test)')
            }
            
            logger.info("R structural break test completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"R structural break test failed: {e}")
            raise
    
    def get_r_summary(self) -> Dict[str, Any]:
        """Get summary of R environment and available packages."""
        if not self.r_available:
            return {'status': 'R not available'}
        
        try:
            r_version = robjects.r('R.version.string')[0]
            installed_packages = list(robjects.r('installed.packages()[,1]'))
            
            return {
                'status': 'R available',
                'r_version': r_version,
                'forecast_available': R_FORECAST_AVAILABLE,
                'vars_available': R_VARS_AVAILABLE,
                'urca_available': R_URCA_AVAILABLE,
                'total_packages': len(installed_packages),
                'key_packages': [pkg for pkg in ['forecast', 'vars', 'urca', 'rugarch', 'strucchange'] 
                               if pkg in installed_packages]
            }
            
        except Exception as e:
            logger.error(f"Failed to get R summary: {e}")
            return {'status': 'R available but error occurred', 'error': str(e)}


# Convenience functions for easy access
def fit_arima_with_r(data: pd.Series, order: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
    """Convenience function to fit ARIMA model with R."""
    r_models = RStatisticalModels()
    return r_models.fit_arima_r(data, order)

def fit_var_with_r(data: pd.DataFrame, lag_order: Optional[int] = None) -> Dict[str, Any]:
    """Convenience function to fit VAR model with R."""
    r_models = RStatisticalModels()
    return r_models.fit_var_model_r(data, lag_order)

def test_cointegration_with_r(data: pd.DataFrame, test_type: str = 'johansen') -> Dict[str, Any]:
    """Convenience function to test cointegration with R."""
    r_models = RStatisticalModels()
    return r_models.cointegration_test_r(data, test_type)