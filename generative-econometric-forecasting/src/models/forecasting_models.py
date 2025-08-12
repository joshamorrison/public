"""
Core econometric forecasting models for time series analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Prophet for trend analysis
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Install with: pip install prophet")

logger = logging.getLogger(__name__)


class EconometricForecaster:
    """Advanced econometric forecasting with multiple model types."""
    
    def __init__(self):
        self.models = {}
        self.fitted_models = {}
        self.forecasts = {}
        self.model_performance = {}
        
    def check_stationarity(self, series: pd.Series, 
                          method: str = 'adf') -> Dict[str, Any]:
        """
        Test for stationarity using ADF or KPSS tests.
        
        Args:
            series: Time series to test
            method: 'adf' for Augmented Dickey-Fuller or 'kpss' for KPSS
        
        Returns:
            Dictionary with test results
        """
        series_clean = series.dropna()
        
        if method == 'adf':
            result = adfuller(series_clean)
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05,
                'test_type': 'Augmented Dickey-Fuller'
            }
        elif method == 'kpss':
            result = kpss(series_clean)
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[3],
                'is_stationary': result[1] > 0.05,  # KPSS null is stationary
                'test_type': 'KPSS'
            }
    
    def decompose_series(self, series: pd.Series, 
                        model: str = 'additive',
                        period: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            series: Time series to decompose
            model: 'additive' or 'multiplicative'
            period: Seasonal period (auto-detected if None)
        
        Returns:
            Dictionary with decomposed components
        """
        try:
            decomposition = seasonal_decompose(
                series.dropna(), 
                model=model, 
                period=period
            )
            
            return {
                'original': series,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
        except Exception as e:
            logger.error(f"Error in decomposition: {e}")
            return {'original': series}
    
    def fit_arima(self, series: pd.Series, order: Tuple[int, int, int] = None,
                  seasonal_order: Tuple[int, int, int, int] = None,
                  auto_order: bool = True) -> Dict[str, Any]:
        """
        Fit ARIMA or SARIMA model to time series.
        
        Args:
            series: Time series data
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
            auto_order: Automatically determine best order
        
        Returns:
            Dictionary with fitted model and diagnostics
        """
        series_clean = series.dropna()
        
        if auto_order:
            # Simple grid search for best parameters
            best_aic = np.inf
            best_order = None
            best_model = None
            
            # Test different combinations
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(series_clean, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                                best_model = fitted
                        except:
                            continue
            
            if best_model is None:
                # Fallback to simple ARIMA(1,1,1)
                order = (1, 1, 1)
                model = ARIMA(series_clean, order=order)
                fitted_model = model.fit()
            else:
                fitted_model = best_model
                order = best_order
        else:
            # Use provided order
            if seasonal_order:
                model = SARIMAX(series_clean, order=order, seasonal_order=seasonal_order)
            else:
                model = ARIMA(series_clean, order=order)
            fitted_model = model.fit()
        
        # Store model
        model_key = f"arima_{series.name}"
        self.fitted_models[model_key] = fitted_model
        
        return {
            'model': fitted_model,
            'order': order,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'model_key': model_key
        }
    
    def fit_var(self, data: pd.DataFrame, maxlags: int = 5) -> Dict[str, Any]:
        """
        Fit Vector Autoregression (VAR) model for multivariate analysis.
        
        Args:
            data: DataFrame with multiple time series
            maxlags: Maximum number of lags to consider
        
        Returns:
            Dictionary with fitted VAR model
        """
        # Clean data
        data_clean = data.dropna()
        
        if len(data_clean) < 50:
            raise ValueError("Insufficient data for VAR model")
        
        # Fit VAR model
        model = VAR(data_clean)
        
        # Select optimal lag order
        lag_order = model.select_order(maxlags=maxlags)
        optimal_lags = lag_order.aic
        
        # Fit with optimal lags
        fitted_model = model.fit(optimal_lags)
        
        # Store model
        model_key = "var_multivariate"
        self.fitted_models[model_key] = fitted_model
        
        return {
            'model': fitted_model,
            'optimal_lags': optimal_lags,
            'aic': lag_order.aic,
            'model_key': model_key
        }
    
    def fit_prophet(self, series: pd.Series, 
                   yearly_seasonality: bool = True,
                   weekly_seasonality: bool = False) -> Dict[str, Any]:
        """
        Fit Prophet model for trend and seasonality analysis.
        
        Args:
            series: Time series data
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
        
        Returns:
            Dictionary with fitted Prophet model
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available")
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        }).dropna()
        
        # Fit Prophet model
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False
        )
        
        fitted_model = model.fit(df)
        
        # Store model
        model_key = f"prophet_{series.name}"
        self.fitted_models[model_key] = fitted_model
        
        return {
            'model': fitted_model,
            'model_key': model_key,
            'training_data': df
        }
    
    def generate_forecast(self, model_key: str, periods: int = 12,
                         confidence_interval: float = 0.95) -> Dict[str, Any]:
        """
        Generate forecasts from fitted model.
        
        Args:
            model_key: Key identifying the fitted model
            periods: Number of periods to forecast
            confidence_interval: Confidence level for intervals
        
        Returns:
            Dictionary with forecast results
        """
        if model_key not in self.fitted_models:
            raise ValueError(f"Model {model_key} not found")
        
        model = self.fitted_models[model_key]
        
        if 'arima' in model_key:
            # ARIMA forecast
            forecast = model.forecast(steps=periods)
            conf_int = model.get_forecast(steps=periods).conf_int()
            
            return {
                'forecast': forecast,
                'lower_bound': conf_int.iloc[:, 0],
                'upper_bound': conf_int.iloc[:, 1],
                'model_type': 'ARIMA'
            }
        
        elif 'var' in model_key:
            # VAR forecast
            forecast = model.forecast(model.y, steps=periods)
            
            return {
                'forecast': forecast,
                'model_type': 'VAR',
                'variables': model.names
            }
        
        elif 'prophet' in model_key:
            # Prophet forecast
            future_dates = model.make_future_dataframe(periods=periods, freq='M')
            forecast = model.predict(future_dates)
            
            # Extract last 'periods' rows for the actual forecast
            forecast_values = forecast.tail(periods)
            
            return {
                'forecast': forecast_values['yhat'].values,
                'lower_bound': forecast_values['yhat_lower'].values,
                'upper_bound': forecast_values['yhat_upper'].values,
                'model_type': 'Prophet',
                'full_forecast': forecast
            }
    
    def evaluate_model(self, model_key: str, test_data: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            model_key: Key identifying the fitted model
            test_data: True values for evaluation
        
        Returns:
            Dictionary with performance metrics
        """
        if model_key not in self.forecasts:
            raise ValueError(f"No forecasts found for {model_key}")
        
        forecast = self.forecasts[model_key]['forecast']
        
        # Align lengths
        min_len = min(len(forecast), len(test_data))
        forecast_aligned = forecast[:min_len]
        test_aligned = test_data.iloc[:min_len]
        
        # Calculate metrics
        mae = mean_absolute_error(test_aligned, forecast_aligned)
        mse = mean_squared_error(test_aligned, forecast_aligned)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test_aligned - forecast_aligned) / test_aligned)) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }
        
        self.model_performance[model_key] = metrics
        return metrics
    
    def generate_ensemble_forecast(self, series: pd.Series, 
                                 periods: int = 12) -> Dict[str, Any]:
        """
        Generate ensemble forecast using multiple models.
        
        Args:
            series: Time series to forecast
            periods: Number of periods to forecast
        
        Returns:
            Dictionary with ensemble forecast
        """
        forecasts = {}
        
        # Fit and forecast with ARIMA
        try:
            arima_result = self.fit_arima(series)
            arima_forecast = self.generate_forecast(arima_result['model_key'], periods)
            forecasts['arima'] = arima_forecast['forecast']
        except Exception as e:
            logger.warning(f"ARIMA failed: {e}")
        
        # Fit and forecast with Prophet if available
        if PROPHET_AVAILABLE:
            try:
                prophet_result = self.fit_prophet(series)
                prophet_forecast = self.generate_forecast(prophet_result['model_key'], periods)
                forecasts['prophet'] = prophet_forecast['forecast']
            except Exception as e:
                logger.warning(f"Prophet failed: {e}")
        
        if not forecasts:
            raise ValueError("No models successfully fitted")
        
        # Simple average ensemble
        forecast_values = np.array(list(forecasts.values()))
        ensemble_forecast = np.mean(forecast_values, axis=0)
        
        return {
            'ensemble_forecast': ensemble_forecast,
            'individual_forecasts': forecasts,
            'model_count': len(forecasts)
        }


def analyze_forecast_accuracy(forecasts: Dict[str, np.ndarray], 
                            actual: pd.Series) -> pd.DataFrame:
    """
    Compare accuracy of multiple forecasting methods.
    
    Args:
        forecasts: Dictionary of model forecasts
        actual: Actual observed values
    
    Returns:
        DataFrame with accuracy comparison
    """
    results = []
    
    for model_name, forecast in forecasts.items():
        min_len = min(len(forecast), len(actual))
        forecast_aligned = forecast[:min_len]
        actual_aligned = actual.iloc[:min_len]
        
        mae = mean_absolute_error(actual_aligned, forecast_aligned)
        mse = mean_squared_error(actual_aligned, forecast_aligned)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_aligned - forecast_aligned) / actual_aligned)) * 100
        
        results.append({
            'model': model_name,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        })
    
    return pd.DataFrame(results).sort_values('mae')


if __name__ == "__main__":
    # Example usage
    forecaster = EconometricForecaster()
    
    # Generate sample data
    dates = pd.date_range('2010-01-01', '2023-12-01', freq='M')
    np.random.seed(42)
    data = pd.Series(
        100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        index=dates,
        name='sample_series'
    )
    
    # Test stationarity
    stationarity = forecaster.check_stationarity(data)
    print(f"Stationarity test: {stationarity}")
    
    # Fit ARIMA model
    arima_result = forecaster.fit_arima(data)
    print(f"ARIMA model fitted with order: {arima_result['order']}")
    
    # Generate forecast
    forecast_result = forecaster.generate_forecast(arima_result['model_key'], periods=6)
    print(f"6-period forecast: {forecast_result['forecast']}")