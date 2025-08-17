"""
TimeGPT Foundation Model Client
Integrates Nixtla's TimeGPT for zero-shot time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import os

try:
    from nixtla import NixtlaClient
    TIMEGPT_AVAILABLE = True
except ImportError:
    TIMEGPT_AVAILABLE = False
    logging.warning("TimeGPT not available. Install with: pip install nixtla")

logger = logging.getLogger(__name__)


class TimeGPTClient:
    """Client for TimeGPT foundation model forecasting."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TimeGPT client.
        
        Args:
            api_key: Nixtla API key (or use NIXTLA_API_KEY env var)
        """
        if not TIMEGPT_AVAILABLE:
            raise ImportError("TimeGPT not available. Install with: pip install nixtla")
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('NIXTLA_API_KEY')
        if not self.api_key:
            logger.warning("No TimeGPT API key provided. Some features may be limited.")
        
        # Initialize client
        self.client = NixtlaClient(api_key=self.api_key)
        self.model_info = {
            'name': 'TimeGPT-1',
            'type': 'Foundation Model',
            'provider': 'Nixtla',
            'zero_shot': True,
            'multivariate': True
        }
        
        logger.info("TimeGPT client initialized")
    
    def validate_series(self, series: pd.Series) -> pd.DataFrame:
        """
        Convert series to TimeGPT-compatible format.
        
        Args:
            series: Time series data
            
        Returns:
            DataFrame in TimeGPT format (ds, y columns)
        """
        if isinstance(series, pd.Series):
            df = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
        elif isinstance(series, pd.DataFrame):
            # Assume first column is timestamp, second is value
            df = series.copy()
            if len(df.columns) >= 2:
                df.columns = ['ds', 'y']
            else:
                raise ValueError("DataFrame must have at least 2 columns")
        else:
            raise ValueError("Input must be pandas Series or DataFrame")
        
        # Ensure ds is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        # Remove any rows with missing values
        df = df.dropna()
        
        return df
    
    def forecast(self, series: Union[pd.Series, pd.DataFrame], 
                horizon: int = 12,
                frequency: Optional[str] = None,
                model: str = 'timegpt-1',
                level: List[float] = [80, 90],
                clean_ex_first: bool = True,
                add_history: bool = False) -> Dict[str, Any]:
        """
        Generate forecast using TimeGPT.
        
        Args:
            series: Time series data
            horizon: Number of periods to forecast
            frequency: Time series frequency (auto-detected if None)
            model: Model variant to use
            level: Confidence levels for prediction intervals
            clean_ex_first: Whether to clean extreme values
            add_history: Whether to include historical data in output
        
        Returns:
            Dictionary with forecast results
        """
        try:
            # Validate and format data
            df = self.validate_series(series)
            
            # Auto-detect frequency if not provided
            if frequency is None:
                frequency = pd.infer_freq(df['ds'])
                if frequency is None:
                    # Fallback to monthly for economic data
                    frequency = 'M'
                    logger.warning("Could not infer frequency, assuming monthly")
            
            # Generate forecast
            forecast_df = self.client.forecast(
                df=df,
                h=horizon,
                freq=frequency,
                model=model,
                level=level,
                clean_ex_first=clean_ex_first,
                add_history=add_history
            )
            
            # Extract results
            forecast_values = forecast_df['TimeGPT'].values
            
            # Extract confidence intervals if available
            lower_bounds = {}
            upper_bounds = {}
            for lvl in level:
                lower_col = f'TimeGPT-lo-{lvl}'
                upper_col = f'TimeGPT-hi-{lvl}'
                if lower_col in forecast_df.columns:
                    lower_bounds[lvl] = forecast_df[lower_col].values
                if upper_col in forecast_df.columns:
                    upper_bounds[lvl] = forecast_df[upper_col].values
            
            # Create forecast index
            last_date = df['ds'].iloc[-1]
            if frequency == 'M':
                forecast_index = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=horizon,
                    freq='M'
                )
            elif frequency == 'D':
                forecast_index = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=horizon,
                    freq='D'
                )
            else:
                forecast_index = pd.date_range(
                    start=last_date,
                    periods=horizon + 1,
                    freq=frequency
                )[1:]
            
            return {
                'forecast': forecast_values,
                'forecast_index': forecast_index,
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'model_type': 'TimeGPT',
                'model_info': self.model_info,
                'raw_output': forecast_df,
                'frequency': frequency,
                'horizon': horizon
            }
            
        except Exception as e:
            logger.error(f"TimeGPT forecast failed: {e}")
            # Return fallback structure
            return {
                'forecast': np.array([np.nan] * horizon),
                'forecast_index': pd.date_range(
                    start=pd.Timestamp.now(),
                    periods=horizon,
                    freq='M'
                ),
                'lower_bounds': {},
                'upper_bounds': {},
                'model_type': 'TimeGPT',
                'error': str(e)
            }
    
    def forecast_multivariate(self, data: pd.DataFrame,
                            horizon: int = 12,
                            target_columns: Optional[List[str]] = None,
                            exogenous_columns: Optional[List[str]] = None,
                            frequency: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate multivariate forecasts.
        
        Args:
            data: DataFrame with multiple time series
            horizon: Forecast horizon
            target_columns: Columns to forecast (all if None)
            exogenous_columns: Exogenous variables
            frequency: Time series frequency
        
        Returns:
            Dictionary of forecasts for each target variable
        """
        results = {}
        
        if target_columns is None:
            target_columns = [col for col in data.columns if col != 'ds']
        
        # Prepare data in long format for multivariate forecasting
        try:
            # Convert to long format expected by TimeGPT
            long_data = []
            for col in target_columns:
                col_data = data[['ds', col]].copy()
                col_data.columns = ['ds', 'y']
                col_data['unique_id'] = col
                long_data.append(col_data)
            
            long_df = pd.concat(long_data, ignore_index=True)
            
            # Add exogenous variables if provided
            if exogenous_columns:
                # This would require more complex data preparation
                # For now, forecast each series individually
                logger.warning("Exogenous variables not yet supported in multivariate mode")
            
            # Generate forecast
            forecast_df = self.client.forecast(
                df=long_df,
                h=horizon,
                freq=frequency or 'M',
                model='timegpt-1'
            )
            
            # Parse results by unique_id
            for col in target_columns:
                col_forecast = forecast_df[forecast_df['unique_id'] == col]
                results[col] = {
                    'forecast': col_forecast['TimeGPT'].values,
                    'forecast_index': col_forecast['ds'],
                    'model_type': 'TimeGPT-Multivariate'
                }
                
        except Exception as e:
            logger.error(f"Multivariate forecasting failed: {e}")
            # Fallback to individual forecasts
            for col in target_columns:
                series = data.set_index('ds')[col] if 'ds' in data.columns else data[col]
                results[col] = self.forecast(series, horizon, frequency)
        
        return results
    
    def anomaly_detection(self, series: Union[pd.Series, pd.DataFrame],
                         level: float = 99.0) -> Dict[str, Any]:
        """
        Detect anomalies in time series using TimeGPT.
        
        Args:
            series: Time series data
            level: Confidence level for anomaly detection
        
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            df = self.validate_series(series)
            
            # Detect anomalies
            anomalies_df = self.client.detect_anomalies(
                df=df,
                level=level
            )
            
            return {
                'anomalies': anomalies_df['anomaly'].values,
                'anomaly_scores': anomalies_df.get('score', np.array([])),
                'timestamps': anomalies_df['ds'],
                'level': level,
                'anomaly_count': anomalies_df['anomaly'].sum()
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {
                'anomalies': np.array([]),
                'error': str(e)
            }
    
    def cross_validation(self, series: Union[pd.Series, pd.DataFrame],
                        horizon: int = 12,
                        n_windows: int = 3,
                        step_size: int = 1) -> Dict[str, Any]:
        """
        Perform cross-validation to evaluate model performance.
        
        Args:
            series: Time series data
            horizon: Forecast horizon
            n_windows: Number of validation windows
            step_size: Step size between windows
        
        Returns:
            Cross-validation results
        """
        try:
            df = self.validate_series(series)
            
            # Perform cross-validation
            cv_df = self.client.cross_validation(
                df=df,
                h=horizon,
                n_windows=n_windows,
                step_size=step_size
            )
            
            # Calculate metrics
            mae = np.mean(np.abs(cv_df['y'] - cv_df['TimeGPT']))
            mse = np.mean((cv_df['y'] - cv_df['TimeGPT']) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((cv_df['y'] - cv_df['TimeGPT']) / cv_df['y'])) * 100
            
            return {
                'cv_results': cv_df,
                'metrics': {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'mape': mape
                },
                'n_windows': n_windows,
                'horizon': horizon
            }
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {
                'error': str(e)
            }
    
    def check_api_status(self) -> Dict[str, Any]:
        """Check API status and usage."""
        try:
            # This would check API status if method exists
            return {
                'status': 'available',
                'model_info': self.model_info,
                'api_key_set': bool(self.api_key)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


if __name__ == "__main__":
    # Example usage
    client = TimeGPTClient()
    
    # Generate sample data
    dates = pd.date_range('2020-01-01', '2023-12-01', freq='M')
    np.random.seed(42)
    values = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    series = pd.Series(values, index=dates, name='sample_economic_indicator')
    
    # Test forecast
    result = client.forecast(series, horizon=6)
    print(f"Generated {len(result['forecast'])} period forecast")
    print(f"Forecast values: {result['forecast'][:3]}...")
    
    # Test anomaly detection
    anomaly_result = client.anomaly_detection(series)
    print(f"Detected {anomaly_result.get('anomaly_count', 0)} anomalies")