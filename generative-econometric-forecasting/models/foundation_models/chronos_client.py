"""
Chronos Foundation Model Client
Integrates Amazon's Chronos transformer models for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    torch = None
    AutoConfig = None
    AutoTokenizer = None
    ChronosPipeline = None
    CHRONOS_AVAILABLE = False
    logging.warning("Chronos not available. Install with: pip install chronos-forecasting")

logger = logging.getLogger(__name__)


class ChronosClient:
    """Client for Chronos foundation model forecasting."""
    
    def __init__(self, model_size: str = "small", device: Optional[str] = None):
        """
        Initialize Chronos client.
        
        Args:
            model_size: Model size ('tiny', 'mini', 'small', 'base', 'large')
            device: Device to run model on ('cpu', 'cuda', 'mps')
        """
        if not CHRONOS_AVAILABLE:
            raise ImportError("Chronos not available. Install with: pip install chronos-forecasting")
        
        self.model_size = model_size
        if torch is not None:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        
        # Model configurations
        self.model_configs = {
            'tiny': {
                'model_id': 'amazon/chronos-t5-tiny',
                'params': '8M',
                'context_length': 512
            },
            'mini': {
                'model_id': 'amazon/chronos-t5-mini', 
                'params': '20M',
                'context_length': 512
            },
            'small': {
                'model_id': 'amazon/chronos-t5-small',
                'params': '46M', 
                'context_length': 512
            },
            'base': {
                'model_id': 'amazon/chronos-t5-base',
                'params': '200M',
                'context_length': 512
            },
            'large': {
                'model_id': 'amazon/chronos-t5-large',
                'params': '710M',
                'context_length': 512
            }
        }
        
        if model_size not in self.model_configs:
            raise ValueError(f"Model size must be one of: {list(self.model_configs.keys())}")
        
        self.config = self.model_configs[model_size]
        self.model_id = self.config['model_id']
        
        # Initialize pipeline
        self.pipeline = None
        self._initialize_pipeline()
        
        logger.info(f"Chronos {model_size} model initialized on {self.device}")
    
    def _initialize_pipeline(self):
        """Initialize the Chronos pipeline."""
        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=torch.bfloat16 if (torch is not None and self.device != "cpu") else (torch.float32 if torch is not None else None)
            )
            logger.info(f"Loaded Chronos model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to load Chronos model: {e}")
            raise
    
    def prepare_context(self, series: pd.Series, 
                       context_length: Optional[int] = None) -> Any:
        """
        Prepare time series context for Chronos model.
        
        Args:
            series: Time series data
            context_length: Length of context (uses model default if None)
        
        Returns:
            Prepared context tensor
        """
        if context_length is None:
            context_length = self.config['context_length']
        
        # Clean the series
        series_clean = series.dropna()
        
        # Take the last context_length points
        if len(series_clean) > context_length:
            context_data = series_clean.tail(context_length).values
        else:
            context_data = series_clean.values
        
        # Convert to tensor
        if torch is not None:
            context = torch.tensor(context_data, dtype=torch.float32)
        else:
            context = context_data
        
        return context
    
    def forecast(self, series: pd.Series,
                horizon: int = 12,
                num_samples: int = 100,
                temperature: float = 1.0,
                top_k: Optional[int] = 50,
                top_p: float = 1.0) -> Dict[str, Any]:
        """
        Generate forecast using Chronos model.
        
        Args:
            series: Time series data
            horizon: Number of periods to forecast
            num_samples: Number of forecast samples to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
        
        Returns:
            Dictionary with forecast results
        """
        try:
            # Prepare context
            context = self.prepare_context(series)
            
            # Generate forecast samples
            forecast_samples = self.pipeline.predict(
                context=context.unsqueeze(0),  # Add batch dimension
                prediction_length=horizon,
                num_samples=num_samples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Convert to numpy and remove batch dimension
            forecast_samples = forecast_samples.squeeze(0).numpy()
            
            # Calculate statistics
            forecast_mean = np.mean(forecast_samples, axis=0)
            forecast_median = np.median(forecast_samples, axis=0)
            forecast_std = np.std(forecast_samples, axis=0)
            
            # Calculate confidence intervals
            confidence_levels = [80, 90, 95]
            lower_bounds = {}
            upper_bounds = {}
            
            for level in confidence_levels:
                alpha = (100 - level) / 2
                lower_bounds[level] = np.percentile(forecast_samples, alpha, axis=0)
                upper_bounds[level] = np.percentile(forecast_samples, 100 - alpha, axis=0)
            
            # Create forecast index
            last_date = series.index[-1]
            if isinstance(last_date, pd.Timestamp):
                # Infer frequency
                freq = pd.infer_freq(series.index)
                if freq is None:
                    freq = 'M'  # Default to monthly for economic data
                
                forecast_index = pd.date_range(
                    start=last_date,
                    periods=horizon + 1,
                    freq=freq
                )[1:]
            else:
                # Numeric index
                forecast_index = range(len(series), len(series) + horizon)
            
            return {
                'forecast': forecast_mean,
                'forecast_median': forecast_median,
                'forecast_std': forecast_std,
                'forecast_samples': forecast_samples,
                'forecast_index': forecast_index,
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'model_type': 'Chronos',
                'model_size': self.model_size,
                'model_info': self.config,
                'num_samples': num_samples,
                'horizon': horizon
            }
            
        except Exception as e:
            logger.error(f"Chronos forecast failed: {e}")
            return {
                'forecast': np.array([np.nan] * horizon),
                'forecast_index': range(horizon),
                'model_type': 'Chronos',
                'error': str(e)
            }
    
    def forecast_multivariate(self, data: pd.DataFrame,
                            horizon: int = 12,
                            target_columns: Optional[List[str]] = None,
                            num_samples: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        Generate forecasts for multiple time series.
        
        Args:
            data: DataFrame with multiple time series
            horizon: Forecast horizon
            target_columns: Columns to forecast
            num_samples: Number of samples per forecast
        
        Returns:
            Dictionary of forecasts for each series
        """
        if target_columns is None:
            target_columns = data.columns.tolist()
        
        results = {}
        
        for col in target_columns:
            try:
                series = data[col].dropna()
                if len(series) < 10:  # Minimum length check
                    logger.warning(f"Series {col} too short for forecasting")
                    continue
                
                forecast_result = self.forecast(
                    series=series,
                    horizon=horizon,
                    num_samples=num_samples
                )
                results[col] = forecast_result
                
            except Exception as e:
                logger.error(f"Failed to forecast {col}: {e}")
                results[col] = {
                    'forecast': np.array([np.nan] * horizon),
                    'error': str(e)
                }
        
        return results
    
    def batch_forecast(self, series_dict: Dict[str, pd.Series],
                      horizon: int = 12,
                      num_samples: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        Generate forecasts for multiple series in batch.
        
        Args:
            series_dict: Dictionary of named time series
            horizon: Forecast horizon
            num_samples: Number of samples per forecast
        
        Returns:
            Dictionary of forecast results
        """
        results = {}
        
        for name, series in series_dict.items():
            logger.info(f"Forecasting {name}")
            results[name] = self.forecast(
                series=series,
                horizon=horizon,
                num_samples=num_samples
            )
        
        return results
    
    def evaluate_forecast(self, series: pd.Series,
                         forecast_horizon: int = 12,
                         test_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate forecast performance using historical data.
        
        Args:
            series: Full time series including test period
            forecast_horizon: Horizon to forecast
            test_size: Size of test set (uses forecast_horizon if None)
        
        Returns:
            Evaluation metrics
        """
        if test_size is None:
            test_size = forecast_horizon
        
        if len(series) < test_size + 20:  # Minimum training size
            raise ValueError("Series too short for evaluation")
        
        # Split data
        train_series = series.iloc[:-test_size]
        test_series = series.iloc[-test_size:]
        
        # Generate forecast
        forecast_result = self.forecast(
            series=train_series,
            horizon=min(forecast_horizon, test_size)
        )
        
        # Calculate metrics
        forecast_values = forecast_result['forecast']
        actual_values = test_series.iloc[:len(forecast_values)]
        
        mae = np.mean(np.abs(actual_values - forecast_values))
        mse = np.mean((actual_values - forecast_values) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
        
        return {
            'metrics': {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape
            },
            'forecast': forecast_values,
            'actual': actual_values.values,
            'forecast_result': forecast_result
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_id': self.model_id,
            'model_size': self.model_size,
            'parameters': self.config['params'],
            'context_length': self.config['context_length'],
            'device': self.device,
            'available': self.pipeline is not None
        }
    
    def warmup(self, dummy_series_length: int = 100):
        """Warm up the model with a dummy forecast."""
        try:
            # Create dummy series
            dummy_data = pd.Series(np.random.randn(dummy_series_length))
            
            # Run a small forecast to warm up
            self.forecast(dummy_data, horizon=1, num_samples=1)
            logger.info("Model warmed up successfully")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")


if __name__ == "__main__":
    # Example usage
    client = ChronosClient(model_size="small")
    
    # Generate sample economic data
    dates = pd.date_range('2020-01-01', '2023-12-01', freq='M')
    np.random.seed(42)
    # Simulate GDP-like growth with trend and noise
    trend = np.linspace(100, 120, len(dates))
    noise = np.random.randn(len(dates)) * 2
    seasonal = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    values = trend + seasonal + noise
    
    series = pd.Series(values, index=dates, name='gdp_index')
    
    # Test forecast
    result = client.forecast(series, horizon=6, num_samples=50)
    print(f"Generated forecast with {len(result['forecast'])} periods")
    print(f"Mean forecast: {result['forecast'][:3]}")
    print(f"Forecast std: {result['forecast_std'][:3]}")
    
    # Test evaluation
    eval_result = client.evaluate_forecast(series, forecast_horizon=6)
    print(f"MAPE: {eval_result['metrics']['mape']:.2f}%")
    print(f"RMSE: {eval_result['metrics']['rmse']:.2f}")
    
    # Model info
    info = client.get_model_info()
    print(f"Model: {info['model_id']} ({info['parameters']} parameters)")