"""
Hugging Face Foundation Models for Time Series Forecasting
Free alternative to paid services with Chronos and other transformer models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Chronos models - free Hugging Face transformer models
try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    logging.warning("Chronos not available. Install with: pip install chronos-forecasting")

# Standard transformers for time series
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers torch")

logger = logging.getLogger(__name__)


class HuggingFaceForecaster:
    """Free Hugging Face models for time series forecasting."""
    
    def __init__(self, model_name: str = "amazon/chronos-t5-small"):
        """
        Initialize Hugging Face forecaster.
        
        Args:
            model_name: Name of the model to use
                - amazon/chronos-t5-small (fast, good accuracy)
                - amazon/chronos-t5-base (better accuracy)  
                - amazon/chronos-t5-large (best accuracy, slower)
        """
        self.model_name = model_name
        self.pipeline = None
        self.model_info = {
            'name': model_name,
            'type': 'Foundation Model',
            'provider': 'Hugging Face',
            'zero_shot': True,
            'cost': 'Free'
        }
        
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model pipeline."""
        try:
            if CHRONOS_AVAILABLE and "chronos" in self.model_name.lower():
                self.pipeline = ChronosPipeline.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                )
                self.model_type = "chronos"
                logger.info(f"Initialized Chronos model: {self.model_name}")
                
            elif TRANSFORMERS_AVAILABLE:
                # Fallback to generic transformer approach
                self.pipeline = pipeline(
                    "text-generation", 
                    model=self.model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.model_type = "transformer"
                logger.info(f"Initialized transformer model: {self.model_name}")
                
            else:
                logger.warning("No suitable model available, using simple statistical fallback")
                self.model_type = "statistical_fallback"
                
        except Exception as e:
            logger.warning(f"Could not initialize {self.model_name}: {e}")
            logger.info("Using statistical fallback instead")
            self.model_type = "statistical_fallback"
    
    def validate_series(self, series: pd.Series) -> np.ndarray:
        """
        Convert series to model-compatible format.
        
        Args:
            series: Time series data
            
        Returns:
            NumPy array of values
        """
        if isinstance(series, pd.Series):
            values = series.values
        elif isinstance(series, pd.DataFrame):
            # Use first numeric column
            numeric_cols = series.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                values = series[numeric_cols[0]].values
            else:
                raise ValueError("No numeric columns found in DataFrame")
        else:
            values = np.array(series)
        
        # Remove NaN values
        values = values[~np.isnan(values)]
        
        if len(values) < 3:
            raise ValueError("Need at least 3 data points for forecasting")
            
        return values
    
    def forecast_chronos(self, values: np.ndarray, horizon: int) -> Dict[str, Any]:
        """Generate forecast using Chronos model."""
        try:
            # Prepare context (use last 64 points or all if less)
            context_length = min(len(values), 64)
            context = torch.tensor(values[-context_length:]).float()
            
            # Generate forecast
            forecast = self.pipeline.predict(
                context=context,
                prediction_length=horizon,
                num_samples=100,  # For uncertainty quantification
                temperature=1.0,
                top_k=50,
                top_p=1.0,
            )
            
            # Extract median forecast and quantiles
            forecast_median = np.median(forecast[0].numpy(), axis=0)
            
            # Calculate confidence intervals
            lower_80 = np.percentile(forecast[0].numpy(), 10, axis=0)
            upper_80 = np.percentile(forecast[0].numpy(), 90, axis=0)
            lower_90 = np.percentile(forecast[0].numpy(), 5, axis=0)
            upper_90 = np.percentile(forecast[0].numpy(), 95, axis=0)
            
            return {
                'forecast': forecast_median,
                'lower_bounds': {80: lower_80, 90: upper_80},
                'upper_bounds': {80: upper_80, 90: upper_90},
                'model_type': 'Chronos',
                'samples': forecast[0].numpy()  # All samples for analysis
            }
            
        except Exception as e:
            logger.error(f"Chronos forecasting failed: {e}")
            raise
    
    def forecast_statistical_fallback(self, values: np.ndarray, horizon: int) -> Dict[str, Any]:
        """Simple statistical forecast when models unavailable."""
        # Use exponential smoothing as fallback
        alpha = 0.3  # Smoothing parameter
        
        # Calculate trend
        trend = np.mean(np.diff(values[-min(12, len(values)-1):]))
        
        # Apply exponential smoothing
        smoothed = [values[0]]
        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
        
        # Generate forecast
        last_value = smoothed[-1]
        forecast = []
        for i in range(horizon):
            next_value = last_value + trend * (i + 1)
            forecast.append(next_value)
        
        forecast = np.array(forecast)
        
        # Simple confidence intervals based on historical volatility
        volatility = np.std(values)
        lower_80 = forecast - 1.28 * volatility
        upper_80 = forecast + 1.28 * volatility
        lower_90 = forecast - 1.64 * volatility
        upper_90 = forecast + 1.64 * volatility
        
        return {
            'forecast': forecast,
            'lower_bounds': {80: lower_80, 90: lower_90},
            'upper_bounds': {80: upper_80, 90: upper_90},
            'model_type': 'Statistical Fallback'
        }
    
    def forecast(self, series: Union[pd.Series, pd.DataFrame], 
                horizon: int = 12,
                frequency: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate forecast using available model.
        
        Args:
            series: Time series data
            horizon: Number of periods to forecast
            frequency: Time series frequency (for index generation)
        
        Returns:
            Dictionary with forecast results
        """
        try:
            # Validate and convert data
            values = self.validate_series(series)
            
            # Generate forecast based on available model
            if self.model_type == "chronos":
                result = self.forecast_chronos(values, horizon)
            else:
                result = self.forecast_statistical_fallback(values, horizon)
            
            # Create forecast index
            if isinstance(series, pd.Series) and hasattr(series, 'index'):
                last_date = series.index[-1]
                if frequency == 'M' or pd.infer_freq(series.index) == 'M':
                    forecast_index = pd.date_range(
                        start=last_date + pd.DateOffset(months=1),
                        periods=horizon,
                        freq='M'
                    )
                elif frequency == 'D' or pd.infer_freq(series.index) == 'D':
                    forecast_index = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=horizon,
                        freq='D'
                    )
                else:
                    forecast_index = pd.date_range(
                        start=last_date,
                        periods=horizon + 1,
                        freq=frequency or 'M'
                    )[1:]
            else:
                # Simple integer index
                forecast_index = np.arange(len(values), len(values) + horizon)
            
            # Add metadata
            result.update({
                'forecast_index': forecast_index,
                'model_info': self.model_info,
                'frequency': frequency,
                'horizon': horizon,
                'input_length': len(values)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            # Return fallback result
            return {
                'forecast': np.array([np.nan] * horizon),
                'forecast_index': pd.date_range(
                    start=pd.Timestamp.now(),
                    periods=horizon,
                    freq='M'
                ),
                'lower_bounds': {},
                'upper_bounds': {},
                'model_type': 'Error Fallback',
                'error': str(e)
            }
    
    def check_model_status(self) -> Dict[str, Any]:
        """Check model availability and status."""
        status = {
            'chronos_available': CHRONOS_AVAILABLE,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'model_info': self.model_info
        }
        
        if self.pipeline is not None:
            status['status'] = 'ready'
        else:
            status['status'] = 'fallback'
            
        return status


class HybridFoundationEnsemble:
    """
    Hybrid ensemble that combines paid (Nixtla TimeGPT), free open source (Nixtla OSS), 
    and free transformer (Hugging Face) models.
    Automatically falls back to free models when paid APIs unavailable.
    """
    
    def __init__(self, 
                 nixtla_api_key: Optional[str] = None,
                 hf_model: str = "amazon/chronos-t5-small",
                 include_nixtla_oss: bool = True,
                 nixtla_oss_type: str = "statistical",
                 prefer_paid: bool = True):
        """
        Initialize hybrid ensemble.
        
        Args:
            nixtla_api_key: Nixtla API key (optional)
            hf_model: Hugging Face model name
            include_nixtla_oss: Whether to include Nixtla open source models
            nixtla_oss_type: Type of Nixtla OSS models ('statistical', 'neural', 'ml', 'ensemble')
            prefer_paid: Whether to prefer paid models when available
        """
        self.prefer_paid = prefer_paid
        self.models = {}
        
        # Initialize Nixtla TimeGPT (paid) if API key available
        try:
            if nixtla_api_key and nixtla_api_key != "your_nixtla_api_key_here":
                from .timegpt_client import TimeGPTClient
                self.models['nixtla_timegpt'] = TimeGPTClient(api_key=nixtla_api_key)
                logger.info("Nixtla TimeGPT initialized (paid)")
        except Exception as e:
            logger.warning(f"Could not initialize Nixtla TimeGPT: {e}")
        
        # Initialize Nixtla Open Source (free)
        if include_nixtla_oss:
            try:
                from .nixtla_opensource import NixtlaOpenSourceForecaster
                self.models['nixtla_oss'] = NixtlaOpenSourceForecaster(model_type=nixtla_oss_type)
                logger.info(f"Nixtla open source ({nixtla_oss_type}) initialized (free)")
            except Exception as e:
                logger.warning(f"Could not initialize Nixtla open source: {e}")
        
        # Initialize Hugging Face (free)
        try:
            self.models['huggingface'] = HuggingFaceForecaster(model_name=hf_model)
            logger.info("Hugging Face model initialized (free)")
        except Exception as e:
            logger.warning(f"Could not initialize Hugging Face model: {e}")
        
        if not self.models:
            raise RuntimeError("No foundation models available")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def forecast(self, series: Union[pd.Series, pd.DataFrame],
                horizon: int = 12,
                model_preference: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate forecast using available models with fallback.
        
        Args:
            series: Time series data
            horizon: Forecast horizon
            model_preference: Specific model to use ('nixtla', 'huggingface', None for auto)
        
        Returns:
            Forecast results with model information
        """
        # Determine which model to use
        if model_preference and model_preference in self.models:
            model_to_use = model_preference
        elif self.prefer_paid and 'nixtla_timegpt' in self.models:
            model_to_use = 'nixtla_timegpt'
        elif 'nixtla_oss' in self.models:
            model_to_use = 'nixtla_oss'  # Prefer Nixtla OSS over HuggingFace
        elif 'huggingface' in self.models:
            model_to_use = 'huggingface'
        elif 'nixtla_timegpt' in self.models:
            model_to_use = 'nixtla_timegpt'
        else:
            raise RuntimeError("No models available for forecasting")
        
        logger.info(f"Using {model_to_use} model for forecasting")
        
        try:
            # Generate forecast
            model = self.models[model_to_use]
            
            # Handle different model interfaces
            if model_to_use == 'nixtla_oss':
                # Nixtla OSS uses fit_and_forecast method
                result = model.fit_and_forecast(series, horizon=horizon)
            else:
                # Standard forecast method
                result = model.forecast(series, horizon=horizon)
            
            # Add ensemble metadata
            result['ensemble_info'] = {
                'primary_model': model_to_use,
                'available_models': list(self.models.keys()),
                'model_preference': model_preference or 'auto'
            }
            
            return result
            
        except Exception as e:
            # Try fallback to other model
            fallback_models = [m for m in self.models.keys() if m != model_to_use]
            
            if fallback_models:
                fallback_model = fallback_models[0]
                logger.warning(f"{model_to_use} failed, falling back to {fallback_model}")
                
                try:
                    fallback_model_obj = self.models[fallback_model]
                    
                    # Handle different model interfaces for fallback
                    if fallback_model == 'nixtla_oss':
                        result = fallback_model_obj.fit_and_forecast(series, horizon=horizon)
                    else:
                        result = fallback_model_obj.forecast(series, horizon=horizon)
                    
                    result['ensemble_info'] = {
                        'primary_model': fallback_model,
                        'fallback_from': model_to_use,
                        'fallback_reason': str(e)
                    }
                    return result
                    
                except Exception as e2:
                    logger.error(f"All models failed: {e}, {e2}")
                    raise RuntimeError(f"All foundation models failed: {e}, {e2}")
            else:
                raise RuntimeError(f"Foundation model failed and no fallback available: {e}")
    
    def generate_ensemble_forecast(self, series: Union[pd.Series, pd.DataFrame],
                                  horizon: int = 12) -> Dict[str, Any]:
        """
        Generate forecasts from all available models and combine them.
        
        Args:
            series: Time series data
            horizon: Forecast horizon
        
        Returns:
            Combined ensemble forecast
        """
        forecasts = {}
        
        # Generate forecasts from all available models
        for model_name, model in self.models.items():
            try:
                # Handle different model interfaces
                if model_name == 'nixtla_oss':
                    result = model.fit_and_forecast(series, horizon=horizon)
                else:
                    result = model.forecast(series, horizon=horizon)
                
                forecasts[model_name] = result
                logger.info(f"Generated forecast using {model_name}")
            except Exception as e:
                logger.warning(f"Could not generate forecast with {model_name}: {e}")
        
        if not forecasts:
            raise RuntimeError("No models could generate forecasts")
        
        # Combine forecasts (simple average)
        forecast_values = []
        for model_name, result in forecasts.items():
            if 'forecast' in result and not np.any(np.isnan(result['forecast'])):
                forecast_values.append(result['forecast'])
        
        if not forecast_values:
            # Return single available forecast
            return list(forecasts.values())[0]
        
        # Average forecasts
        ensemble_forecast = np.mean(forecast_values, axis=0)
        
        # Use first available forecast_index
        forecast_index = list(forecasts.values())[0]['forecast_index']
        
        return {
            'forecast': ensemble_forecast,
            'forecast_index': forecast_index,
            'model_type': 'Hybrid Ensemble',
            'ensemble_info': {
                'models_used': list(forecasts.keys()),
                'total_models': len(self.models),
                'successful_models': len(forecast_values)
            },
            'individual_forecasts': forecasts,
            'horizon': horizon
        }


if __name__ == "__main__":
    # Example usage
    print("Testing Hybrid Foundation Model System...")
    
    # Generate sample data
    dates = pd.date_range('2020-01-01', '2023-12-01', freq='M')
    np.random.seed(42)
    values = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    series = pd.Series(values, index=dates, name='economic_indicator')
    
    # Test individual Hugging Face model
    print("\n1. Testing Hugging Face Model:")
    hf_model = HuggingFaceForecaster()
    status = hf_model.check_model_status()
    print(f"Model status: {status}")
    
    result = hf_model.forecast(series, horizon=6)
    print(f"Forecast generated: {len(result['forecast'])} periods")
    print(f"Model type: {result['model_type']}")
    
    # Test hybrid ensemble
    print("\n2. Testing Hybrid Ensemble:")
    try:
        ensemble = HybridFoundationEnsemble(prefer_paid=False)  # Prefer free models
        available = ensemble.get_available_models()
        print(f"Available models: {available}")
        
        ensemble_result = ensemble.forecast(series, horizon=6)
        print(f"Ensemble forecast: {ensemble_result['ensemble_info']}")
        
        # Test ensemble forecast if multiple models available
        if len(available) > 1:
            combined_result = ensemble.generate_ensemble_forecast(series, horizon=6)
            print(f"Combined ensemble: {combined_result['ensemble_info']}")
        
    except Exception as e:
        print(f"Ensemble test failed: {e}")
    
    print("\n✅ Hybrid Foundation Model System Test Complete!")