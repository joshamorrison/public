"""
Nixtla Open Source Foundation Models
Integrates StatsForecast, NeuralForecast, and other free Nixtla libraries.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Nixtla Open Source Libraries
try:
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoARIMA, AutoETS, AutoCES, AutoTheta,
        ARIMA, Theta, MSTL, TBATS,
        Naive, SeasonalNaive, HistoricAverage
    )
    STATSFORECAST_AVAILABLE = True
except ImportError:
    STATSFORECAST_AVAILABLE = False
    logging.warning("StatsForecast not available. Install with: pip install statsforecast")

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import (
        NBEATS, NHITS, MLP, LSTM, GRU, RNN,
        TFT, PatchTST, TimesNet, DeepAR
    )
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False
    logging.warning("NeuralForecast not available. Install with: pip install neuralforecast")

try:
    from mlforecast import MLForecast
    from mlforecast.target_transforms import Differences
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    MLFORECAST_AVAILABLE = True
except ImportError:
    MLFORECAST_AVAILABLE = False
    logging.warning("MLForecast not available. Install with: pip install mlforecast")

logger = logging.getLogger(__name__)


class NixtlaOpenSourceForecaster:
    """Free Nixtla open source models for time series forecasting."""
    
    def __init__(self, model_type: str = "statistical"):
        """
        Initialize Nixtla open source forecaster.
        
        Args:
            model_type: Type of models to use
                - 'statistical': StatsForecast models (fast, interpretable)
                - 'neural': NeuralForecast models (deep learning)
                - 'ml': MLForecast models (machine learning)
                - 'ensemble': Combination of multiple types
        """
        self.model_type = model_type
        self.forecaster = None
        self.fitted_models = {}
        
        self.model_info = {
            'name': f'Nixtla {model_type.title()}',
            'type': 'Open Source Foundation Models',
            'provider': 'Nixtla',
            'cost': 'Free',
            'zero_shot': False  # Requires fitting
        }
        
        logger.info(f"Initialized Nixtla {model_type} forecaster")
    
    def _prepare_data(self, series: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Convert series to Nixtla-compatible format."""
        if isinstance(series, pd.Series):
            df = pd.DataFrame({
                'unique_id': 'series_1',
                'ds': series.index,
                'y': series.values
            })
        elif isinstance(series, pd.DataFrame):
            if 'ds' in series.columns and 'y' in series.columns:
                df = series.copy()
                if 'unique_id' not in df.columns:
                    df['unique_id'] = 'series_1'
            else:
                # Assume first column is dates, second is values
                df = pd.DataFrame({
                    'unique_id': 'series_1',
                    'ds': series.iloc[:, 0],
                    'y': series.iloc[:, 1]
                })
        else:
            raise ValueError("Input must be pandas Series or DataFrame")
        
        # Ensure proper types
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])
        df = df.dropna()
        
        return df
    
    def _create_statistical_models(self) -> List:
        """Create StatsForecast models."""
        if not STATSFORECAST_AVAILABLE:
            raise ImportError("StatsForecast not available")
        
        models = [
            AutoARIMA(season_length=12),  # For monthly data
            AutoETS(season_length=12),
            AutoTheta(season_length=12),
            Naive(),
            SeasonalNaive(season_length=12),
            # Fast baseline models
            HistoricAverage(),
        ]
        
        # Add more sophisticated models if we have enough data
        try:
            models.extend([
                MSTL(season_length=[12]),  # Multiple seasonal decomposition
                AutoCES(season_length=12),  # Complex exponential smoothing
            ])
        except Exception as e:
            logger.warning(f"Could not add advanced statistical models: {e}")
        
        return models
    
    def _create_neural_models(self, input_size: int = 24, horizon: int = 12) -> List:
        """Create NeuralForecast models."""
        if not NEURALFORECAST_AVAILABLE:
            raise ImportError("NeuralForecast not available")
        
        models = [
            # Fast and effective models
            NBEATS(input_size=input_size, h=horizon, max_steps=50),
            NHITS(input_size=input_size, h=horizon, max_steps=50),
            
            # Classic neural networks
            MLP(input_size=input_size, h=horizon, max_steps=50),
            
            # RNN family (if enough data)
            LSTM(input_size=input_size, h=horizon, max_steps=50),
        ]
        
        # Add transformer models for larger datasets
        try:
            if input_size >= 48:  # Need sufficient context
                models.extend([
                    PatchTST(input_size=input_size, h=horizon, max_steps=30),
                    TFT(input_size=input_size, h=horizon, max_steps=30),
                ])
        except Exception as e:
            logger.warning(f"Could not add transformer models: {e}")
        
        return models
    
    def _create_ml_models(self) -> Dict:
        """Create MLForecast models."""
        if not MLFORECAST_AVAILABLE:
            raise ImportError("MLForecast not available")
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
        }
        
        return models
    
    def fit_and_forecast(self, series: Union[pd.Series, pd.DataFrame], 
                        horizon: int = 12,
                        level: List[int] = [80, 90]) -> Dict[str, Any]:
        """
        Fit models and generate forecasts.
        
        Args:
            series: Time series data
            horizon: Forecast horizon
            level: Confidence levels for prediction intervals
        
        Returns:
            Dictionary with forecast results
        """
        try:
            # Prepare data
            df = self._prepare_data(series)
            data_length = len(df)
            
            logger.info(f"Prepared data: {data_length} observations for {self.model_type} forecasting")
            
            if self.model_type == "statistical":
                return self._forecast_statistical(df, horizon, level)
            elif self.model_type == "neural":
                return self._forecast_neural(df, horizon, level, data_length)
            elif self.model_type == "ml":
                return self._forecast_ml(df, horizon, level)
            elif self.model_type == "ensemble":
                return self._forecast_ensemble(df, horizon, level, data_length)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Nixtla forecasting failed: {e}")
            # Return fallback
            return {
                'forecast': np.array([np.nan] * horizon),
                'forecast_index': pd.date_range(
                    start=pd.Timestamp.now(),
                    periods=horizon,
                    freq='M'
                ),
                'lower_bounds': {},
                'upper_bounds': {},
                'model_type': f'Nixtla {self.model_type} (Error)',
                'error': str(e)
            }
    
    def _forecast_statistical(self, df: pd.DataFrame, horizon: int, level: List[int]) -> Dict[str, Any]:
        """Generate forecast using StatsForecast."""
        models = self._create_statistical_models()
        
        # Infer frequency
        freq = pd.infer_freq(df['ds'])
        if freq is None:
            freq = 'M'  # Default to monthly
        
        sf = StatsForecast(
            models=models,
            freq=freq,
            n_jobs=-1  # Use all cores
        )
        
        # Fit and forecast
        sf.fit(df)
        forecasts = sf.predict(h=horizon, level=level)
        
        # Process results
        forecast_values = forecasts['AutoARIMA'].values  # Use AutoARIMA as primary
        
        # Extract confidence intervals
        lower_bounds = {}
        upper_bounds = {}
        for lvl in level:
            lower_col = f'AutoARIMA-lo-{lvl}'
            upper_col = f'AutoARIMA-hi-{lvl}'
            if lower_col in forecasts.columns:
                lower_bounds[lvl] = forecasts[lower_col].values
            if upper_col in forecasts.columns:
                upper_bounds[lvl] = forecasts[upper_col].values
        
        # Create forecast index
        last_date = df['ds'].iloc[-1]
        if freq == 'M':
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq='M'
            )
        else:
            forecast_index = pd.date_range(
                start=last_date,
                periods=horizon + 1,
                freq=freq
            )[1:]
        
        return {
            'forecast': forecast_values,
            'forecast_index': forecast_index,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'model_type': 'Nixtla StatsForecast',
            'model_info': self.model_info,
            'raw_forecasts': forecasts,
            'models_used': [type(model).__name__ for model in models]
        }
    
    def _forecast_neural(self, df: pd.DataFrame, horizon: int, level: List[int], data_length: int) -> Dict[str, Any]:
        """Generate forecast using NeuralForecast."""
        input_size = min(24, data_length // 2)  # Adaptive input size
        models = self._create_neural_models(input_size, horizon)
        
        # Infer frequency
        freq = pd.infer_freq(df['ds'])
        if freq is None:
            freq = 'M'
        
        nf = NeuralForecast(models=models, freq=freq)
        
        # Fit and forecast
        nf.fit(df)
        forecasts = nf.predict()
        
        # Use NBEATS as primary model (usually best performer)
        primary_model = 'NBEATS'
        if primary_model not in forecasts.columns:
            primary_model = forecasts.columns[0]  # Use first available
        
        forecast_values = forecasts[primary_model].values
        
        # Neural models don't always provide confidence intervals
        # Generate approximate intervals based on recent volatility
        recent_values = df['y'].tail(min(24, len(df)))
        volatility = recent_values.std()
        
        lower_bounds = {}
        upper_bounds = {}
        for lvl in level:
            z_score = 1.28 if lvl == 80 else 1.64  # Approximate z-scores
            margin = z_score * volatility
            lower_bounds[lvl] = forecast_values - margin
            upper_bounds[lvl] = forecast_values + margin
        
        # Create forecast index
        last_date = df['ds'].iloc[-1]
        if freq == 'M':
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq='M'
            )
        else:
            forecast_index = pd.date_range(
                start=last_date,
                periods=horizon + 1,
                freq=freq
            )[1:]
        
        return {
            'forecast': forecast_values,
            'forecast_index': forecast_index,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'model_type': 'Nixtla NeuralForecast',
            'model_info': self.model_info,
            'raw_forecasts': forecasts,
            'primary_model': primary_model,
            'models_used': [type(model).__name__ for model in models]
        }
    
    def _forecast_ml(self, df: pd.DataFrame, horizon: int, level: List[int]) -> Dict[str, Any]:
        """Generate forecast using MLForecast.""" 
        models = self._create_ml_models()
        
        # Infer frequency
        freq = pd.infer_freq(df['ds'])
        if freq is None:
            freq = 'M'
        
        mlf = MLForecast(
            models=models,
            freq=freq,
            target_transforms=[Differences([1])],  # First difference for stationarity
        )
        
        # Fit and forecast
        mlf.fit(df)
        forecasts = mlf.predict(h=horizon)
        
        # Use RandomForest as primary
        primary_model = 'RandomForest'
        if primary_model not in forecasts.columns:
            primary_model = forecasts.columns[0]
        
        forecast_values = forecasts[primary_model].values
        
        # Generate confidence intervals based on model residuals
        recent_values = df['y'].tail(min(24, len(df)))
        volatility = recent_values.std()
        
        lower_bounds = {}
        upper_bounds = {}
        for lvl in level:
            z_score = 1.28 if lvl == 80 else 1.64
            margin = z_score * volatility
            lower_bounds[lvl] = forecast_values - margin
            upper_bounds[lvl] = forecast_values + margin
        
        # Create forecast index
        last_date = df['ds'].iloc[-1]
        if freq == 'M':
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq='M'
            )
        else:
            forecast_index = pd.date_range(
                start=last_date,
                periods=horizon + 1,
                freq=freq
            )[1:]
        
        return {
            'forecast': forecast_values,
            'forecast_index': forecast_index,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'model_type': 'Nixtla MLForecast',
            'model_info': self.model_info,
            'raw_forecasts': forecasts,
            'primary_model': primary_model,
            'models_used': list(models.keys())
        }
    
    def _forecast_ensemble(self, df: pd.DataFrame, horizon: int, level: List[int], data_length: int) -> Dict[str, Any]:
        """Generate ensemble forecast using multiple Nixtla libraries."""
        results = {}
        forecasts_to_combine = []
        
        # Try statistical models
        if STATSFORECAST_AVAILABLE:
            try:
                stats_result = self._forecast_statistical(df, horizon, level)
                results['statistical'] = stats_result
                forecasts_to_combine.append(stats_result['forecast'])
            except Exception as e:
                logger.warning(f"Statistical forecasting failed: {e}")
        
        # Try ML models
        if MLFORECAST_AVAILABLE:
            try:
                ml_result = self._forecast_ml(df, horizon, level)
                results['ml'] = ml_result
                forecasts_to_combine.append(ml_result['forecast'])
            except Exception as e:
                logger.warning(f"ML forecasting failed: {e}")
        
        # Try neural models (if enough data)
        if NEURALFORECAST_AVAILABLE and data_length >= 48:
            try:
                neural_result = self._forecast_neural(df, horizon, level, data_length)
                results['neural'] = neural_result
                forecasts_to_combine.append(neural_result['forecast'])
            except Exception as e:
                logger.warning(f"Neural forecasting failed: {e}")
        
        if not forecasts_to_combine:
            raise RuntimeError("No Nixtla models could generate forecasts")
        
        # Combine forecasts (simple average)
        ensemble_forecast = np.mean(forecasts_to_combine, axis=0)
        
        # Use first result for metadata
        first_result = list(results.values())[0]
        
        return {
            'forecast': ensemble_forecast,
            'forecast_index': first_result['forecast_index'],
            'lower_bounds': first_result['lower_bounds'],  # Use first model's intervals
            'upper_bounds': first_result['upper_bounds'],
            'model_type': 'Nixtla Ensemble',
            'model_info': self.model_info,
            'individual_results': results,
            'models_used': len(forecasts_to_combine),
            'ensemble_components': list(results.keys())
        }
    
    def check_availability(self) -> Dict[str, Any]:
        """Check which Nixtla libraries are available."""
        return {
            'statsforecast_available': STATSFORECAST_AVAILABLE,
            'neuralforecast_available': NEURALFORECAST_AVAILABLE,
            'mlforecast_available': MLFORECAST_AVAILABLE,
            'recommended_install': [
                'pip install statsforecast',
                'pip install neuralforecast', 
                'pip install mlforecast'
            ],
            'model_type': self.model_type,
            'status': 'ready' if any([STATSFORECAST_AVAILABLE, NEURALFORECAST_AVAILABLE, MLFORECAST_AVAILABLE]) else 'needs_install'
        }


if __name__ == "__main__":
    # Example usage
    print("Testing Nixtla Open Source Forecasting...")
    
    # Generate sample data
    dates = pd.date_range('2020-01-01', '2023-12-01', freq='M')
    np.random.seed(42)
    values = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    series = pd.Series(values, index=dates, name='economic_indicator')
    
    # Test different model types
    model_types = ['statistical', 'neural', 'ml', 'ensemble']
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Testing {model_type.upper()} Models")
        print('='*50)
        
        try:
            forecaster = NixtlaOpenSourceForecaster(model_type=model_type)
            availability = forecaster.check_availability()
            print(f"Status: {availability['status']}")
            
            if availability['status'] == 'ready':
                result = forecaster.fit_and_forecast(series, horizon=6)
                print(f"Forecast generated: {len(result['forecast'])} periods")
                print(f"Model type: {result['model_type']}")
                if 'models_used' in result:
                    print(f"Models used: {result['models_used']}")
                print(f"Sample forecast: {result['forecast'][:3]}")
            else:
                print("Libraries not available - install with:")
                for install_cmd in availability['recommended_install']:
                    print(f"  {install_cmd}")
                    
        except Exception as e:
            print(f"Error testing {model_type}: {e}")
    
    print(f"\n{'='*50}")
    print("Nixtla Open Source Test Complete!")
    print('='*50)