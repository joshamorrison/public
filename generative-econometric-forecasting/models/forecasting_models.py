"""
Enhanced econometric forecasting models with GenAI capabilities.
Integrates foundation models, uncertainty quantification, synthetic data generation, and sentiment analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
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

# Enhanced GenAI imports
try:
    from ..foundation_models.timegpt_client import TimeGPTClient
    from ..foundation_models.huggingface_forecaster import HuggingFaceForecaster, HybridFoundationEnsemble
    from ..foundation_models.foundation_ensemble import FoundationModelEnsemble
    FOUNDATION_MODELS_AVAILABLE = True
except ImportError:
    FOUNDATION_MODELS_AVAILABLE = False
    logging.warning("Foundation models not available")

try:
    from ..uncertainty.bayesian_forecaster import BayesianForecaster
    from ..uncertainty.monte_carlo_simulator import MonteCarloSimulator
    UNCERTAINTY_MODELS_AVAILABLE = True
except ImportError:
    UNCERTAINTY_MODELS_AVAILABLE = False
    logging.warning("Uncertainty models not available")

try:
    from ..synthetic.economic_gan import EconomicGAN
    from ..synthetic.economic_vae import EconomicVAE
    from ..synthetic.data_augmentation import EconomicDataAugmentor
    SYNTHETIC_MODELS_AVAILABLE = True
except ImportError:
    SYNTHETIC_MODELS_AVAILABLE = False
    logging.warning("Synthetic data models not available")

try:
    from ..data.unstructured.ai_economy_score import AIEconomyScoreGenerator
    SENTIMENT_ANALYSIS_AVAILABLE = True
except ImportError:
    SENTIMENT_ANALYSIS_AVAILABLE = False
    logging.warning("Sentiment analysis not available")

logger = logging.getLogger(__name__)


class EconometricForecaster:
    """Enhanced econometric forecasting with GenAI capabilities."""
    
    def __init__(self, enable_genai: bool = True):
        """
        Initialize enhanced forecaster with GenAI capabilities.
        
        Args:
            enable_genai: Whether to enable GenAI features
        """
        self.models = {}
        self.fitted_models = {}
        self.forecasts = {}
        self.model_performance = {}
        self.enable_genai = enable_genai
        
        # Initialize GenAI components
        if enable_genai:
            self._initialize_genai_components()
    
    def _initialize_genai_components(self):
        """Initialize GenAI components."""
        # Foundation models - hybrid approach with paid and free models
        if FOUNDATION_MODELS_AVAILABLE:
            try:
                # Initialize hybrid ensemble with both paid and free models
                import os
                nixtla_api_key = os.getenv('NIXTLA_API_KEY')
                
                # Prefer free models by default to ensure accessibility
                self.hybrid_foundation_ensemble = HybridFoundationEnsemble(
                    nixtla_api_key=nixtla_api_key,
                    hf_model="amazon/chronos-t5-small",  # Fast free model
                    prefer_paid=False  # Prefer free models for accessibility
                )
                
                # Also keep individual clients for compatibility
                self.huggingface_client = HuggingFaceForecaster()
                if nixtla_api_key and nixtla_api_key != "your_nixtla_api_key_here":
                    self.timegpt_client = TimeGPTClient(api_key=nixtla_api_key)
                else:
                    self.timegpt_client = None
                    logger.info("Nixtla API key not configured, using free alternatives only")
                
                available_models = self.hybrid_foundation_ensemble.get_available_models()
                logger.info(f"Foundation models initialized: {available_models}")
                
            except Exception as e:
                logger.warning(f"Could not initialize foundation models: {e}")
                self.hybrid_foundation_ensemble = None
                self.huggingface_client = None
                self.timegpt_client = None
        else:
            logger.warning("Foundation models not available")
        
        # Uncertainty quantification
        if UNCERTAINTY_MODELS_AVAILABLE:
            self.bayesian_forecaster = BayesianForecaster()
            self.monte_carlo_simulator = MonteCarloSimulator()
            logger.info("Uncertainty models initialized")
        else:
            logger.warning("Uncertainty models not available")
        
        # Synthetic data generation
        if SYNTHETIC_MODELS_AVAILABLE:
            self.data_augmentor = EconomicDataAugmentor()
            logger.info("Synthetic data models initialized")
        else:
            logger.warning("Synthetic data models not available")
        
        # Sentiment analysis
        if SENTIMENT_ANALYSIS_AVAILABLE:
            self.ai_economy_score = AIEconomyScoreGenerator()
            logger.info("Sentiment analysis initialized")
        else:
            logger.warning("Sentiment analysis not available")
        
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
    
    def generate_foundation_model_forecast(self, series: pd.Series, 
                                         periods: int = 12,
                                         model_type: str = 'hybrid') -> Dict[str, Any]:
        """
        Generate forecasts using foundation models (TimeGPT, Hugging Face, etc.).
        Now supports hybrid ensemble with paid and free models.
        
        Args:
            series: Time series to forecast
            periods: Number of periods to forecast
            model_type: 'timegpt', 'huggingface', 'hybrid', or 'ensemble'
        
        Returns:
            Dictionary with foundation model forecast results
        """
        if not self.enable_genai or not FOUNDATION_MODELS_AVAILABLE:
            raise ValueError("Foundation models not available")
        
        results = {}
        
        # Use hybrid ensemble by default (supports both paid and free models)
        if model_type in ['hybrid', 'ensemble'] and hasattr(self, 'hybrid_foundation_ensemble'):
            try:
                if self.hybrid_foundation_ensemble:
                    # Generate ensemble forecast using all available models
                    ensemble_result = self.hybrid_foundation_ensemble.generate_ensemble_forecast(
                        series, horizon=periods
                    )
                    results['hybrid_ensemble'] = ensemble_result
                    
                    # Also get individual model results for comparison
                    available_models = self.hybrid_foundation_ensemble.get_available_models()
                    for model_name in available_models:
                        try:
                            individual_result = self.hybrid_foundation_ensemble.forecast(
                                series, horizon=periods, model_preference=model_name
                            )
                            results[model_name] = individual_result
                        except Exception as e:
                            logger.warning(f"Individual {model_name} forecast failed: {e}")
                    
                    logger.info(f"Hybrid ensemble forecast with {len(available_models)} models")
                else:
                    raise ValueError("Hybrid foundation ensemble not initialized")
                    
            except Exception as e:
                logger.error(f"Hybrid ensemble forecast failed: {e}")
                # Fallback to individual models
                model_type = 'huggingface'  # Try free model as fallback
        
        # Individual TimeGPT (paid model)
        if model_type in ['timegpt', 'ensemble'] and hasattr(self, 'timegpt_client'):
            try:
                timegpt_result = self.timegpt_client.forecast(series, horizon=periods)
                results['timegpt'] = {
                    'forecast': timegpt_result['forecast'],
                    'confidence_intervals': timegpt_result.get('confidence_intervals', {}),
                    'model_type': 'TimeGPT'
                }
                logger.info("TimeGPT forecast generated")
            except Exception as e:
                logger.error(f"TimeGPT forecast failed: {e}")
        
        # Individual Hugging Face models (free alternative)
        if model_type in ['huggingface', 'chronos', 'ensemble'] and hasattr(self, 'huggingface_client'):
            try:
                if self.huggingface_client:
                    hf_result = self.huggingface_client.forecast(series, horizon=periods)
                    results['huggingface'] = hf_result
                    logger.info("Hugging Face forecast generated")
                else:
                    logger.warning("Hugging Face client not available")
            except Exception as e:
                logger.error(f"Hugging Face forecast failed: {e}")
        
        # Legacy Chronos client (if available separately)  
        if model_type in ['chronos', 'ensemble'] and hasattr(self, 'chronos_client'):
            try:
                chronos_result = self.chronos_client.forecast(series, horizon=periods)
                results['chronos'] = {
                    'forecast': chronos_result['forecast'],
                    'confidence_intervals': chronos_result.get('confidence_intervals', {}),
                    'model_type': 'Chronos'
                }
                logger.info("Legacy Chronos forecast generated")
            except Exception as e:
                logger.error(f"Legacy Chronos forecast failed: {e}")
        
        # Legacy foundation ensemble (if available)
        if model_type == 'ensemble' and hasattr(self, 'foundation_ensemble') and len(results) > 1:
            try:
                ensemble_result = self.foundation_ensemble.forecast(series, horizon=periods)
                results['foundation_ensemble'] = {
                    'forecast': ensemble_result['forecast'],
                    'confidence_intervals': ensemble_result.get('confidence_intervals', {}),
                    'model_type': 'Foundation Ensemble',
                    'individual_results': ensemble_result.get('individual_results', {})
                }
                logger.info("Legacy foundation ensemble forecast generated")
            except Exception as e:
                logger.error(f"Legacy foundation ensemble forecast failed: {e}")
        
        if not results:
            raise ValueError("No foundation models could generate forecasts")
        
        return results
    
    def generate_bayesian_forecast(self, series: pd.Series, 
                                 periods: int = 12,
                                 model_type: str = 'linear_trend',
                                 include_uncertainty: bool = True) -> Dict[str, Any]:
        """
        Generate Bayesian forecasts with uncertainty quantification.
        
        Args:
            series: Time series to forecast
            periods: Number of periods to forecast
            model_type: Type of Bayesian model
            include_uncertainty: Whether to include uncertainty bands
        
        Returns:
            Dictionary with Bayesian forecast results
        """
        if not self.enable_genai or not UNCERTAINTY_MODELS_AVAILABLE:
            raise ValueError("Uncertainty models not available")
        
        try:
            # Fit Bayesian model
            bayesian_result = self.bayesian_forecaster.fit_bayesian_linear_trend(
                series, model_name=f'bayesian_{series.name}'
            )
            
            # Generate forecast
            forecast_result = self.bayesian_forecaster.forecast_bayesian_model(
                f'bayesian_{series.name}', horizon=periods
            )
            
            result = {
                'forecast': forecast_result['mean_forecast'],
                'model_type': 'Bayesian',
                'model_summary': bayesian_result.get('model_summary', {}),
                'convergence_diagnostics': bayesian_result.get('convergence_diagnostics', {})
            }
            
            if include_uncertainty:
                result.update({
                    'credible_intervals': forecast_result.get('credible_intervals', {}),
                    'prediction_intervals': forecast_result.get('prediction_intervals', {}),
                    'uncertainty_measures': forecast_result.get('uncertainty_measures', {})
                })
            
            logger.info("Bayesian forecast generated")
            return result
            
        except Exception as e:
            logger.error(f"Bayesian forecast failed: {e}")
            raise
    
    def generate_monte_carlo_scenarios(self, series: pd.Series,
                                     periods: int = 12,
                                     n_scenarios: int = 1000,
                                     scenario_type: str = 'economic') -> Dict[str, Any]:
        """
        Generate Monte Carlo scenario forecasts.
        
        Args:
            series: Time series to forecast
            periods: Number of periods to forecast
            n_scenarios: Number of scenarios to generate
            scenario_type: Type of scenarios to generate
        
        Returns:
            Dictionary with Monte Carlo scenario results
        """
        if not self.enable_genai or not UNCERTAINTY_MODELS_AVAILABLE:
            raise ValueError("Uncertainty models not available")
        
        try:
            # Set up simulation parameters
            from ..uncertainty.monte_carlo_simulator import SimulationParameters
            
            params = SimulationParameters(
                n_scenarios=n_scenarios,
                time_horizon=periods,
                variables=[series.name] if series.name else ['series'],
                correlations={},
                initial_values={series.name or 'series': series.iloc[-1]}
            )
            
            # Run simulation
            simulation_result = self.monte_carlo_simulator.simulate_economic_paths(params)
            
            result = {
                'scenarios': simulation_result['scenarios'],
                'statistics': simulation_result['scenario_statistics'],
                'percentiles': simulation_result.get('percentiles', {}),
                'model_type': 'Monte Carlo',
                'n_scenarios': n_scenarios
            }
            
            logger.info(f"Generated {n_scenarios} Monte Carlo scenarios")
            return result
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {e}")
            raise
    
    def enhance_forecast_with_sentiment(self, series: pd.Series, 
                                      periods: int = 12,
                                      base_forecast: np.ndarray = None) -> Dict[str, Any]:
        """
        Enhance forecasts with sentiment analysis.
        
        Args:
            series: Time series to forecast
            periods: Number of periods to forecast
            base_forecast: Base forecast to enhance (optional)
        
        Returns:
            Dictionary with sentiment-enhanced forecast
        """
        if not self.enable_genai or not SENTIMENT_ANALYSIS_AVAILABLE:
            raise ValueError("Sentiment analysis not available")
        
        try:
            # Generate AI Economy Score
            economy_score = self.ai_economy_score.generate_ai_economy_score()
            
            # Generate base forecast if not provided
            if base_forecast is None:
                ensemble_result = self.generate_ensemble_forecast(series, periods)
                base_forecast = ensemble_result['ensemble_forecast']
            
            # Apply sentiment adjustment
            sentiment_value = economy_score['overall_score']
            sentiment_trend = economy_score.get('trend_analysis', {}).get('trend_direction', 0)
            
            # Simple sentiment adjustment (can be made more sophisticated)
            sentiment_multiplier = 1.0 + (sentiment_value * 0.1)  # 10% max adjustment
            trend_adjustment = np.linspace(0, sentiment_trend * 0.05, periods)  # 5% max trend
            
            enhanced_forecast = base_forecast * sentiment_multiplier + trend_adjustment
            
            result = {
                'enhanced_forecast': enhanced_forecast,
                'base_forecast': base_forecast,
                'sentiment_score': sentiment_value,
                'sentiment_trend': sentiment_trend,
                'economy_score_details': economy_score,
                'model_type': 'Sentiment Enhanced'
            }
            
            logger.info("Sentiment-enhanced forecast generated")
            return result
            
        except Exception as e:
            logger.error(f"Sentiment enhancement failed: {e}")
            raise
    
    def generate_synthetic_data_forecast(self, series: pd.Series,
                                       periods: int = 12,
                                       augmentation_method: str = 'gan',
                                       n_synthetic_series: int = 100) -> Dict[str, Any]:
        """
        Generate forecasts using synthetic data augmentation.
        
        Args:
            series: Time series to forecast
            periods: Number of periods to forecast
            augmentation_method: 'gan', 'vae', or 'augmentation'
            n_synthetic_series: Number of synthetic series to generate
        
        Returns:
            Dictionary with synthetic data enhanced forecast
        """
        if not self.enable_genai or not SYNTHETIC_MODELS_AVAILABLE:
            raise ValueError("Synthetic data models not available")
        
        try:
            if augmentation_method == 'augmentation':
                # Use data augmentation techniques
                augmented_series = self.data_augmentor.augment_dataset(
                    series, 
                    augmentation_factor=n_synthetic_series // 5,
                    preserve_properties=True
                )
                
                # Generate forecasts on augmented data
                forecasts = []
                for aug_series in augmented_series[:10]:  # Limit to prevent memory issues
                    try:
                        ensemble_result = self.generate_ensemble_forecast(aug_series, periods)
                        forecasts.append(ensemble_result['ensemble_forecast'])
                    except:
                        continue
                
                if forecasts:
                    mean_forecast = np.mean(forecasts, axis=0)
                    std_forecast = np.std(forecasts, axis=0)
                    
                    result = {
                        'forecast': mean_forecast,
                        'uncertainty': std_forecast,
                        'individual_forecasts': forecasts,
                        'model_type': 'Data Augmentation Enhanced',
                        'n_augmented_series': len(augmented_series)
                    }
                else:
                    raise ValueError("No successful forecasts from augmented data")
            
            elif augmentation_method == 'gan':
                # Use GAN for synthetic data generation
                gan = EconomicGAN(sequence_length=min(50, len(series)))
                training_data = gan.prepare_data(series)
                
                # Train GAN (simplified training)
                gan.train(training_data, epochs=100, batch_size=8)
                
                # Generate synthetic scenarios
                scenarios = gan.generate_economic_scenarios(
                    scenario_types=['normal', 'recession', 'boom'],
                    samples_per_scenario=n_synthetic_series // 3
                )
                
                # Combine scenarios for forecast
                all_synthetic = np.concatenate(list(scenarios.values()))
                mean_forecast = np.mean(all_synthetic[:, -periods:], axis=0)
                std_forecast = np.std(all_synthetic[:, -periods:], axis=0)
                
                result = {
                    'forecast': mean_forecast,
                    'uncertainty': std_forecast,
                    'scenarios': scenarios,
                    'model_type': 'GAN Enhanced',
                    'n_synthetic_series': len(all_synthetic)
                }
            
            elif augmentation_method == 'vae':
                # Use VAE for synthetic data generation
                vae = EconomicVAE(latent_dim=16, use_time_series=True)
                training_data = vae.prepare_data(series, sequence_length=min(50, len(series)))
                
                # Train VAE (simplified training)
                vae.train(training_data, epochs=100, batch_size=8)
                
                # Generate scenarios
                scenarios = vae.generate_economic_scenarios(
                    base_conditions={},
                    n_scenarios=n_synthetic_series,
                    uncertainty_scale=1.0
                )
                
                # Combine scenarios for forecast
                all_synthetic = np.concatenate(list(scenarios.values()))
                mean_forecast = np.mean(all_synthetic[:, -periods:], axis=0)
                std_forecast = np.std(all_synthetic[:, -periods:], axis=0)
                
                result = {
                    'forecast': mean_forecast,
                    'uncertainty': std_forecast,
                    'scenarios': scenarios,
                    'model_type': 'VAE Enhanced',
                    'n_synthetic_series': len(all_synthetic)
                }
            
            logger.info(f"Synthetic data forecast generated using {augmentation_method}")
            return result
            
        except Exception as e:
            logger.error(f"Synthetic data forecast failed: {e}")
            raise
    
    def generate_comprehensive_forecast(self, series: pd.Series,
                                      periods: int = 12,
                                      include_traditional: bool = True,
                                      include_foundation: bool = True,
                                      include_bayesian: bool = True,
                                      include_sentiment: bool = True,
                                      include_synthetic: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive forecast using all available methods.
        
        Args:
            series: Time series to forecast
            periods: Number of periods to forecast
            include_traditional: Include traditional econometric models
            include_foundation: Include foundation models
            include_bayesian: Include Bayesian uncertainty quantification
            include_sentiment: Include sentiment analysis
            include_synthetic: Include synthetic data enhancement
        
        Returns:
            Dictionary with comprehensive forecast results
        """
        results = {}
        
        # Traditional econometric models
        if include_traditional:
            try:
                traditional_result = self.generate_ensemble_forecast(series, periods)
                results['traditional'] = traditional_result
                logger.info("Traditional forecast generated")
            except Exception as e:
                logger.error(f"Traditional forecast failed: {e}")
        
        # Foundation models
        if include_foundation and self.enable_genai and FOUNDATION_MODELS_AVAILABLE:
            try:
                foundation_result = self.generate_foundation_model_forecast(
                    series, periods, model_type='ensemble'
                )
                results['foundation'] = foundation_result
                logger.info("Foundation model forecast generated")
            except Exception as e:
                logger.error(f"Foundation model forecast failed: {e}")
        
        # Bayesian uncertainty quantification
        if include_bayesian and self.enable_genai and UNCERTAINTY_MODELS_AVAILABLE:
            try:
                bayesian_result = self.generate_bayesian_forecast(series, periods)
                results['bayesian'] = bayesian_result
                logger.info("Bayesian forecast generated")
            except Exception as e:
                logger.error(f"Bayesian forecast failed: {e}")
        
        # Sentiment enhancement
        if include_sentiment and self.enable_genai and SENTIMENT_ANALYSIS_AVAILABLE:
            try:
                # Use traditional forecast as base if available
                base_forecast = None
                if 'traditional' in results:
                    base_forecast = results['traditional']['ensemble_forecast']
                
                sentiment_result = self.enhance_forecast_with_sentiment(
                    series, periods, base_forecast
                )
                results['sentiment'] = sentiment_result
                logger.info("Sentiment-enhanced forecast generated")
            except Exception as e:
                logger.error(f"Sentiment forecast failed: {e}")
        
        # Synthetic data enhancement
        if include_synthetic and self.enable_genai and SYNTHETIC_MODELS_AVAILABLE:
            try:
                synthetic_result = self.generate_synthetic_data_forecast(
                    series, periods, augmentation_method='augmentation'
                )
                results['synthetic'] = synthetic_result
                logger.info("Synthetic data forecast generated")
            except Exception as e:
                logger.error(f"Synthetic forecast failed: {e}")
        
        # Create meta-ensemble if multiple forecasts available
        if len(results) > 1:
            try:
                forecasts = []
                weights = []
                
                for method_name, method_result in results.items():
                    if method_name == 'traditional':
                        forecast = method_result['ensemble_forecast']
                        weight = 0.3
                    elif method_name == 'foundation':
                        # Use ensemble result if available, otherwise first available
                        if 'foundation_ensemble' in method_result:
                            forecast = method_result['foundation_ensemble']['forecast']
                        else:
                            forecast = list(method_result.values())[0]['forecast']
                        weight = 0.3
                    elif method_name == 'bayesian':
                        forecast = method_result['forecast']
                        weight = 0.2
                    elif method_name == 'sentiment':
                        forecast = method_result['enhanced_forecast']
                        weight = 0.1
                    elif method_name == 'synthetic':
                        forecast = method_result['forecast']
                        weight = 0.1
                    else:
                        continue
                    
                    forecasts.append(forecast)
                    weights.append(weight)
                
                if forecasts:
                    # Normalize weights
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    
                    # Weighted ensemble
                    meta_forecast = np.average(forecasts, axis=0, weights=weights)
                    
                    results['meta_ensemble'] = {
                        'forecast': meta_forecast,
                        'weights': dict(zip(results.keys(), weights)),
                        'individual_forecasts': dict(zip(results.keys(), forecasts)),
                        'model_type': 'Meta Ensemble'
                    }
                    
                    logger.info("Meta-ensemble forecast generated")
            
            except Exception as e:
                logger.error(f"Meta-ensemble failed: {e}")
        
        return results


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
    # Example usage of enhanced GenAI forecasting
    print("Enhanced GenAI Econometric Forecasting Demo")
    print("=" * 50)
    
    # Initialize enhanced forecaster
    forecaster = EconometricForecaster(enable_genai=True)
    
    # Generate sample economic data
    dates = pd.date_range('2020-01-01', '2023-12-01', freq='M')
    np.random.seed(42)
    
    # Create realistic economic time series with trend, seasonality, and noise
    trend = np.linspace(0.02, 0.025, len(dates))
    seasonal = 0.005 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = np.random.normal(0, 0.003, len(dates))
    gdp_growth = trend + seasonal + noise
    
    economic_series = pd.Series(gdp_growth, index=dates, name='gdp_growth')
    
    print(f"Sample data created: {len(economic_series)} observations")
    print(f"Data range: {economic_series.min():.4f} to {economic_series.max():.4f}")
    
    # Test stationarity
    try:
        stationarity = forecaster.check_stationarity(economic_series)
        print(f"\nStationarity test ({stationarity['test_type']}): {stationarity['is_stationary']}")
    except Exception as e:
        print(f"Stationarity test failed: {e}")
    
    # Traditional econometric forecasting
    print("\n1. Traditional Econometric Forecasting")
    try:
        traditional_result = forecaster.generate_ensemble_forecast(economic_series, periods=12)
        print(f"   Traditional ensemble forecast (12 periods): {len(traditional_result['individual_forecasts'])} models")
        print(f"   Mean forecast: {np.mean(traditional_result['ensemble_forecast']):.4f}")
    except Exception as e:
        print(f"   Traditional forecasting failed: {e}")
    
    # Foundation model forecasting
    print("\n2. Foundation Model Forecasting")
    if FOUNDATION_MODELS_AVAILABLE:
        try:
            foundation_result = forecaster.generate_foundation_model_forecast(
                economic_series, periods=12, model_type='ensemble'
            )
            print(f"   Foundation models available: {list(foundation_result.keys())}")
            for model_name, result in foundation_result.items():
                if 'forecast' in result:
                    print(f"   {model_name} mean forecast: {np.mean(result['forecast']):.4f}")
        except Exception as e:
            print(f"   Foundation model forecasting failed: {e}")
    else:
        print("   Foundation models not available")
    
    # Bayesian uncertainty quantification
    print("\n3. Bayesian Uncertainty Quantification")
    if UNCERTAINTY_MODELS_AVAILABLE:
        try:
            bayesian_result = forecaster.generate_bayesian_forecast(
                economic_series, periods=12, include_uncertainty=True
            )
            print(f"   Bayesian forecast mean: {np.mean(bayesian_result['forecast']):.4f}")
            if 'credible_intervals' in bayesian_result:
                print("   Credible intervals included")
        except Exception as e:
            print(f"   Bayesian forecasting failed: {e}")
    else:
        print("   Uncertainty models not available")
    
    # Sentiment-enhanced forecasting
    print("\n4. Sentiment-Enhanced Forecasting")
    if SENTIMENT_ANALYSIS_AVAILABLE:
        try:
            sentiment_result = forecaster.enhance_forecast_with_sentiment(
                economic_series, periods=12
            )
            print(f"   Sentiment score: {sentiment_result['sentiment_score']:.4f}")
            print(f"   Enhanced forecast mean: {np.mean(sentiment_result['enhanced_forecast']):.4f}")
        except Exception as e:
            print(f"   Sentiment-enhanced forecasting failed: {e}")
    else:
        print("   Sentiment analysis not available")
    
    # Monte Carlo scenario generation
    print("\n5. Monte Carlo Scenario Generation")
    if UNCERTAINTY_MODELS_AVAILABLE:
        try:
            mc_result = forecaster.generate_monte_carlo_scenarios(
                economic_series, periods=12, n_scenarios=100
            )
            print(f"   Generated {mc_result['n_scenarios']} Monte Carlo scenarios")
            if 'statistics' in mc_result:
                print("   Scenario statistics calculated")
        except Exception as e:
            print(f"   Monte Carlo simulation failed: {e}")
    else:
        print("   Monte Carlo models not available")
    
    # Comprehensive GenAI forecasting
    print("\n6. Comprehensive GenAI Forecasting")
    try:
        comprehensive_result = forecaster.generate_comprehensive_forecast(
            economic_series, 
            periods=12,
            include_traditional=True,
            include_foundation=True,
            include_bayesian=True,
            include_sentiment=True,
            include_synthetic=False  # Disabled for demo (computationally intensive)
        )
        
        print(f"   Available forecast methods: {list(comprehensive_result.keys())}")
        
        if 'meta_ensemble' in comprehensive_result:
            meta_result = comprehensive_result['meta_ensemble']
            print(f"   Meta-ensemble forecast mean: {np.mean(meta_result['forecast']):.4f}")
            print(f"   Model weights: {meta_result['weights']}")
        
        # Compare forecasts
        print("\n   Forecast Comparison:")
        for method_name, result in comprehensive_result.items():
            if method_name == 'meta_ensemble':
                forecast = result['forecast']
            elif method_name == 'traditional':
                forecast = result['ensemble_forecast']
            elif method_name == 'foundation':
                # Get first available forecast
                first_model = list(result.values())[0]
                forecast = first_model['forecast']
            elif method_name == 'bayesian':
                forecast = result['forecast']
            elif method_name == 'sentiment':
                forecast = result['enhanced_forecast']
            else:
                continue
            
            print(f"     {method_name}: mean={np.mean(forecast):.4f}, std={np.std(forecast):.4f}")
    
    except Exception as e:
        print(f"   Comprehensive forecasting failed: {e}")
    
    print("\n" + "=" * 50)
    print("Enhanced GenAI Econometric Forecasting Demo Complete")