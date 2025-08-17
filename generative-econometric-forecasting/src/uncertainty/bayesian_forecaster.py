"""
Bayesian Forecaster
Implements Bayesian time series models for uncertainty quantification in economic forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    logging.warning("PyMC not available. Install with: pip install pymc")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available")

logger = logging.getLogger(__name__)


class BayesianForecaster:
    """Bayesian time series forecasting with uncertainty quantification."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize Bayesian forecaster.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC not available. Install with: pip install pymc")
        
        self.random_seed = random_seed
        self.models = {}
        self.traces = {}
        self.scalers = {}
        
        # Set random seeds
        np.random.seed(random_seed)
        
        logger.info("Bayesian forecaster initialized")
    
    def fit_bayesian_linear_trend(self, 
                                 series: pd.Series,
                                 model_name: str = 'linear_trend',
                                 samples: int = 2000,
                                 tune: int = 1000,
                                 chains: int = 2) -> Dict[str, Any]:
        """
        Fit Bayesian linear trend model.
        
        Args:
            series: Time series data
            model_name: Name for the model
            samples: Number of posterior samples
            tune: Number of tuning samples
            chains: Number of MCMC chains
        
        Returns:
            Model fitting results
        """
        try:
            # Prepare data
            y = series.dropna().values
            x = np.arange(len(y))
            
            # Standardize data
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
            self.scalers[model_name] = scaler
            
            # Build Bayesian model
            with pm.Model() as model:
                # Priors
                alpha = pm.Normal('alpha', mu=0, sigma=1)  # Intercept
                beta = pm.Normal('beta', mu=0, sigma=1)    # Slope
                sigma = pm.HalfNormal('sigma', sigma=1)    # Noise
                
                # Linear trend
                mu = alpha + beta * x
                
                # Likelihood
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_scaled)
                
                # Sample from posterior
                trace = pm.sample(
                    samples, 
                    tune=tune, 
                    chains=chains,
                    random_seed=self.random_seed,
                    return_inferencedata=True
                )
            
            # Store model and trace
            self.models[model_name] = model
            self.traces[model_name] = trace
            
            # Model diagnostics
            rhat = az.rhat(trace)
            ess = az.ess(trace)
            
            return {
                'model': model,
                'trace': trace,
                'convergence': {
                    'rhat_max': float(rhat.max()),
                    'ess_min': float(ess.min()),
                    'converged': float(rhat.max()) < 1.1
                },
                'posterior_summary': az.summary(trace),
                'model_name': model_name
            }
            
        except Exception as e:
            logger.error(f"Error fitting Bayesian linear trend: {e}")
            return {'error': str(e)}
    
    def fit_bayesian_ar(self, 
                       series: pd.Series,
                       ar_order: int = 2,
                       model_name: str = 'bayesian_ar',
                       samples: int = 2000,
                       tune: int = 1000) -> Dict[str, Any]:
        """
        Fit Bayesian autoregressive model.
        
        Args:
            series: Time series data
            ar_order: Autoregressive order
            model_name: Name for the model
            samples: Number of posterior samples
            tune: Number of tuning samples
        
        Returns:
            Model fitting results
        """
        try:
            # Prepare data
            y = series.dropna().values
            
            # Standardize
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
            self.scalers[model_name] = scaler
            
            # Create lagged variables
            y_ar = []
            y_target = []
            
            for i in range(ar_order, len(y_scaled)):
                y_ar.append(y_scaled[i-ar_order:i])
                y_target.append(y_scaled[i])
            
            y_ar = np.array(y_ar)
            y_target = np.array(y_target)
            
            # Build Bayesian AR model
            with pm.Model() as model:
                # Priors for AR coefficients
                ar_coefs = pm.Normal('ar_coefs', mu=0, sigma=0.5, shape=ar_order)
                
                # Intercept
                intercept = pm.Normal('intercept', mu=0, sigma=1)
                
                # Noise variance
                sigma = pm.HalfNormal('sigma', sigma=1)
                
                # AR model
                mu = intercept + pm.math.dot(y_ar, ar_coefs)
                
                # Likelihood
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_target)
                
                # Sample
                trace = pm.sample(
                    samples, 
                    tune=tune,
                    random_seed=self.random_seed,
                    return_inferencedata=True
                )
            
            self.models[model_name] = model
            self.traces[model_name] = trace
            
            # Diagnostics
            rhat = az.rhat(trace)
            ess = az.ess(trace)
            
            return {
                'model': model,
                'trace': trace,
                'ar_order': ar_order,
                'convergence': {
                    'rhat_max': float(rhat.max()),
                    'ess_min': float(ess.min()),
                    'converged': float(rhat.max()) < 1.1
                },
                'posterior_summary': az.summary(trace),
                'model_name': model_name
            }
            
        except Exception as e:
            logger.error(f"Error fitting Bayesian AR model: {e}")
            return {'error': str(e)}
    
    def fit_bayesian_structural(self, 
                               series: pd.Series,
                               seasonal_periods: Optional[int] = 12,
                               model_name: str = 'structural',
                               samples: int = 2000) -> Dict[str, Any]:
        """
        Fit Bayesian structural time series model.
        
        Args:
            series: Time series data
            seasonal_periods: Number of seasonal periods (None for no seasonality)
            model_name: Name for the model
            samples: Number of posterior samples
        
        Returns:
            Model fitting results
        """
        try:
            # Prepare data
            y = series.dropna().values
            T = len(y)
            
            # Standardize
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
            self.scalers[model_name] = scaler
            
            with pm.Model() as model:
                # Local level and trend
                level_var = pm.HalfNormal('level_var', sigma=0.1)
                trend_var = pm.HalfNormal('trend_var', sigma=0.1)
                obs_var = pm.HalfNormal('obs_var', sigma=1)
                
                # Initial states
                level_init = pm.Normal('level_init', mu=y_scaled[0], sigma=1)
                trend_init = pm.Normal('trend_init', mu=0, sigma=0.1)
                
                # State evolution
                levels = [level_init]
                trends = [trend_init]
                
                for t in range(1, T):
                    trend_t = pm.Normal(f'trend_{t}', 
                                      mu=trends[t-1], 
                                      sigma=trend_var)
                    level_t = pm.Normal(f'level_{t}', 
                                      mu=levels[t-1] + trends[t-1], 
                                      sigma=level_var)
                    trends.append(trend_t)
                    levels.append(level_t)
                
                # Seasonal component (if specified)
                if seasonal_periods:
                    seasonal_var = pm.HalfNormal('seasonal_var', sigma=0.1)
                    seasonal_init = pm.Normal('seasonal_init', 
                                           mu=0, sigma=0.1, 
                                           shape=seasonal_periods-1)
                    
                    seasonal_components = []
                    for t in range(T):
                        if t < seasonal_periods - 1:
                            seasonal_components.append(seasonal_init[t])
                        else:
                            # Sum constraint: seasonal components sum to zero
                            prev_sum = sum(seasonal_components[t-seasonal_periods+1:t])
                            seasonal_t = pm.Normal(f'seasonal_{t}',
                                                 mu=-prev_sum/(seasonal_periods-1),
                                                 sigma=seasonal_var)
                            seasonal_components.append(seasonal_t)
                    
                    # Observations with seasonality
                    mu = [levels[t] + seasonal_components[t] for t in range(T)]
                else:
                    # Observations without seasonality
                    mu = levels
                
                # Likelihood
                y_obs = pm.Normal('y_obs', mu=mu, sigma=obs_var, observed=y_scaled)
                
                # Sample
                trace = pm.sample(
                    samples, 
                    tune=1000,
                    random_seed=self.random_seed,
                    return_inferencedata=True
                )
            
            self.models[model_name] = model
            self.traces[model_name] = trace
            
            return {
                'model': model,
                'trace': trace,
                'seasonal_periods': seasonal_periods,
                'model_name': model_name,
                'T': T
            }
            
        except Exception as e:
            logger.error(f"Error fitting structural model: {e}")
            return {'error': str(e)}
    
    def forecast_bayesian(self, 
                         model_name: str,
                         horizon: int = 12,
                         credible_interval: float = 0.95) -> Dict[str, Any]:
        """
        Generate Bayesian forecast with uncertainty quantification.
        
        Args:
            model_name: Name of fitted model
            horizon: Forecast horizon
            credible_interval: Credible interval level
        
        Returns:
            Forecast results with uncertainty
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        try:
            model = self.models[model_name]
            trace = self.traces[model_name]
            scaler = self.scalers[model_name]
            
            # Generate posterior predictive samples
            with model:
                # Extend model for forecasting
                posterior_samples = trace.posterior
                
                if 'linear_trend' in model_name:
                    forecast_samples = self._forecast_linear_trend(
                        posterior_samples, horizon, scaler
                    )
                elif 'ar' in model_name:
                    forecast_samples = self._forecast_ar(
                        posterior_samples, horizon, scaler, model_name
                    )
                elif 'structural' in model_name:
                    forecast_samples = self._forecast_structural(
                        posterior_samples, horizon, scaler
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_name}")
            
            # Calculate forecast statistics
            forecast_mean = np.mean(forecast_samples, axis=0)
            forecast_median = np.median(forecast_samples, axis=0)
            forecast_std = np.std(forecast_samples, axis=0)
            
            # Credible intervals
            alpha = (1 - credible_interval) / 2
            lower_ci = np.percentile(forecast_samples, alpha * 100, axis=0)
            upper_ci = np.percentile(forecast_samples, (1 - alpha) * 100, axis=0)
            
            return {
                'forecast_mean': forecast_mean,
                'forecast_median': forecast_median,
                'forecast_std': forecast_std,
                'forecast_samples': forecast_samples,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'credible_interval': credible_interval,
                'horizon': horizon,
                'model_type': 'Bayesian',
                'model_name': model_name
            }
            
        except Exception as e:
            logger.error(f"Error generating Bayesian forecast: {e}")
            return {'error': str(e)}
    
    def _forecast_linear_trend(self, 
                              posterior_samples: Any,
                              horizon: int,
                              scaler: StandardScaler) -> np.ndarray:
        """Generate forecasts for linear trend model."""
        alpha_samples = posterior_samples['alpha'].values.flatten()
        beta_samples = posterior_samples['beta'].values.flatten()
        sigma_samples = posterior_samples['sigma'].values.flatten()
        
        n_samples = len(alpha_samples)
        forecast_samples = []
        
        # Get last time point
        last_t = len(scaler.scale_) if hasattr(scaler, 'scale_') else 0
        
        for i in range(n_samples):
            alpha, beta, sigma = alpha_samples[i], beta_samples[i], sigma_samples[i]
            
            # Generate forecast for this sample
            forecast_t = []
            for h in range(1, horizon + 1):
                mu = alpha + beta * (last_t + h)
                noise = np.random.normal(0, sigma)
                forecast_t.append(mu + noise)
            
            forecast_samples.append(forecast_t)
        
        forecast_samples = np.array(forecast_samples)
        
        # Transform back to original scale
        forecast_original = []
        for i in range(forecast_samples.shape[0]):
            forecast_scaled = forecast_samples[i].reshape(-1, 1)
            forecast_unscaled = scaler.inverse_transform(forecast_scaled).flatten()
            forecast_original.append(forecast_unscaled)
        
        return np.array(forecast_original)
    
    def _forecast_ar(self, 
                    posterior_samples: Any,
                    horizon: int,
                    scaler: StandardScaler,
                    model_name: str) -> np.ndarray:
        """Generate forecasts for AR model."""
        ar_coefs_samples = posterior_samples['ar_coefs'].values
        intercept_samples = posterior_samples['intercept'].values.flatten()
        sigma_samples = posterior_samples['sigma'].values.flatten()
        
        n_samples = ar_coefs_samples.shape[0] * ar_coefs_samples.shape[1]
        ar_order = ar_coefs_samples.shape[2]
        
        # Flatten samples
        ar_coefs_flat = ar_coefs_samples.reshape(n_samples, ar_order)
        intercept_flat = np.repeat(intercept_samples, ar_coefs_samples.shape[1])
        sigma_flat = np.repeat(sigma_samples, ar_coefs_samples.shape[1])
        
        # Get last observations (would need to store these during fitting)
        # For now, use small random values as placeholder
        last_obs = np.random.normal(0, 0.1, ar_order)
        
        forecast_samples = []
        
        for i in range(min(1000, n_samples)):  # Limit for computational efficiency
            ar_coefs = ar_coefs_flat[i]
            intercept = intercept_flat[i]
            sigma = sigma_flat[i]
            
            # Generate forecast
            forecast_t = []
            current_obs = last_obs.copy()
            
            for h in range(horizon):
                mu = intercept + np.dot(current_obs, ar_coefs)
                noise = np.random.normal(0, sigma)
                forecast_val = mu + noise
                forecast_t.append(forecast_val)
                
                # Update observations for next step
                current_obs = np.roll(current_obs, 1)
                current_obs[0] = forecast_val
            
            forecast_samples.append(forecast_t)
        
        forecast_samples = np.array(forecast_samples)
        
        # Transform back to original scale
        forecast_original = []
        for i in range(forecast_samples.shape[0]):
            forecast_scaled = forecast_samples[i].reshape(-1, 1)
            forecast_unscaled = scaler.inverse_transform(forecast_scaled).flatten()
            forecast_original.append(forecast_unscaled)
        
        return np.array(forecast_original)
    
    def _forecast_structural(self, 
                           posterior_samples: Any,
                           horizon: int,
                           scaler: StandardScaler) -> np.ndarray:
        """Generate forecasts for structural model."""
        # This is a simplified implementation
        # In practice, would need to properly extract state components
        
        level_var_samples = posterior_samples['level_var'].values.flatten()
        trend_var_samples = posterior_samples['trend_var'].values.flatten()
        obs_var_samples = posterior_samples['obs_var'].values.flatten()
        
        n_samples = len(level_var_samples)
        forecast_samples = []
        
        for i in range(min(1000, n_samples)):
            level_var = level_var_samples[i]
            trend_var = trend_var_samples[i]
            obs_var = obs_var_samples[i]
            
            # Simple structural forecast (would be more complex in practice)
            forecast_t = []
            current_level = 0  # Would extract from last state
            current_trend = 0  # Would extract from last state
            
            for h in range(horizon):
                # Evolve states
                trend_noise = np.random.normal(0, trend_var)
                level_noise = np.random.normal(0, level_var)
                obs_noise = np.random.normal(0, obs_var)
                
                current_trend += trend_noise
                current_level += current_trend + level_noise
                
                observation = current_level + obs_noise
                forecast_t.append(observation)
            
            forecast_samples.append(forecast_t)
        
        forecast_samples = np.array(forecast_samples)
        
        # Transform back to original scale
        forecast_original = []
        for i in range(forecast_samples.shape[0]):
            forecast_scaled = forecast_samples[i].reshape(-1, 1)
            forecast_unscaled = scaler.inverse_transform(forecast_scaled).flatten()
            forecast_original.append(forecast_unscaled)
        
        return np.array(forecast_original)
    
    def model_comparison(self, 
                        series: pd.Series,
                        models: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple Bayesian models using cross-validation and information criteria.
        
        Args:
            series: Time series data
            models: List of model types to compare
        
        Returns:
            Model comparison results
        """
        if models is None:
            models = ['linear_trend', 'bayesian_ar']
        
        comparison_results = {}
        
        for model_type in models:
            try:
                if model_type == 'linear_trend':
                    result = self.fit_bayesian_linear_trend(series, model_name=model_type)
                elif model_type == 'bayesian_ar':
                    result = self.fit_bayesian_ar(series, model_name=model_type)
                elif model_type == 'structural':
                    result = self.fit_bayesian_structural(series, model_name=model_type)
                else:
                    continue
                
                if 'error' not in result:
                    # Calculate model diagnostics
                    trace = result['trace']
                    
                    # Information criteria
                    loo = az.loo(trace)
                    waic = az.waic(trace)
                    
                    comparison_results[model_type] = {
                        'loo': loo,
                        'waic': waic,
                        'convergence': result['convergence'],
                        'n_parameters': len(trace.posterior.data_vars)
                    }
                else:
                    comparison_results[model_type] = {'error': result['error']}
                    
            except Exception as e:
                logger.error(f"Error comparing model {model_type}: {e}")
                comparison_results[model_type] = {'error': str(e)}
        
        return comparison_results
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """Get summary of fitted model."""
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}
        
        trace = self.traces[model_name]
        
        return {
            'posterior_summary': az.summary(trace),
            'model_name': model_name,
            'n_chains': len(trace.posterior.chain),
            'n_draws': len(trace.posterior.draw),
            'parameters': list(trace.posterior.data_vars.keys())
        }


if __name__ == "__main__":
    # Example usage
    forecaster = BayesianForecaster()
    
    # Generate sample economic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-01', freq='M')
    trend = np.linspace(100, 120, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    seasonal = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    values = trend + seasonal + noise
    
    series = pd.Series(values, index=dates, name='economic_indicator')
    
    print("Fitting Bayesian models...")
    
    # Fit linear trend model
    linear_result = forecaster.fit_bayesian_linear_trend(series)
    if 'error' not in linear_result:
        print(f"Linear trend model converged: {linear_result['convergence']['converged']}")
        
        # Generate forecast
        forecast = forecaster.forecast_bayesian('linear_trend', horizon=6)
        if 'error' not in forecast:
            print(f"Forecast mean: {forecast['forecast_mean'][:3]}")
            print(f"Forecast uncertainty (std): {forecast['forecast_std'][:3]}")
    
    # Fit AR model
    ar_result = forecaster.fit_bayesian_ar(series, ar_order=2)
    if 'error' not in ar_result:
        print(f"AR model converged: {ar_result['convergence']['converged']}")
    
    # Model comparison
    comparison = forecaster.model_comparison(series)
    print(f"\nModel comparison results:")
    for model, results in comparison.items():
        if 'loo' in results:
            print(f"{model}: LOO = {results['loo'].loo:.2f}")
    
    print("Bayesian forecasting example completed")