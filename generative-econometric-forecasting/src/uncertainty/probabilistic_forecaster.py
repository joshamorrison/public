"""
Probabilistic Forecaster
Implements probabilistic forecasting methods for distribution-based predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal, MixtureSameFamily, Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available for deep probabilistic models")

logger = logging.getLogger(__name__)


class ProbabilisticDistribution:
    """Base class for probabilistic distributions."""
    
    def __init__(self, distribution_type: str, parameters: Dict[str, float]):
        self.distribution_type = distribution_type
        self.parameters = parameters
        self._create_distribution()
    
    def _create_distribution(self):
        """Create scipy distribution object."""
        if self.distribution_type == 'normal':
            self.distribution = stats.norm(
                loc=self.parameters['mean'],
                scale=self.parameters['std']
            )
        elif self.distribution_type == 'student_t':
            self.distribution = stats.t(
                df=self.parameters['df'],
                loc=self.parameters['mean'],
                scale=self.parameters['std']
            )
        elif self.distribution_type == 'skew_normal':
            self.distribution = stats.skewnorm(
                a=self.parameters['skewness'],
                loc=self.parameters['mean'],
                scale=self.parameters['std']
            )
        elif self.distribution_type == 'gamma':
            self.distribution = stats.gamma(
                a=self.parameters['shape'],
                scale=self.parameters['scale'],
                loc=self.parameters.get('loc', 0)
            )
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function."""
        return self.distribution.pdf(x)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        return self.distribution.cdf(x)
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Percent point function (inverse CDF)."""
        return self.distribution.ppf(q)
    
    def sample(self, size: int = 1) -> np.ndarray:
        """Generate random samples."""
        return self.distribution.rvs(size=size)
    
    def confidence_interval(self, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval."""
        alpha = (1 - confidence_level) / 2
        return self.ppf(alpha), self.ppf(1 - alpha)
    
    def mean(self) -> float:
        """Expected value."""
        return self.distribution.mean()
    
    def variance(self) -> float:
        """Variance."""
        return self.distribution.var()


class DeepProbabilisticNetwork(nn.Module):
    """Deep neural network for probabilistic forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 2):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for deep probabilistic models")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Network layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
        
        # Separate heads for mean and log variance
        self.mean_head = nn.Linear(hidden_size, 1)
        self.logvar_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Get features
        features = self.layers[:-1](x)  # All layers except last
        
        # Get mean and log variance
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        
        # Ensure positive variance
        std = torch.exp(0.5 * logvar)
        
        return mean, std


class ProbabilisticForecaster:
    """Probabilistic forecasting with distribution-based predictions."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize probabilistic forecaster.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.models = {}
        self.scalers = {}
        self.distributions = {}
        
        # Forecasting methods
        self.methods = {
            'quantile_regression': self._fit_quantile_regression,
            'distributional_regression': self._fit_distributional_regression,
            'deep_probabilistic': self._fit_deep_probabilistic,
            'mixture_model': self._fit_mixture_model
        }
        
        logger.info("Probabilistic forecaster initialized")
    
    def fit_probabilistic_model(self, 
                               series: pd.Series,
                               method: str = 'quantile_regression',
                               model_name: str = 'prob_model',
                               **kwargs) -> Dict[str, Any]:
        """
        Fit probabilistic forecasting model.
        
        Args:
            series: Time series data
            method: Forecasting method
            model_name: Name for the model
            **kwargs: Method-specific parameters
        
        Returns:
            Model fitting results
        """
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")
        
        try:
            # Prepare data
            y = series.dropna().values
            
            # Standardize data
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
            self.scalers[model_name] = scaler
            
            # Fit model using specified method
            result = self.methods[method](y_scaled, model_name, **kwargs)
            
            # Store model
            self.models[model_name] = result
            
            logger.info(f"Fitted {method} model: {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error fitting probabilistic model: {e}")
            return {'error': str(e)}
    
    def _fit_quantile_regression(self, 
                                y: np.ndarray,
                                model_name: str,
                                quantiles: List[float] = None,
                                lookback: int = 12) -> Dict[str, Any]:
        """Fit quantile regression model."""
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        from sklearn.linear_model import QuantileRegressor
        
        # Create lagged features
        X, y_target = self._create_lagged_features(y, lookback)
        
        # Fit quantile regressors
        quantile_models = {}
        
        for q in quantiles:
            model = QuantileRegressor(quantile=q, alpha=0.01)
            model.fit(X, y_target)
            quantile_models[q] = model
        
        return {
            'method': 'quantile_regression',
            'quantile_models': quantile_models,
            'quantiles': quantiles,
            'lookback': lookback,
            'model_name': model_name
        }
    
    def _fit_distributional_regression(self, 
                                     y: np.ndarray,
                                     model_name: str,
                                     distribution: str = 'normal',
                                     lookback: int = 12) -> Dict[str, Any]:
        """Fit distributional regression model."""
        from sklearn.linear_model import LinearRegression
        
        # Create features
        X, y_target = self._create_lagged_features(y, lookback)
        
        # Fit mean model
        mean_model = LinearRegression()
        mean_model.fit(X, y_target)
        
        # Calculate residuals
        y_pred_mean = mean_model.predict(X)
        residuals = y_target - y_pred_mean
        
        # Fit variance model (predict log variance to ensure positivity)
        residual_squared = residuals ** 2
        variance_model = LinearRegression()
        variance_model.fit(X, np.log(residual_squared + 1e-6))
        
        # Estimate distribution parameters from residuals
        if distribution == 'normal':
            dist_params = {
                'mean': 0,  # Will be predicted
                'std': np.std(residuals)
            }
        elif distribution == 'student_t':
            # Estimate degrees of freedom
            df_estimated = 2 / (stats.kurtosis(residuals, fisher=False) - 3) if stats.kurtosis(residuals, fisher=False) > 3 else 30
            dist_params = {
                'df': max(df_estimated, 3),
                'mean': 0,
                'std': np.std(residuals)
            }
        else:
            raise ValueError(f"Distribution {distribution} not supported")
        
        return {
            'method': 'distributional_regression',
            'mean_model': mean_model,
            'variance_model': variance_model,
            'distribution': distribution,
            'dist_params': dist_params,
            'lookback': lookback,
            'model_name': model_name
        }
    
    def _fit_deep_probabilistic(self, 
                              y: np.ndarray,
                              model_name: str,
                              lookback: int = 12,
                              epochs: int = 100) -> Dict[str, Any]:
        """Fit deep probabilistic neural network."""
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        # Create features
        X, y_target = self._create_lagged_features(y, lookback)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y_target)
        
        # Initialize model
        model = DeepProbabilisticNetwork(input_size=X.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        losses = []
        model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            mean_pred, std_pred = model(X_tensor)
            
            # Negative log-likelihood loss
            dist = Normal(mean_pred.squeeze(), std_pred.squeeze())
            loss = -dist.log_prob(y_tensor).mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        model.eval()
        
        return {
            'method': 'deep_probabilistic',
            'model': model,
            'losses': losses,
            'lookback': lookback,
            'model_name': model_name
        }
    
    def _fit_mixture_model(self, 
                          y: np.ndarray,
                          model_name: str,
                          n_components: int = 3,
                          lookback: int = 12) -> Dict[str, Any]:
        """Fit mixture model for multimodal distributions."""
        from sklearn.mixture import GaussianMixture
        from sklearn.linear_model import LinearRegression
        
        # Create features
        X, y_target = self._create_lagged_features(y, lookback)
        
        # Fit regression model first
        base_model = LinearRegression()
        base_model.fit(X, y_target)
        
        # Get residuals
        y_pred = base_model.predict(X)
        residuals = y_target - y_pred
        
        # Fit mixture model to residuals
        mixture_model = GaussianMixture(
            n_components=n_components,
            random_state=self.random_seed
        )
        mixture_model.fit(residuals.reshape(-1, 1))
        
        return {
            'method': 'mixture_model',
            'base_model': base_model,
            'mixture_model': mixture_model,
            'n_components': n_components,
            'lookback': lookback,
            'model_name': model_name
        }
    
    def _create_lagged_features(self, 
                              y: np.ndarray, 
                              lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create lagged features for time series."""
        X, y_target = [], []
        
        for i in range(lookback, len(y)):
            X.append(y[i-lookback:i])
            y_target.append(y[i])
        
        return np.array(X), np.array(y_target)
    
    def forecast_probabilistic(self, 
                             model_name: str,
                             horizon: int = 12,
                             confidence_levels: List[float] = None) -> Dict[str, Any]:
        """
        Generate probabilistic forecast.
        
        Args:
            model_name: Name of fitted model
            horizon: Forecast horizon
            confidence_levels: Confidence levels for intervals
        
        Returns:
            Probabilistic forecast results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if confidence_levels is None:
            confidence_levels = [0.50, 0.80, 0.90, 0.95]
        
        model_config = self.models[model_name]
        method = model_config['method']
        
        try:
            if method == 'quantile_regression':
                return self._forecast_quantile_regression(model_config, horizon, confidence_levels)
            elif method == 'distributional_regression':
                return self._forecast_distributional_regression(model_config, horizon, confidence_levels)
            elif method == 'deep_probabilistic':
                return self._forecast_deep_probabilistic(model_config, horizon, confidence_levels)
            elif method == 'mixture_model':
                return self._forecast_mixture_model(model_config, horizon, confidence_levels)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            logger.error(f"Error generating probabilistic forecast: {e}")
            return {'error': str(e)}
    
    def _forecast_quantile_regression(self, 
                                    model_config: Dict[str, Any],
                                    horizon: int,
                                    confidence_levels: List[float]) -> Dict[str, Any]:
        """Generate forecast using quantile regression."""
        quantile_models = model_config['quantile_models']
        quantiles = model_config['quantiles']
        lookback = model_config['lookback']
        
        # Would need last observations for forecasting
        # For demo, using dummy data
        last_obs = np.random.normal(0, 0.1, lookback)
        
        forecasts_by_quantile = {}
        
        for q, model in quantile_models.items():
            forecast_q = []
            current_obs = last_obs.copy()
            
            for h in range(horizon):
                # Predict next value
                X_next = current_obs.reshape(1, -1)
                pred = model.predict(X_next)[0]
                forecast_q.append(pred)
                
                # Update observations
                current_obs = np.roll(current_obs, -1)
                current_obs[-1] = pred
            
            forecasts_by_quantile[q] = np.array(forecast_q)
        
        # Transform back to original scale
        scaler = self.scalers[model_config['model_name']]
        for q in forecasts_by_quantile:
            forecasts_by_quantile[q] = scaler.inverse_transform(
                forecasts_by_quantile[q].reshape(-1, 1)
            ).flatten()
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            alpha = (1 - conf_level) / 2
            lower_q = alpha
            upper_q = 1 - alpha
            
            # Find closest quantiles
            lower_quantile = min(quantiles, key=lambda x: abs(x - lower_q))
            upper_quantile = min(quantiles, key=lambda x: abs(x - upper_q))
            
            confidence_intervals[conf_level] = {
                'lower': forecasts_by_quantile[lower_quantile],
                'upper': forecasts_by_quantile[upper_quantile]
            }
        
        # Use median as point forecast
        median_forecast = forecasts_by_quantile.get(0.5, forecasts_by_quantile[quantiles[len(quantiles)//2]])
        
        return {
            'forecast': median_forecast,
            'forecasts_by_quantile': forecasts_by_quantile,
            'confidence_intervals': confidence_intervals,
            'method': 'quantile_regression',
            'horizon': horizon
        }
    
    def _forecast_distributional_regression(self, 
                                          model_config: Dict[str, Any],
                                          horizon: int,
                                          confidence_levels: List[float]) -> Dict[str, Any]:
        """Generate forecast using distributional regression."""
        mean_model = model_config['mean_model']
        variance_model = model_config['variance_model']
        distribution = model_config['distribution']
        dist_params = model_config['dist_params']
        lookback = model_config['lookback']
        
        # Generate forecast distributions
        forecast_distributions = []
        last_obs = np.random.normal(0, 0.1, lookback)  # Dummy data
        
        current_obs = last_obs.copy()
        
        for h in range(horizon):
            X_next = current_obs.reshape(1, -1)
            
            # Predict mean and variance
            mean_pred = mean_model.predict(X_next)[0]
            log_var_pred = variance_model.predict(X_next)[0]
            var_pred = np.exp(log_var_pred)
            
            # Create distribution
            if distribution == 'normal':
                dist = ProbabilisticDistribution('normal', {
                    'mean': mean_pred,
                    'std': np.sqrt(var_pred)
                })
            elif distribution == 'student_t':
                dist = ProbabilisticDistribution('student_t', {
                    'df': dist_params['df'],
                    'mean': mean_pred,
                    'std': np.sqrt(var_pred)
                })
            
            forecast_distributions.append(dist)
            
            # Update observations
            current_obs = np.roll(current_obs, -1)
            current_obs[-1] = mean_pred
        
        # Extract point forecasts and confidence intervals
        forecasts = np.array([dist.mean() for dist in forecast_distributions])
        confidence_intervals = {}
        
        for conf_level in confidence_levels:
            lower_bounds = []
            upper_bounds = []
            
            for dist in forecast_distributions:
                lower, upper = dist.confidence_interval(conf_level)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
            
            confidence_intervals[conf_level] = {
                'lower': np.array(lower_bounds),
                'upper': np.array(upper_bounds)
            }
        
        # Transform back to original scale
        scaler = self.scalers[model_config['model_name']]
        forecasts = scaler.inverse_transform(forecasts.reshape(-1, 1)).flatten()
        
        for conf_level in confidence_intervals:
            confidence_intervals[conf_level]['lower'] = scaler.inverse_transform(
                confidence_intervals[conf_level]['lower'].reshape(-1, 1)
            ).flatten()
            confidence_intervals[conf_level]['upper'] = scaler.inverse_transform(
                confidence_intervals[conf_level]['upper'].reshape(-1, 1)
            ).flatten()
        
        return {
            'forecast': forecasts,
            'forecast_distributions': forecast_distributions,
            'confidence_intervals': confidence_intervals,
            'method': 'distributional_regression',
            'horizon': horizon
        }
    
    def _forecast_deep_probabilistic(self, 
                                   model_config: Dict[str, Any],
                                   horizon: int,
                                   confidence_levels: List[float]) -> Dict[str, Any]:
        """Generate forecast using deep probabilistic model."""
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        model = model_config['model']
        lookback = model_config['lookback']
        
        model.eval()
        
        # Generate forecasts
        forecasts_mean = []
        forecasts_std = []
        
        last_obs = np.random.normal(0, 0.1, lookback)  # Dummy data
        current_obs = last_obs.copy()
        
        with torch.no_grad():
            for h in range(horizon):
                X_next = torch.FloatTensor(current_obs).unsqueeze(0)
                mean_pred, std_pred = model(X_next)
                
                forecasts_mean.append(mean_pred.item())
                forecasts_std.append(std_pred.item())
                
                # Update observations
                current_obs = np.roll(current_obs, -1)
                current_obs[-1] = mean_pred.item()
        
        forecasts_mean = np.array(forecasts_mean)
        forecasts_std = np.array(forecasts_std)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            
            confidence_intervals[conf_level] = {
                'lower': forecasts_mean - z_score * forecasts_std,
                'upper': forecasts_mean + z_score * forecasts_std
            }
        
        # Transform back to original scale
        scaler = self.scalers[model_config['model_name']]
        forecasts_mean = scaler.inverse_transform(forecasts_mean.reshape(-1, 1)).flatten()
        
        for conf_level in confidence_intervals:
            confidence_intervals[conf_level]['lower'] = scaler.inverse_transform(
                confidence_intervals[conf_level]['lower'].reshape(-1, 1)
            ).flatten()
            confidence_intervals[conf_level]['upper'] = scaler.inverse_transform(
                confidence_intervals[conf_level]['upper'].reshape(-1, 1)
            ).flatten()
        
        return {
            'forecast': forecasts_mean,
            'forecast_std': forecasts_std,
            'confidence_intervals': confidence_intervals,
            'method': 'deep_probabilistic',
            'horizon': horizon
        }
    
    def _forecast_mixture_model(self, 
                              model_config: Dict[str, Any],
                              horizon: int,
                              confidence_levels: List[float]) -> Dict[str, Any]:
        """Generate forecast using mixture model."""
        base_model = model_config['base_model']
        mixture_model = model_config['mixture_model']
        lookback = model_config['lookback']
        
        # Generate base forecasts
        forecasts = []
        last_obs = np.random.normal(0, 0.1, lookback)  # Dummy data
        current_obs = last_obs.copy()
        
        for h in range(horizon):
            X_next = current_obs.reshape(1, -1)
            base_pred = base_model.predict(X_next)[0]
            
            # Sample from mixture distribution for uncertainty
            mixture_sample = mixture_model.sample(1)[0][0]
            final_pred = base_pred + mixture_sample
            
            forecasts.append(final_pred)
            
            # Update observations
            current_obs = np.roll(current_obs, -1)
            current_obs[-1] = final_pred
        
        forecasts = np.array(forecasts)
        
        # Generate confidence intervals using mixture components
        confidence_intervals = {}
        for conf_level in confidence_levels:
            # Simplified: use mixture std for intervals
            mixture_std = np.sqrt(np.sum(mixture_model.weights_ * 
                                       (mixture_model.means_.flatten() ** 2 + 
                                        mixture_model.covariances_.flatten())))
            
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            
            confidence_intervals[conf_level] = {
                'lower': forecasts - z_score * mixture_std,
                'upper': forecasts + z_score * mixture_std
            }
        
        # Transform back to original scale
        scaler = self.scalers[model_config['model_name']]
        forecasts = scaler.inverse_transform(forecasts.reshape(-1, 1)).flatten()
        
        for conf_level in confidence_intervals:
            confidence_intervals[conf_level]['lower'] = scaler.inverse_transform(
                confidence_intervals[conf_level]['lower'].reshape(-1, 1)
            ).flatten()
            confidence_intervals[conf_level]['upper'] = scaler.inverse_transform(
                confidence_intervals[conf_level]['upper'].reshape(-1, 1)
            ).flatten()
        
        return {
            'forecast': forecasts,
            'confidence_intervals': confidence_intervals,
            'method': 'mixture_model',
            'horizon': horizon
        }
    
    def evaluate_probabilistic_forecast(self, 
                                      model_name: str,
                                      test_series: pd.Series,
                                      horizon: int = 12) -> Dict[str, Any]:
        """
        Evaluate probabilistic forecast performance.
        
        Args:
            model_name: Name of model to evaluate
            test_series: Test data
            horizon: Forecast horizon
        
        Returns:
            Evaluation metrics
        """
        # Generate forecast
        forecast_result = self.forecast_probabilistic(model_name, horizon)
        
        if 'error' in forecast_result:
            return forecast_result
        
        # Extract actuals and predictions
        actual_values = test_series.iloc[:horizon].values
        forecast_values = forecast_result['forecast'][:len(actual_values)]
        
        # Point forecast metrics
        mae = mean_absolute_error(actual_values, forecast_values)
        mse = mean_squared_error(actual_values, forecast_values)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
        
        # Probabilistic metrics
        confidence_intervals = forecast_result.get('confidence_intervals', {})
        coverage_rates = {}
        interval_widths = {}
        
        for conf_level, intervals in confidence_intervals.items():
            lower = intervals['lower'][:len(actual_values)]
            upper = intervals['upper'][:len(actual_values)]
            
            # Coverage rate
            coverage = np.mean((actual_values >= lower) & (actual_values <= upper))
            coverage_rates[conf_level] = coverage
            
            # Average interval width
            interval_widths[conf_level] = np.mean(upper - lower)
        
        # Continuous Ranked Probability Score (if distributions available)
        crps = None
        if 'forecast_distributions' in forecast_result:
            crps_values = []
            distributions = forecast_result['forecast_distributions']
            
            for i, (actual, dist) in enumerate(zip(actual_values, distributions[:len(actual_values)])):
                # Simplified CRPS calculation
                samples = dist.sample(1000)
                crps_val = np.mean(np.abs(samples - actual)) - 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
                crps_values.append(crps_val)
            
            crps = np.mean(crps_values)
        
        return {
            'point_forecast_metrics': {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape
            },
            'probabilistic_metrics': {
                'coverage_rates': coverage_rates,
                'interval_widths': interval_widths,
                'crps': crps
            },
            'forecast_horizon': len(actual_values),
            'model_name': model_name
        }
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """Get summary of probabilistic model."""
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}
        
        model_config = self.models[model_name]
        
        summary = {
            'model_name': model_name,
            'method': model_config['method'],
            'parameters': {}
        }
        
        # Add method-specific information
        if 'quantiles' in model_config:
            summary['parameters']['quantiles'] = model_config['quantiles']
        if 'lookback' in model_config:
            summary['parameters']['lookback'] = model_config['lookback']
        if 'distribution' in model_config:
            summary['parameters']['distribution'] = model_config['distribution']
        
        return summary


if __name__ == "__main__":
    # Example usage
    forecaster = ProbabilisticForecaster()
    
    # Generate sample economic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-01', freq='M')
    trend = np.linspace(100, 120, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    seasonal = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    values = trend + seasonal + noise
    
    series = pd.Series(values, index=dates, name='economic_indicator')
    
    print("Fitting probabilistic models...")
    
    # Fit quantile regression
    quantile_result = forecaster.fit_probabilistic_model(
        series, 
        method='quantile_regression',
        model_name='quantile_model'
    )
    
    if 'error' not in quantile_result:
        print("Quantile regression model fitted successfully")
        
        # Generate probabilistic forecast
        forecast = forecaster.forecast_probabilistic('quantile_model', horizon=6)
        
        if 'error' not in forecast:
            print(f"Point forecast: {forecast['forecast'][:3]}")
            print(f"95% confidence interval:")
            ci_95 = forecast['confidence_intervals'][0.95]
            print(f"  Lower: {ci_95['lower'][:3]}")
            print(f"  Upper: {ci_95['upper'][:3]}")
    
    # Fit distributional regression
    dist_result = forecaster.fit_probabilistic_model(
        series,
        method='distributional_regression',
        model_name='dist_model',
        distribution='normal'
    )
    
    if 'error' not in dist_result:
        print("Distributional regression model fitted successfully")
    
    # Fit mixture model
    mixture_result = forecaster.fit_probabilistic_model(
        series,
        method='mixture_model',
        model_name='mixture_model'
    )
    
    if 'error' not in mixture_result:
        print("Mixture model fitted successfully")
    
    # Model summary
    for model_name in ['quantile_model', 'dist_model', 'mixture_model']:
        if model_name in forecaster.models:
            summary = forecaster.get_model_summary(model_name)
            print(f"\n{model_name}: {summary['method']}")
    
    print("\nProbabilistic forecasting example completed")