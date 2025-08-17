"""
Confidence Estimator
Advanced confidence interval estimation and uncertainty quantification methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class ConfidenceEstimator:
    """Advanced confidence interval estimation for economic forecasts."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize confidence estimator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.estimators = {}
        self.historical_errors = {}
        self.confidence_models = {}
        
        # Available methods
        self.methods = {
            'bootstrap': self._bootstrap_confidence,
            'residual_bootstrap': self._residual_bootstrap_confidence,
            'conformal_prediction': self._conformal_prediction_confidence,
            'bayesian_bootstrap': self._bayesian_bootstrap_confidence,
            'error_model': self._error_model_confidence,
            'ensemble_variance': self._ensemble_variance_confidence
        }
        
        logger.info("Confidence estimator initialized")
    
    def estimate_confidence_intervals(self, 
                                    forecasts: Union[np.ndarray, List[float]],
                                    historical_data: pd.Series,
                                    method: str = 'bootstrap',
                                    confidence_levels: List[float] = None,
                                    **kwargs) -> Dict[str, Any]:
        """
        Estimate confidence intervals for forecasts.
        
        Args:
            forecasts: Point forecasts
            historical_data: Historical time series data
            method: Confidence estimation method
            confidence_levels: Confidence levels to calculate
            **kwargs: Method-specific parameters
        
        Returns:
            Confidence intervals and diagnostics
        """
        if confidence_levels is None:
            confidence_levels = [0.80, 0.90, 0.95, 0.99]
        
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")
        
        try:
            # Convert forecasts to numpy array
            forecasts = np.array(forecasts)
            
            # Call appropriate method
            result = self.methods[method](
                forecasts, historical_data, confidence_levels, **kwargs
            )
            
            # Add metadata
            result.update({
                'method': method,
                'confidence_levels': confidence_levels,
                'forecast_horizon': len(forecasts),
                'estimation_timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Estimated confidence intervals using {method}")
            return result
            
        except Exception as e:
            logger.error(f"Error estimating confidence intervals: {e}")
            return {'error': str(e)}
    
    def _bootstrap_confidence(self, 
                            forecasts: np.ndarray,
                            historical_data: pd.Series,
                            confidence_levels: List[float],
                            n_bootstrap: int = 1000,
                            block_size: Optional[int] = None) -> Dict[str, Any]:
        """Bootstrap confidence intervals."""
        # Block bootstrap for time series
        if block_size is None:
            block_size = max(1, len(historical_data) // 20)
        
        data_values = historical_data.values
        bootstrap_forecasts = []
        
        for _ in range(n_bootstrap):
            # Generate bootstrap sample
            if len(data_values) > block_size:
                # Block bootstrap
                n_blocks = len(data_values) - block_size + 1
                start_indices = np.random.choice(n_blocks, 
                                               size=len(data_values) // block_size + 1,
                                               replace=True)
                
                bootstrap_sample = []
                for start_idx in start_indices:
                    block = data_values[start_idx:start_idx + block_size]
                    bootstrap_sample.extend(block)
                
                bootstrap_sample = np.array(bootstrap_sample[:len(data_values)])
            else:
                # Regular bootstrap
                bootstrap_sample = np.random.choice(data_values, 
                                                  size=len(data_values), 
                                                  replace=True)
            
            # Simple forecast adjustment based on bootstrap sample
            sample_mean = np.mean(bootstrap_sample)
            original_mean = np.mean(data_values)
            adjustment = sample_mean - original_mean
            
            # Apply adjustment to forecasts
            adjusted_forecasts = forecasts + adjustment
            bootstrap_forecasts.append(adjusted_forecasts)
        
        bootstrap_forecasts = np.array(bootstrap_forecasts)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            alpha = (1 - conf_level) / 2
            lower_percentile = alpha * 100
            upper_percentile = (1 - alpha) * 100
            
            lower_bound = np.percentile(bootstrap_forecasts, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_forecasts, upper_percentile, axis=0)
            
            confidence_intervals[conf_level] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }
        
        return {
            'confidence_intervals': confidence_intervals,
            'bootstrap_forecasts': bootstrap_forecasts,
            'n_bootstrap': n_bootstrap,
            'block_size': block_size
        }
    
    def _residual_bootstrap_confidence(self, 
                                     forecasts: np.ndarray,
                                     historical_data: pd.Series,
                                     confidence_levels: List[float],
                                     n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Residual bootstrap confidence intervals."""
        # Fit a simple model to get residuals
        data_values = historical_data.values
        
        # Create simple trend model
        x = np.arange(len(data_values))
        trend_coeffs = np.polyfit(x, data_values, deg=1)
        trend_fit = np.polyval(trend_coeffs, x)
        residuals = data_values - trend_fit
        
        # Bootstrap residuals
        bootstrap_forecasts = []
        
        for _ in range(n_bootstrap):
            # Sample residuals with replacement
            bootstrap_residuals = np.random.choice(residuals, 
                                                 size=len(forecasts), 
                                                 replace=True)
            
            # Add residuals to forecasts
            bootstrap_forecast = forecasts + bootstrap_residuals
            bootstrap_forecasts.append(bootstrap_forecast)
        
        bootstrap_forecasts = np.array(bootstrap_forecasts)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            alpha = (1 - conf_level) / 2
            lower_percentile = alpha * 100
            upper_percentile = (1 - alpha) * 100
            
            lower_bound = np.percentile(bootstrap_forecasts, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_forecasts, upper_percentile, axis=0)
            
            confidence_intervals[conf_level] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }
        
        return {
            'confidence_intervals': confidence_intervals,
            'bootstrap_forecasts': bootstrap_forecasts,
            'residuals_std': np.std(residuals),
            'n_bootstrap': n_bootstrap
        }
    
    def _conformal_prediction_confidence(self, 
                                       forecasts: np.ndarray,
                                       historical_data: pd.Series,
                                       confidence_levels: List[float],
                                       calibration_ratio: float = 0.5) -> Dict[str, Any]:
        """Conformal prediction confidence intervals."""
        data_values = historical_data.values
        
        # Split data for calibration
        split_point = int(len(data_values) * calibration_ratio)
        train_data = data_values[:split_point]
        calib_data = data_values[split_point:]
        
        if len(calib_data) < 10:
            # Fallback if insufficient calibration data
            return self._bootstrap_confidence(forecasts, historical_data, confidence_levels)
        
        # Simple model for calibration (using moving average)
        window_size = min(12, len(train_data) // 4)
        
        # Calculate calibration errors
        calibration_errors = []
        for i in range(window_size, len(calib_data)):
            # Simple forecast using moving average
            ma_forecast = np.mean(calib_data[i-window_size:i])
            actual = calib_data[i]
            error = abs(actual - ma_forecast)
            calibration_errors.append(error)
        
        calibration_errors = np.array(calibration_errors)
        
        # Calculate quantiles for confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            
            # Conformal quantile
            quantile_level = (1 - alpha) * (1 + 1/len(calibration_errors))
            quantile_level = min(quantile_level, 1.0)
            
            error_quantile = np.quantile(calibration_errors, quantile_level)
            
            confidence_intervals[conf_level] = {
                'lower': forecasts - error_quantile,
                'upper': forecasts + error_quantile,
                'width': 2 * error_quantile
            }
        
        return {
            'confidence_intervals': confidence_intervals,
            'calibration_errors': calibration_errors,
            'calibration_size': len(calibration_errors),
            'method': 'conformal_prediction'
        }
    
    def _bayesian_bootstrap_confidence(self, 
                                     forecasts: np.ndarray,
                                     historical_data: pd.Series,
                                     confidence_levels: List[float],
                                     n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Bayesian bootstrap confidence intervals."""
        data_values = historical_data.values
        n_data = len(data_values)
        
        bootstrap_forecasts = []
        
        for _ in range(n_bootstrap):
            # Generate Dirichlet weights (Bayesian bootstrap)
            weights = np.random.dirichlet([1] * n_data)
            
            # Weighted statistics
            weighted_mean = np.sum(weights * data_values)
            weighted_var = np.sum(weights * (data_values - weighted_mean) ** 2)
            
            # Adjust forecasts
            original_mean = np.mean(data_values)
            adjustment = weighted_mean - original_mean
            noise = np.random.normal(0, np.sqrt(weighted_var), size=len(forecasts))
            
            bootstrap_forecast = forecasts + adjustment + 0.1 * noise
            bootstrap_forecasts.append(bootstrap_forecast)
        
        bootstrap_forecasts = np.array(bootstrap_forecasts)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            alpha = (1 - conf_level) / 2
            lower_percentile = alpha * 100
            upper_percentile = (1 - alpha) * 100
            
            lower_bound = np.percentile(bootstrap_forecasts, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_forecasts, upper_percentile, axis=0)
            
            confidence_intervals[conf_level] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }
        
        return {
            'confidence_intervals': confidence_intervals,
            'bootstrap_forecasts': bootstrap_forecasts,
            'n_bootstrap': n_bootstrap,
            'method': 'bayesian_bootstrap'
        }
    
    def _error_model_confidence(self, 
                              forecasts: np.ndarray,
                              historical_data: pd.Series,
                              confidence_levels: List[float],
                              lookback_window: int = 24) -> Dict[str, Any]:
        """Error model-based confidence intervals."""
        data_values = historical_data.values
        
        # Create features and targets for error modeling
        if len(data_values) < lookback_window + 10:
            return self._bootstrap_confidence(forecasts, historical_data, confidence_levels)
        
        # Generate pseudo out-of-sample errors
        errors = []
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_idx, test_idx in tscv.split(data_values):
            if len(test_idx) == 0:
                continue
                
            train_data = data_values[train_idx]
            test_data = data_values[test_idx]
            
            # Simple forecast using last value or mean
            simple_forecast = train_data[-1] if len(train_data) > 0 else np.mean(train_data)
            
            for actual in test_data[:min(len(forecasts), len(test_data))]:
                error = abs(actual - simple_forecast)
                errors.append(error)
        
        if len(errors) < 5:
            return self._bootstrap_confidence(forecasts, historical_data, confidence_levels)
        
        errors = np.array(errors)
        
        # Fit error distribution
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        
        # Test for different distributions
        distributions = {
            'normal': stats.norm,
            'gamma': stats.gamma,
            'lognormal': stats.lognorm
        }
        
        best_dist = 'normal'
        best_params = (error_mean, error_std)
        best_aic = np.inf
        
        for dist_name, dist_class in distributions.items():
            try:
                if dist_name == 'normal':
                    params = stats.norm.fit(errors)
                    aic = -2 * np.sum(stats.norm.logpdf(errors, *params)) + 2 * len(params)
                elif dist_name == 'gamma':
                    params = stats.gamma.fit(errors, floc=0)
                    aic = -2 * np.sum(stats.gamma.logpdf(errors, *params)) + 2 * len(params)
                elif dist_name == 'lognormal':
                    params = stats.lognorm.fit(errors, floc=0)
                    aic = -2 * np.sum(stats.lognorm.logpdf(errors, *params)) + 2 * len(params)
                
                if aic < best_aic:
                    best_aic = aic
                    best_dist = dist_name
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Error fitting {dist_name} distribution: {e}")
                continue
        
        # Generate confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            alpha = (1 - conf_level) / 2
            
            if best_dist == 'normal':
                error_quantile = stats.norm.ppf(1 - alpha, *best_params)
            elif best_dist == 'gamma':
                error_quantile = stats.gamma.ppf(1 - alpha, *best_params)
            elif best_dist == 'lognormal':
                error_quantile = stats.lognorm.ppf(1 - alpha, *best_params)
            else:
                error_quantile = np.quantile(errors, 1 - alpha)
            
            confidence_intervals[conf_level] = {
                'lower': forecasts - error_quantile,
                'upper': forecasts + error_quantile,
                'width': 2 * error_quantile
            }
        
        return {
            'confidence_intervals': confidence_intervals,
            'error_distribution': best_dist,
            'error_parameters': best_params,
            'historical_errors': errors,
            'error_aic': best_aic
        }
    
    def _ensemble_variance_confidence(self, 
                                    forecasts: np.ndarray,
                                    historical_data: pd.Series,
                                    confidence_levels: List[float],
                                    n_models: int = 50) -> Dict[str, Any]:
        """Ensemble variance-based confidence intervals."""
        data_values = historical_data.values
        
        if len(data_values) < 20:
            return self._bootstrap_confidence(forecasts, historical_data, confidence_levels)
        
        # Create ensemble of simple models
        ensemble_forecasts = []
        
        for i in range(n_models):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(len(data_values), 
                                               size=len(data_values), 
                                               replace=True)
            bootstrap_data = data_values[bootstrap_indices]
            
            # Fit simple model (random walk with drift)
            if len(bootstrap_data) > 1:
                drift = np.mean(np.diff(bootstrap_data))
                last_value = bootstrap_data[-1]
                
                # Generate forecast
                model_forecast = []
                current_value = last_value
                for h in range(len(forecasts)):
                    current_value += drift + np.random.normal(0, np.std(bootstrap_data) * 0.1)
                    model_forecast.append(current_value)
                
                ensemble_forecasts.append(model_forecast)
        
        ensemble_forecasts = np.array(ensemble_forecasts)
        
        # Calculate ensemble statistics
        ensemble_mean = np.mean(ensemble_forecasts, axis=0)
        ensemble_std = np.std(ensemble_forecasts, axis=0)
        
        # Generate confidence intervals using ensemble variance
        confidence_intervals = {}
        for conf_level in confidence_levels:
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            
            confidence_intervals[conf_level] = {
                'lower': ensemble_mean - z_score * ensemble_std,
                'upper': ensemble_mean + z_score * ensemble_std,
                'width': 2 * z_score * ensemble_std
            }
        
        return {
            'confidence_intervals': confidence_intervals,
            'ensemble_mean': ensemble_mean,
            'ensemble_std': ensemble_std,
            'n_models': n_models,
            'ensemble_forecasts': ensemble_forecasts
        }
    
    def adaptive_confidence_intervals(self, 
                                    forecasts: np.ndarray,
                                    historical_data: pd.Series,
                                    forecast_horizon: int,
                                    confidence_levels: List[float] = None) -> Dict[str, Any]:
        """
        Adaptive confidence intervals that adjust based on forecast horizon.
        
        Args:
            forecasts: Point forecasts
            historical_data: Historical data
            forecast_horizon: Horizon of forecasts
            confidence_levels: Confidence levels
        
        Returns:
            Adaptive confidence intervals
        """
        if confidence_levels is None:
            confidence_levels = [0.80, 0.90, 0.95]
        
        # Calculate base confidence intervals
        base_result = self.estimate_confidence_intervals(
            forecasts, historical_data, method='error_model', 
            confidence_levels=confidence_levels
        )
        
        if 'error' in base_result:
            return base_result
        
        # Horizon-based adjustment factors
        # Uncertainty typically increases with horizon
        horizon_adjustments = 1 + 0.05 * np.arange(len(forecasts))  # 5% increase per period
        
        # Volatility-based adjustments
        recent_data = historical_data.tail(min(24, len(historical_data)))
        recent_volatility = recent_data.std()
        long_term_volatility = historical_data.std()
        volatility_adjustment = recent_volatility / long_term_volatility if long_term_volatility > 0 else 1.0
        
        # Apply adjustments
        adjusted_intervals = {}
        base_intervals = base_result['confidence_intervals']
        
        for conf_level in confidence_levels:
            base_lower = base_intervals[conf_level]['lower']
            base_upper = base_intervals[conf_level]['upper']
            base_width = base_intervals[conf_level]['width']
            
            # Apply adjustments
            adjusted_width = base_width * horizon_adjustments * volatility_adjustment
            forecast_center = forecasts
            
            adjusted_intervals[conf_level] = {
                'lower': forecast_center - adjusted_width / 2,
                'upper': forecast_center + adjusted_width / 2,
                'width': adjusted_width,
                'horizon_adjustment': horizon_adjustments,
                'volatility_adjustment': volatility_adjustment
            }
        
        return {
            'confidence_intervals': adjusted_intervals,
            'base_intervals': base_intervals,
            'adjustments': {
                'horizon_factors': horizon_adjustments,
                'volatility_factor': volatility_adjustment,
                'recent_volatility': recent_volatility,
                'long_term_volatility': long_term_volatility
            },
            'method': 'adaptive'
        }
    
    def compare_confidence_methods(self, 
                                 forecasts: np.ndarray,
                                 historical_data: pd.Series,
                                 methods: List[str] = None) -> Dict[str, Any]:
        """
        Compare different confidence interval methods.
        
        Args:
            forecasts: Point forecasts
            historical_data: Historical data
            methods: Methods to compare
        
        Returns:
            Comparison results
        """
        if methods is None:
            methods = ['bootstrap', 'residual_bootstrap', 'conformal_prediction', 'error_model']
        
        comparison_results = {}
        confidence_levels = [0.80, 0.90, 0.95]
        
        for method in methods:
            try:
                result = self.estimate_confidence_intervals(
                    forecasts, historical_data, method=method,
                    confidence_levels=confidence_levels
                )
                
                if 'error' not in result:
                    # Calculate interval characteristics
                    intervals = result['confidence_intervals']
                    method_stats = {}
                    
                    for conf_level in confidence_levels:
                        width = intervals[conf_level]['width']
                        method_stats[f'avg_width_{conf_level:.0%}'] = np.mean(width)
                        method_stats[f'width_std_{conf_level:.0%}'] = np.std(width)
                    
                    comparison_results[method] = {
                        'intervals': intervals,
                        'statistics': method_stats,
                        'method_info': {k: v for k, v in result.items() 
                                      if k not in ['confidence_intervals']}
                    }
                else:
                    comparison_results[method] = {'error': result['error']}
                    
            except Exception as e:
                logger.error(f"Error comparing method {method}: {e}")
                comparison_results[method] = {'error': str(e)}
        
        return comparison_results
    
    def uncertainty_decomposition(self, 
                                forecasts: np.ndarray,
                                historical_data: pd.Series,
                                forecast_errors: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Decompose forecast uncertainty into different sources.
        
        Args:
            forecasts: Point forecasts
            historical_data: Historical data
            forecast_errors: Historical forecast errors if available
        
        Returns:
            Uncertainty decomposition
        """
        data_values = historical_data.values
        
        # Model uncertainty (variance in different model predictions)
        model_uncertainty = self._estimate_model_uncertainty(data_values, len(forecasts))
        
        # Parameter uncertainty (uncertainty in model parameters)
        parameter_uncertainty = self._estimate_parameter_uncertainty(data_values, len(forecasts))
        
        # Data uncertainty (noise in observations)
        data_uncertainty = self._estimate_data_uncertainty(data_values, len(forecasts))
        
        # Horizon uncertainty (increasing with forecast distance)
        horizon_uncertainty = self._estimate_horizon_uncertainty(len(forecasts))
        
        # Total uncertainty (combination)
        total_uncertainty = np.sqrt(
            model_uncertainty ** 2 + 
            parameter_uncertainty ** 2 + 
            data_uncertainty ** 2 + 
            horizon_uncertainty ** 2
        )
        
        # Calculate relative contributions
        total_var = total_uncertainty ** 2
        relative_contributions = {
            'model': (model_uncertainty ** 2) / total_var,
            'parameter': (parameter_uncertainty ** 2) / total_var,
            'data': (data_uncertainty ** 2) / total_var,
            'horizon': (horizon_uncertainty ** 2) / total_var
        }
        
        return {
            'uncertainty_components': {
                'model': model_uncertainty,
                'parameter': parameter_uncertainty,
                'data': data_uncertainty,
                'horizon': horizon_uncertainty,
                'total': total_uncertainty
            },
            'relative_contributions': relative_contributions,
            'forecasts': forecasts,
            'forecast_horizon': len(forecasts)
        }
    
    def _estimate_model_uncertainty(self, data: np.ndarray, horizon: int) -> np.ndarray:
        """Estimate model uncertainty component."""
        # Simplified: based on recent volatility changes
        if len(data) < 24:
            return np.full(horizon, np.std(data) * 0.1)
        
        recent_vol = np.std(data[-12:])
        long_term_vol = np.std(data)
        vol_ratio = recent_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        base_uncertainty = np.std(data) * 0.1
        return np.full(horizon, base_uncertainty * vol_ratio)
    
    def _estimate_parameter_uncertainty(self, data: np.ndarray, horizon: int) -> np.ndarray:
        """Estimate parameter uncertainty component."""
        # Based on estimation sample size
        n = len(data)
        param_uncertainty = np.std(data) * 0.05 / np.sqrt(n)
        return np.full(horizon, param_uncertainty)
    
    def _estimate_data_uncertainty(self, data: np.ndarray, horizon: int) -> np.ndarray:
        """Estimate data uncertainty component."""
        # Based on observation noise
        data_std = np.std(data)
        return np.full(horizon, data_std * 0.8)
    
    def _estimate_horizon_uncertainty(self, horizon: int) -> np.ndarray:
        """Estimate horizon-related uncertainty."""
        # Increasing uncertainty with horizon
        base_uncertainty = 0.01
        horizon_factors = 1 + 0.1 * np.arange(horizon)
        return base_uncertainty * horizon_factors


if __name__ == "__main__":
    # Example usage
    estimator = ConfidenceEstimator()
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-01', freq='M')
    trend = np.linspace(100, 120, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    seasonal = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    values = trend + seasonal + noise
    
    historical_series = pd.Series(values, index=dates)
    
    # Generate sample forecasts
    sample_forecasts = np.array([122, 123, 124, 125, 126, 127])
    
    print("Estimating confidence intervals...")
    
    # Bootstrap confidence intervals
    bootstrap_result = estimator.estimate_confidence_intervals(
        sample_forecasts, historical_series, method='bootstrap'
    )
    
    if 'error' not in bootstrap_result:
        print("Bootstrap method:")
        for conf_level, intervals in bootstrap_result['confidence_intervals'].items():
            print(f"  {conf_level:.0%} CI: [{intervals['lower'][0]:.2f}, {intervals['upper'][0]:.2f}]")
    
    # Conformal prediction
    conformal_result = estimator.estimate_confidence_intervals(
        sample_forecasts, historical_series, method='conformal_prediction'
    )
    
    if 'error' not in conformal_result:
        print("\nConformal prediction method:")
        for conf_level, intervals in conformal_result['confidence_intervals'].items():
            print(f"  {conf_level:.0%} CI: [{intervals['lower'][0]:.2f}, {intervals['upper'][0]:.2f}]")
    
    # Compare methods
    comparison = estimator.compare_confidence_methods(
        sample_forecasts, historical_series,
        methods=['bootstrap', 'conformal_prediction', 'error_model']
    )
    
    print(f"\nMethod comparison:")
    for method, results in comparison.items():
        if 'error' not in results:
            avg_width_95 = results['statistics']['avg_width_95%']
            print(f"  {method}: Average 95% CI width = {avg_width_95:.2f}")
    
    # Adaptive confidence intervals
    adaptive_result = estimator.adaptive_confidence_intervals(
        sample_forecasts, historical_series, len(sample_forecasts)
    )
    
    if 'error' not in adaptive_result:
        print(f"\nAdaptive confidence intervals:")
        adjustments = adaptive_result['adjustments']
        print(f"  Volatility adjustment: {adjustments['volatility_factor']:.3f}")
        
        intervals_95 = adaptive_result['confidence_intervals'][0.95]
        print(f"  95% CI: [{intervals_95['lower'][0]:.2f}, {intervals_95['upper'][0]:.2f}]")
    
    # Uncertainty decomposition
    uncertainty_decomp = estimator.uncertainty_decomposition(
        sample_forecasts, historical_series
    )
    
    print(f"\nUncertainty decomposition:")
    contributions = uncertainty_decomp['relative_contributions']
    for source, contribution in contributions.items():
        print(f"  {source}: {contribution:.1%}")
    
    print("\nConfidence estimation example completed")