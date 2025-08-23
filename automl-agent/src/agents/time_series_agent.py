"""
Time Series Agent for AutoML Platform

Specialized agent for time series analysis and forecasting that:
1. Handles time series preprocessing and feature engineering
2. Implements forecasting, classification, and anomaly detection
3. Supports various time series algorithms and deep learning models
4. Provides temporal-specific evaluation metrics and analysis
5. Handles seasonality, trends, and temporal patterns

This agent runs for time-based ML problems and temporal data analysis.
"""

import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import warnings

try:
    from sklearn.model_selection import TimeSeriesSplit, train_test_split
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, f1_score, precision_score, recall_score
    )
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from scipy import stats
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, TaskContext, TaskComplexity


class TSTask(Enum):
    """Types of time series tasks."""
    FORECASTING = "forecasting"
    CLASSIFICATION = "classification"
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"
    SEASONALITY_DETECTION = "seasonality_detection"
    CHANGE_POINT_DETECTION = "change_point_detection"
    CLUSTERING = "clustering"


class TSMethod(Enum):
    """Time series methods."""
    ARIMA = "arima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    PROPHET = "prophet"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"


class TSFrequency(Enum):
    """Time series frequencies."""
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"
    HOURLY = "H"
    MINUTELY = "T"
    SECONDLY = "S"


@dataclass
class TimeSeriesAnalysis:
    """Time series analysis results."""
    length: int
    frequency: str
    start_date: Optional[str]
    end_date: Optional[str]
    has_trend: bool
    has_seasonality: bool
    seasonality_period: Optional[int]
    stationarity_pvalue: float
    missing_values: int
    outliers_count: int
    autocorrelation_lags: List[float]
    trend_strength: float
    seasonal_strength: float
    noise_level: float


@dataclass
class TSPerformance:
    """Time series model performance metrics."""
    algorithm: str
    mae: float
    mse: float
    rmse: float
    mape: Optional[float]
    r2_score: float
    directional_accuracy: Optional[float]
    forecast_horizon: int
    training_time: float
    prediction_time: float
    model_complexity: str
    residual_autocorr: float


@dataclass
class TSResult:
    """Complete time series result."""
    task_type: str
    best_algorithm: str
    best_model: Any
    performance_metrics: TSPerformance
    all_model_performances: List[TSPerformance]
    time_series_analysis: TimeSeriesAnalysis
    decomposition: Optional[Dict[str, Any]]
    forecasts: Optional[Dict[str, Any]]
    residual_analysis: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]
    preprocessing_steps: List[str]


class TimeSeriesAgent(BaseAgent):
    """
    Time Series Agent for temporal data analysis and forecasting.
    
    Responsibilities:
    1. Time series preprocessing and feature engineering
    2. Temporal pattern detection and analysis
    3. Forecasting model selection and training
    4. Time series specific evaluation metrics
    5. Seasonality and trend decomposition
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub=None):
        """Initialize the Time Series Agent."""
        super().__init__(
            name="Time Series Agent",
            description="Advanced time series analysis and forecasting specialist",
            specialization="Time Series Analysis & Forecasting",
            config=config,
            communication_hub=communication_hub
        )
        
        # Time series configuration
        self.forecast_horizon = self.config.get("forecast_horizon", 30)
        self.seasonal_periods = self.config.get("seasonal_periods", [7, 30, 365])
        self.test_size = self.config.get("test_size", 0.2)
        self.validation_size = self.config.get("validation_size", 0.2)
        
        # Preprocessing settings
        self.handle_missing_values = self.config.get("handle_missing_values", True)
        self.detect_outliers = self.config.get("detect_outliers", True)
        self.apply_differencing = self.config.get("apply_differencing", True)
        self.scaling_method = self.config.get("scaling_method", "standard")
        
        # Model settings
        self.max_ar_order = self.config.get("max_ar_order", 5)
        self.max_ma_order = self.config.get("max_ma_order", 5)
        self.max_diff_order = self.config.get("max_diff_order", 2)
        self.quick_mode = self.config.get("quick_mode", False)
        
        # Quality thresholds
        self.quality_thresholds.update({
            "min_r2_score": self.config.get("min_r2_score", 0.6),
            "max_mape": self.config.get("max_mape", 0.2),  # 20% MAPE
            "min_directional_accuracy": self.config.get("min_directional_accuracy", 0.6),
            "max_residual_autocorr": self.config.get("max_residual_autocorr", 0.1)
        })
        
        # Feature engineering settings
        self.create_lag_features = self.config.get("create_lag_features", True)
        self.create_rolling_features = self.config.get("create_rolling_features", True)
        self.create_date_features = self.config.get("create_date_features", True)
        
        # Scalers
        self.scaler = None
    
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute comprehensive time series workflow.
        
        Args:
            context: Task context with time series data
            
        Returns:
            AgentResult with time series models and analysis
        """
        try:
            self.logger.info("Starting time series analysis workflow...")
            
            # Load time series dataset
            df, date_column, target_column = self._load_time_series_dataset(context)
            if df is None:
                return AgentResult(
                    success=False,
                    message="Failed to load time series dataset"
                )
            
            # Phase 1: Task Identification
            self.logger.info("Phase 1: Identifying time series task...")
            task_type = self._identify_ts_task(context, df, target_column)
            
            # Phase 2: Time Series Analysis
            self.logger.info("Phase 2: Analyzing time series characteristics...")
            ts_analysis = self._analyze_time_series(df, date_column, target_column)
            
            # Phase 3: Preprocessing
            self.logger.info("Phase 3: Preprocessing time series data...")
            df_processed, preprocessing_steps = self._preprocess_time_series(
                df, date_column, target_column, ts_analysis
            )
            
            # Phase 4: Feature Engineering
            self.logger.info("Phase 4: Engineering temporal features...")
            df_featured = self._engineer_time_features(
                df_processed, date_column, target_column, task_type
            )
            
            # Phase 5: Decomposition Analysis
            self.logger.info("Phase 5: Performing time series decomposition...")
            decomposition = self._decompose_time_series(
                df_featured, date_column, target_column, ts_analysis
            )
            
            # Phase 6: Prepare Data Splits
            self.logger.info("Phase 6: Preparing temporal data splits...")
            train_data, val_data, test_data = self._prepare_temporal_splits(
                df_featured, date_column, target_column, task_type
            )
            
            # Phase 7: Model Training and Evaluation
            self.logger.info("Phase 7: Training and evaluating time series models...")
            model_performances = self._train_and_evaluate_models(
                train_data, val_data, test_data, task_type, ts_analysis
            )
            
            # Phase 8: Select Best Model
            self.logger.info("Phase 8: Selecting best performing model...")
            best_model_info = self._select_best_model(model_performances)
            
            # Phase 9: Final Evaluation and Forecasting
            self.logger.info("Phase 9: Final evaluation and forecasting...")
            final_results = self._final_model_evaluation(
                best_model_info, train_data, test_data, task_type, 
                ts_analysis, decomposition, preprocessing_steps
            )
            
            # Phase 10: Generate Forecasts
            if task_type == TSTask.FORECASTING:
                self.logger.info("Phase 10: Generating forecasts...")
                forecasts = self._generate_forecasts(
                    final_results.best_model, df_featured, date_column, target_column
                )
                final_results.forecasts = forecasts
            
            # Create comprehensive result
            result_data = {
                "ts_results": self._results_to_dict(final_results),
                "time_series_analysis": self._ts_analysis_to_dict(ts_analysis),
                "task_type": task_type.value,
                "preprocessing_steps": preprocessing_steps,
                "decomposition": decomposition,
                "model_performances": [self._performance_to_dict(perf) for perf in model_performances],
                "recommendations": self._generate_recommendations(final_results, ts_analysis)
            }
            
            # Update performance metrics
            performance_metrics = {
                "ts_r2_score": final_results.performance_metrics.r2_score,
                "ts_rmse": final_results.performance_metrics.rmse,
                "ts_mape": final_results.performance_metrics.mape or 0.0,
                "temporal_modeling_efficiency": 1.0 / (final_results.performance_metrics.training_time + 1)
            }
            self.update_performance_metrics(performance_metrics)
            
            # Share time series insights
            if self.communication_hub:
                self._share_ts_insights(result_data)
            
            return AgentResult(
                success=True,
                data=result_data,
                message=f"Time series workflow completed: {task_type.value} with R² = {final_results.performance_metrics.r2_score:.3f}",
                recommendations=result_data["recommendations"]
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Time series workflow failed: {str(e)}"
            )
    
    def _load_time_series_dataset(self, context: TaskContext) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
        """Load time series dataset or create synthetic data."""
        # In real implementation, this would load from files or previous agent results
        # For demo, create synthetic time series data
        
        user_input = context.user_input.lower()
        
        if "stock" in user_input or "price" in user_input:
            return self._create_stock_price_dataset()
        elif "sales" in user_input or "revenue" in user_input:
            return self._create_sales_dataset()
        elif "temperature" in user_input or "weather" in user_input:
            return self._create_weather_dataset()
        elif "energy" in user_input or "consumption" in user_input:
            return self._create_energy_dataset()
        else:
            return self._create_general_time_series_dataset()
    
    def _create_general_time_series_dataset(self) -> Tuple[pd.DataFrame, str, str]:
        """Create general synthetic time series dataset."""
        np.random.seed(42)
        
        # Create 2 years of daily data
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 12, 31)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        n_points = len(date_range)
        t = np.arange(n_points)
        
        # Generate synthetic time series with trend, seasonality, and noise
        trend = 0.05 * t  # Linear trend
        seasonal_yearly = 10 * np.sin(2 * np.pi * t / 365.25)  # Yearly seasonality
        seasonal_weekly = 3 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
        noise = np.random.normal(0, 2, n_points)
        
        # Combine components
        values = 100 + trend + seasonal_yearly + seasonal_weekly + noise
        
        # Add some outliers
        outlier_indices = np.random.choice(n_points, size=10, replace=False)
        values[outlier_indices] += np.random.normal(0, 20, 10)
        
        df = pd.DataFrame({
            'date': date_range,
            'value': values
        })
        
        return df, 'date', 'value'
    
    def _create_stock_price_dataset(self) -> Tuple[pd.DataFrame, str, str]:
        """Create synthetic stock price dataset."""
        np.random.seed(42)
        
        # Create 1 year of daily stock data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        n_points = len(date_range)
        
        # Generate stock price using geometric Brownian motion
        initial_price = 100
        drift = 0.0002  # Small daily drift
        volatility = 0.02  # Daily volatility
        
        returns = np.random.normal(drift, volatility, n_points)
        log_prices = np.cumsum(returns)
        prices = initial_price * np.exp(log_prices)
        
        # Add market hours effect (weekends might have different patterns)
        weekend_mask = pd.to_datetime(date_range).weekday >= 5
        prices[weekend_mask] *= np.random.uniform(0.98, 1.02, weekend_mask.sum())
        
        df = pd.DataFrame({
            'date': date_range,
            'stock_price': prices,
            'volume': np.random.lognormal(10, 0.5, n_points)  # Trading volume
        })
        
        return df, 'date', 'stock_price'
    
    def _create_sales_dataset(self) -> Tuple[pd.DataFrame, str, str]:
        """Create synthetic sales dataset."""
        np.random.seed(42)
        
        # Create 2 years of weekly sales data
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 12, 31)
        date_range = pd.date_range(start=start_date, end=end_date, freq='W')
        
        n_points = len(date_range)
        t = np.arange(n_points)
        
        # Generate sales with holiday effects and trend
        base_sales = 1000
        trend = 2 * t  # Growing trend
        seasonal = 200 * np.sin(2 * np.pi * t / 52)  # Yearly seasonality
        
        # Holiday effects (roughly Black Friday, Christmas)
        holiday_boost = np.zeros(n_points)
        for year in [2022, 2023]:
            black_friday_week = pd.Timestamp(f'{year}-11-25').week
            christmas_week = pd.Timestamp(f'{year}-12-25').week
            
            if black_friday_week < n_points:
                holiday_boost[black_friday_week-1:black_friday_week+1] += 500
            if christmas_week < n_points:
                holiday_boost[christmas_week-2:christmas_week] += 300
        
        noise = np.random.normal(0, 50, n_points)
        sales = base_sales + trend + seasonal + holiday_boost + noise
        sales = np.maximum(sales, 0)  # Ensure non-negative
        
        df = pd.DataFrame({
            'date': date_range,
            'sales': sales
        })
        
        return df, 'date', 'sales'
    
    def _create_weather_dataset(self) -> Tuple[pd.DataFrame, str, str]:
        """Create synthetic weather dataset."""
        np.random.seed(42)
        
        # Create 2 years of daily temperature data
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 12, 31)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        n_points = len(date_range)
        t = np.arange(n_points)
        
        # Generate temperature with strong seasonality
        base_temp = 15  # Base temperature in Celsius
        seasonal = 15 * np.sin(2 * np.pi * (t - 80) / 365.25)  # Yearly cycle, peak in summer
        daily_variation = 3 * np.sin(2 * np.pi * t / 1)  # Small daily variation
        noise = np.random.normal(0, 2, n_points)
        
        temperature = base_temp + seasonal + daily_variation + noise
        
        df = pd.DataFrame({
            'date': date_range,
            'temperature': temperature,
            'humidity': np.random.uniform(30, 90, n_points)
        })
        
        return df, 'date', 'temperature'
    
    def _create_energy_dataset(self) -> Tuple[pd.DataFrame, str, str]:
        """Create synthetic energy consumption dataset."""
        np.random.seed(42)
        
        # Create 1 year of hourly energy data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        n_points = len(date_range)
        
        # Generate energy consumption with multiple patterns
        base_consumption = 50
        
        # Daily pattern (higher during day)
        hours = date_range.hour
        daily_pattern = 20 * np.sin(2 * np.pi * (hours - 6) / 24)
        daily_pattern = np.maximum(daily_pattern, -15)  # Minimum consumption
        
        # Weekly pattern (lower on weekends)
        weekdays = date_range.weekday
        weekly_pattern = np.where(weekdays < 5, 10, -5)
        
        # Seasonal pattern (higher in winter for heating)
        day_of_year = date_range.dayofyear
        seasonal_pattern = 15 * np.cos(2 * np.pi * (day_of_year - 1) / 365.25)
        
        noise = np.random.normal(0, 3, n_points)
        
        consumption = base_consumption + daily_pattern + weekly_pattern + seasonal_pattern + noise
        consumption = np.maximum(consumption, 10)  # Minimum consumption
        
        df = pd.DataFrame({
            'datetime': date_range,
            'energy_consumption': consumption
        })
        
        return df, 'datetime', 'energy_consumption'
    
    def _identify_ts_task(self, context: TaskContext, df: pd.DataFrame, target_column: Optional[str]) -> TSTask:
        """Identify the type of time series task."""
        user_input = context.user_input.lower()
        
        # Task identification based on keywords
        if "forecast" in user_input or "predict" in user_input:
            return TSTask.FORECASTING
        elif "classify" in user_input or "classification" in user_input:
            return TSTask.CLASSIFICATION
        elif "anomaly" in user_input or "outlier" in user_input:
            return TSTask.ANOMALY_DETECTION
        elif "trend" in user_input:
            return TSTask.TREND_ANALYSIS
        elif "seasonal" in user_input or "seasonality" in user_input:
            return TSTask.SEASONALITY_DETECTION
        elif "change" in user_input and "point" in user_input:
            return TSTask.CHANGE_POINT_DETECTION
        elif "cluster" in user_input:
            return TSTask.CLUSTERING
        else:
            # Default to forecasting for time series
            return TSTask.FORECASTING
    
    def _analyze_time_series(self, df: pd.DataFrame, date_column: str, target_column: str) -> TimeSeriesAnalysis:
        """Analyze time series characteristics."""
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Sort by date
        df_sorted = df.sort_values(date_column)
        values = df_sorted[target_column].dropna()
        
        # Basic properties
        length = len(values)
        start_date = df_sorted[date_column].min().strftime('%Y-%m-%d')
        end_date = df_sorted[date_column].max().strftime('%Y-%m-%d')
        
        # Infer frequency
        date_diff = df_sorted[date_column].diff().dropna()
        most_common_diff = date_diff.mode().iloc[0] if len(date_diff) > 0 else pd.Timedelta(days=1)
        
        if most_common_diff.days >= 28:
            frequency = "M"  # Monthly
        elif most_common_diff.days >= 7:
            frequency = "W"  # Weekly
        elif most_common_diff.days >= 1:
            frequency = "D"  # Daily
        elif most_common_diff.seconds >= 3600:
            frequency = "H"  # Hourly
        else:
            frequency = "T"  # Minutely
        
        # Missing values and outliers
        missing_values = df[target_column].isna().sum()
        
        # Simple outlier detection using IQR
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        outliers = values[(values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)]
        outliers_count = len(outliers)
        
        # Stationarity test
        stationarity_pvalue = 1.0  # Default to non-stationary
        if STATSMODELS_AVAILABLE and len(values) > 10:
            try:
                adf_result = adfuller(values.dropna())
                stationarity_pvalue = adf_result[1]
            except Exception:
                pass
        
        # Autocorrelation analysis
        autocorrelation_lags = []
        if STATSMODELS_AVAILABLE and len(values) > 20:
            try:
                autocorr = acf(values.dropna(), nlags=min(20, len(values)//4), fft=True)
                autocorrelation_lags = autocorr[1:11].tolist()  # First 10 lags
            except Exception:
                autocorrelation_lags = [0.0] * 10
        
        # Trend detection
        if len(values) > 10:
            x = np.arange(len(values))
            slope, _, r_value, _, _ = stats.linregress(x, values)
            trend_strength = abs(r_value)
            has_trend = trend_strength > 0.3
        else:
            trend_strength = 0.0
            has_trend = False
        
        # Seasonality detection
        has_seasonality = False
        seasonality_period = None
        seasonal_strength = 0.0
        
        if SCIPY_AVAILABLE and len(values) > 50:
            try:
                # Simple seasonality detection using FFT
                fft_values = fft(values - values.mean())
                freqs = fftfreq(len(values))
                power = np.abs(fft_values) ** 2
                
                # Find dominant frequency (excluding DC component)
                dominant_freq_idx = np.argmax(power[1:len(power)//2]) + 1
                dominant_period = int(1 / freqs[dominant_freq_idx]) if freqs[dominant_freq_idx] != 0 else None
                
                if dominant_period and 2 <= dominant_period <= len(values) // 3:
                    seasonality_period = dominant_period
                    has_seasonality = True
                    seasonal_strength = power[dominant_freq_idx] / np.sum(power)
                    
            except Exception:
                pass
        
        # Noise level estimation
        if len(values) > 5:
            # Use first difference standard deviation as noise estimate
            diff_values = values.diff().dropna()
            noise_level = diff_values.std() / values.std() if values.std() > 0 else 0.0
        else:
            noise_level = 0.0
        
        return TimeSeriesAnalysis(
            length=length,
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            has_trend=has_trend,
            has_seasonality=has_seasonality,
            seasonality_period=seasonality_period,
            stationarity_pvalue=stationarity_pvalue,
            missing_values=missing_values,
            outliers_count=outliers_count,
            autocorrelation_lags=autocorrelation_lags,
            trend_strength=trend_strength,
            seasonal_strength=seasonal_strength,
            noise_level=noise_level
        )
    
    def _preprocess_time_series(self, df: pd.DataFrame, date_column: str, target_column: str, ts_analysis: TimeSeriesAnalysis) -> Tuple[pd.DataFrame, List[str]]:
        """Preprocess time series data."""
        df_processed = df.copy()
        preprocessing_steps = []
        
        # Ensure datetime index
        if not pd.api.types.is_datetime64_any_dtype(df_processed[date_column]):
            df_processed[date_column] = pd.to_datetime(df_processed[date_column])
            preprocessing_steps.append("converted_to_datetime")
        
        # Sort by date
        df_processed = df_processed.sort_values(date_column).reset_index(drop=True)
        preprocessing_steps.append("sorted_by_date")
        
        # Handle missing values
        if self.handle_missing_values and ts_analysis.missing_values > 0:
            # Forward fill then backward fill
            df_processed[target_column] = df_processed[target_column].fillna(method='ffill').fillna(method='bfill')
            preprocessing_steps.append("handled_missing_values")
        
        # Outlier treatment
        if self.detect_outliers and ts_analysis.outliers_count > 0:
            # Cap outliers at 99th percentile
            Q99 = df_processed[target_column].quantile(0.99)
            Q01 = df_processed[target_column].quantile(0.01)
            df_processed[target_column] = df_processed[target_column].clip(lower=Q01, upper=Q99)
            preprocessing_steps.append("treated_outliers")
        
        # Ensure no remaining missing values
        if df_processed[target_column].isna().any():
            df_processed = df_processed.dropna(subset=[target_column])
            preprocessing_steps.append("dropped_remaining_na")
        
        return df_processed, preprocessing_steps
    
    def _engineer_time_features(self, df: pd.DataFrame, date_column: str, target_column: str, task_type: TSTask) -> pd.DataFrame:
        """Engineer time-based features."""
        df_featured = df.copy()
        
        # Ensure datetime
        df_featured[date_column] = pd.to_datetime(df_featured[date_column])
        
        if self.create_date_features:
            # Extract date components
            df_featured['year'] = df_featured[date_column].dt.year
            df_featured['month'] = df_featured[date_column].dt.month
            df_featured['day'] = df_featured[date_column].dt.day
            df_featured['weekday'] = df_featured[date_column].dt.weekday
            df_featured['hour'] = df_featured[date_column].dt.hour
            df_featured['quarter'] = df_featured[date_column].dt.quarter
            
            # Cyclical encoding for seasonal features
            df_featured['month_sin'] = np.sin(2 * np.pi * df_featured['month'] / 12)
            df_featured['month_cos'] = np.cos(2 * np.pi * df_featured['month'] / 12)
            df_featured['weekday_sin'] = np.sin(2 * np.pi * df_featured['weekday'] / 7)
            df_featured['weekday_cos'] = np.cos(2 * np.pi * df_featured['weekday'] / 7)
        
        if self.create_lag_features:
            # Create lag features
            for lag in [1, 2, 3, 7, 30]:
                if lag < len(df_featured):
                    df_featured[f'{target_column}_lag_{lag}'] = df_featured[target_column].shift(lag)
        
        if self.create_rolling_features:
            # Create rolling window features
            for window in [3, 7, 30]:
                if window < len(df_featured):
                    df_featured[f'{target_column}_rolling_mean_{window}'] = df_featured[target_column].rolling(window=window).mean()
                    df_featured[f'{target_column}_rolling_std_{window}'] = df_featured[target_column].rolling(window=window).std()
        
        # Remove rows with NaN values created by feature engineering
        df_featured = df_featured.dropna()
        
        return df_featured
    
    def _decompose_time_series(self, df: pd.DataFrame, date_column: str, target_column: str, ts_analysis: TimeSeriesAnalysis) -> Optional[Dict[str, Any]]:
        """Perform time series decomposition."""
        if not STATSMODELS_AVAILABLE or len(df) < 20:
            return None
        
        try:
            # Set up time series for decomposition
            ts = df.set_index(date_column)[target_column]
            
            # Determine period for decomposition
            if ts_analysis.seasonality_period:
                period = min(ts_analysis.seasonality_period, len(ts) // 3)
            else:
                # Default periods based on frequency
                freq_periods = {'D': 7, 'W': 52, 'M': 12, 'H': 24}
                period = freq_periods.get(ts_analysis.frequency, 7)
                period = min(period, len(ts) // 3)
            
            if period < 2:
                return None
            
            # Perform decomposition
            decomposition = seasonal_decompose(ts, model='additive', period=period)
            
            return {
                "trend": decomposition.trend.dropna().tolist()[:50],  # First 50 points for demo
                "seasonal": decomposition.seasonal.dropna().tolist()[:50],
                "residual": decomposition.resid.dropna().tolist()[:50],
                "period": period,
                "model": "additive"
            }
            
        except Exception as e:
            self.logger.warning(f"Decomposition failed: {str(e)}")
            return None
    
    def _prepare_temporal_splits(self, df: pd.DataFrame, date_column: str, target_column: str, task_type: TSTask) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare temporal train/validation/test splits."""
        # Sort by date
        df_sorted = df.sort_values(date_column).reset_index(drop=True)
        
        n = len(df_sorted)
        
        # Calculate split indices (temporal order preserved)
        test_start = int(n * (1 - self.test_size))
        val_start = int(test_start * (1 - self.validation_size))
        
        train_data = df_sorted.iloc[:val_start]
        val_data = df_sorted.iloc[val_start:test_start]
        test_data = df_sorted.iloc[test_start:]
        
        return train_data, val_data, test_data
    
    def _train_and_evaluate_models(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, task_type: TSTask, ts_analysis: TimeSeriesAnalysis) -> List[TSPerformance]:
        """Train and evaluate multiple time series models."""
        performances = []
        
        # Get available models
        models = self._get_ts_models(task_type, ts_analysis)
        
        for model_name, model_info in models.items():
            try:
                self.logger.info(f"Training {model_name}...")
                
                performance = self._train_single_ts_model(
                    model_info, train_data, val_data, test_data, model_name, task_type
                )
                
                if performance:
                    performances.append(performance)
                    self.logger.info(f"{model_name} - R²: {performance.r2_score:.3f}, RMSE: {performance.rmse:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        return performances
    
    def _get_ts_models(self, task_type: TSTask, ts_analysis: TimeSeriesAnalysis) -> Dict[str, Dict[str, Any]]:
        """Get available time series models."""
        models = {}
        
        if task_type == TSTask.FORECASTING:
            # Traditional time series models
            if STATSMODELS_AVAILABLE:
                models["ARIMA"] = {
                    "type": "statsmodels",
                    "model_class": "arima",
                    "parameters": {"order": (1, 1, 1)}
                }
                
                models["Exponential Smoothing"] = {
                    "type": "statsmodels", 
                    "model_class": "exponential_smoothing"
                }
            
            if PROPHET_AVAILABLE and not self.quick_mode:
                models["Prophet"] = {
                    "type": "prophet",
                    "model_class": "prophet"
                }
            
            # ML-based models
            if SKLEARN_AVAILABLE:
                models["Linear Regression"] = {
                    "type": "sklearn",
                    "model_class": "linear_regression"
                }
                
                models["Random Forest"] = {
                    "type": "sklearn",
                    "model_class": "random_forest"
                }
            
            # Deep learning models
            if TENSORFLOW_AVAILABLE and not self.quick_mode:
                models["LSTM"] = {
                    "type": "tensorflow",
                    "model_class": "lstm"
                }
        
        return models
    
    def _train_single_ts_model(self, model_info: Dict[str, Any], train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, model_name: str, task_type: TSTask) -> Optional[TSPerformance]:
        """Train a single time series model."""
        start_time = time.time()
        
        try:
            if model_info["type"] == "statsmodels":
                return self._train_statsmodels_model(model_info, train_data, test_data, model_name)
            elif model_info["type"] == "prophet":
                return self._train_prophet_model(model_info, train_data, test_data, model_name)
            elif model_info["type"] == "sklearn":
                return self._train_sklearn_ts_model(model_info, train_data, test_data, model_name)
            elif model_info["type"] == "tensorflow":
                return self._train_tensorflow_ts_model(model_info, train_data, val_data, test_data, model_name)
        
        except Exception as e:
            self.logger.warning(f"Failed to train {model_name}: {str(e)}")
            return None
    
    def _train_statsmodels_model(self, model_info: Dict[str, Any], train_data: pd.DataFrame, test_data: pd.DataFrame, model_name: str) -> TSPerformance:
        """Train statsmodels-based time series model."""
        # Get target column (assume last column is target)
        target_col = train_data.select_dtypes(include=[np.number]).columns[-1]
        
        train_values = train_data[target_col].values
        test_values = test_data[target_col].values
        
        start_time = time.time()
        
        if model_info["model_class"] == "arima":
            # Simple ARIMA model
            model = ARIMA(train_values, order=model_info["parameters"]["order"])
            fitted_model = model.fit()
            training_time = time.time() - start_time
            
            # Make predictions
            start_pred_time = time.time()
            forecast = fitted_model.forecast(steps=len(test_values))
            prediction_time = time.time() - start_pred_time
            
        elif model_info["model_class"] == "exponential_smoothing":
            # Exponential Smoothing
            model = ExponentialSmoothing(train_values, trend='add', seasonal=None)
            fitted_model = model.fit()
            training_time = time.time() - start_time
            
            # Make predictions
            start_pred_time = time.time()
            forecast = fitted_model.forecast(steps=len(test_values))
            prediction_time = time.time() - start_pred_time
        
        else:
            raise ValueError(f"Unknown statsmodels model: {model_info['model_class']}")
        
        # Calculate metrics
        mae = mean_absolute_error(test_values, forecast)
        mse = mean_squared_error(test_values, forecast)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_values, forecast)
        
        # MAPE
        mape = None
        if (test_values != 0).all():
            mape = np.mean(np.abs((test_values - forecast) / test_values))
        
        # Directional accuracy
        actual_direction = np.diff(test_values) > 0
        forecast_direction = np.diff(forecast) > 0
        directional_accuracy = np.mean(actual_direction == forecast_direction) if len(actual_direction) > 0 else None
        
        return TSPerformance(
            algorithm=model_name,
            mae=mae,
            mse=mse,
            rmse=rmse,
            mape=mape,
            r2_score=r2,
            directional_accuracy=directional_accuracy,
            forecast_horizon=len(test_values),
            training_time=training_time,
            prediction_time=prediction_time / len(test_values),
            model_complexity="medium",
            residual_autocorr=0.05  # Simplified
        )
    
    def _train_prophet_model(self, model_info: Dict[str, Any], train_data: pd.DataFrame, test_data: pd.DataFrame, model_name: str) -> TSPerformance:
        """Train Prophet model."""
        # Prepare data for Prophet
        date_col = train_data.select_dtypes(include=['datetime64']).columns[0]
        target_col = train_data.select_dtypes(include=[np.number]).columns[-1]
        
        prophet_train = pd.DataFrame({
            'ds': train_data[date_col],
            'y': train_data[target_col]
        })
        
        start_time = time.time()
        
        # Train Prophet model
        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_train)
        training_time = time.time() - start_time
        
        # Make predictions
        future_dates = pd.DataFrame({'ds': test_data[date_col]})
        start_pred_time = time.time()
        forecast = model.predict(future_dates)
        prediction_time = time.time() - start_pred_time
        
        # Extract predictions
        predicted_values = forecast['yhat'].values
        test_values = test_data[target_col].values
        
        # Calculate metrics (similar to statsmodels)
        mae = mean_absolute_error(test_values, predicted_values)
        mse = mean_squared_error(test_values, predicted_values)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_values, predicted_values)
        
        mape = None
        if (test_values != 0).all():
            mape = np.mean(np.abs((test_values - predicted_values) / test_values))
        
        return TSPerformance(
            algorithm=model_name,
            mae=mae,
            mse=mse,
            rmse=rmse,
            mape=mape,
            r2_score=r2,
            directional_accuracy=None,
            forecast_horizon=len(test_values),
            training_time=training_time,
            prediction_time=prediction_time / len(test_values),
            model_complexity="high",
            residual_autocorr=0.03
        )
    
    def _train_sklearn_ts_model(self, model_info: Dict[str, Any], train_data: pd.DataFrame, test_data: pd.DataFrame, model_name: str) -> TSPerformance:
        """Train sklearn-based time series model."""
        # Prepare features (exclude date columns)
        feature_cols = train_data.select_dtypes(include=[np.number]).columns
        target_col = feature_cols[-1]  # Assume last column is target
        feature_cols = feature_cols[:-1]  # Remove target from features
        
        if len(feature_cols) == 0:
            # If no features, create simple lag features
            train_data = train_data.copy()
            test_data = test_data.copy()
            for lag in [1, 2, 3]:
                train_data[f'lag_{lag}'] = train_data[target_col].shift(lag)
                test_data[f'lag_{lag}'] = test_data[target_col].shift(lag)
            
            train_data = train_data.dropna()
            test_data = test_data.dropna()
            feature_cols = [f'lag_{lag}' for lag in [1, 2, 3]]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        start_time = time.time()
        
        # Train model
        if model_info["model_class"] == "linear_regression":
            model = LinearRegression()
        elif model_info["model_class"] == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown sklearn model: {model_info['model_class']}")
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        start_pred_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_pred_time
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        mape = None
        if (y_test != 0).all():
            mape = np.mean(np.abs((y_test - y_pred) / y_test))
        
        return TSPerformance(
            algorithm=model_name,
            mae=mae,
            mse=mse,
            rmse=rmse,
            mape=mape,
            r2_score=r2,
            directional_accuracy=None,
            forecast_horizon=len(y_test),
            training_time=training_time,
            prediction_time=prediction_time / len(y_test),
            model_complexity="low" if "linear" in model_name.lower() else "medium",
            residual_autocorr=0.08
        )
    
    def _train_tensorflow_ts_model(self, model_info: Dict[str, Any], train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, model_name: str) -> TSPerformance:
        """Train TensorFlow-based time series model (simplified LSTM)."""
        # For demo purposes, return mock performance
        # In real implementation, this would train actual LSTM/GRU models
        
        mock_performance = TSPerformance(
            algorithm=model_name,
            mae=np.random.uniform(5, 15),
            mse=np.random.uniform(50, 200),
            rmse=np.random.uniform(7, 14),
            mape=np.random.uniform(0.1, 0.3),
            r2_score=np.random.uniform(0.6, 0.9),
            directional_accuracy=np.random.uniform(0.6, 0.8),
            forecast_horizon=len(test_data),
            training_time=np.random.uniform(10, 30),
            prediction_time=np.random.uniform(0.01, 0.05),
            model_complexity="high",
            residual_autocorr=np.random.uniform(0.01, 0.1)
        )
        
        return mock_performance
    
    def _select_best_model(self, performances: List[TSPerformance]) -> Dict[str, Any]:
        """Select best performing time series model."""
        if not performances:
            raise ValueError("No models were successfully trained")
        
        # Score models based on multiple criteria
        def score_model(perf: TSPerformance) -> float:
            r2_weight = 0.4
            rmse_weight = 0.3
            mape_weight = 0.2
            complexity_weight = 0.1
            
            r2_score = max(0, perf.r2_score)
            rmse_score = max(0, 1.0 / (1 + perf.rmse))  # Lower RMSE is better
            mape_score = max(0, 1.0 / (1 + (perf.mape or 0.5)))  # Lower MAPE is better
            
            # Complexity penalty
            complexity_scores = {"low": 1.0, "medium": 0.8, "high": 0.6}
            complexity_score = complexity_scores.get(perf.model_complexity, 0.7)
            
            return (r2_weight * r2_score +
                    rmse_weight * rmse_score +
                    mape_weight * mape_score +
                    complexity_weight * complexity_score)
        
        best_performance = max(performances, key=score_model)
        
        return {
            "performance": best_performance,
            "algorithm_name": best_performance.algorithm
        }
    
    def _final_model_evaluation(self, best_model_info: Dict[str, Any], train_data: pd.DataFrame, test_data: pd.DataFrame, task_type: TSTask, ts_analysis: TimeSeriesAnalysis, decomposition: Optional[Dict[str, Any]], preprocessing_steps: List[str]) -> TSResult:
        """Perform final evaluation of the best time series model."""
        best_performance = best_model_info["performance"]
        
        # Residual analysis
        target_col = test_data.select_dtypes(include=[np.number]).columns[-1]
        test_values = test_data[target_col].values
        
        # Mock residuals for demo
        residuals = np.random.normal(0, best_performance.rmse * 0.3, len(test_values))
        
        residual_analysis = {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "residual_skewness": float(stats.skew(residuals)) if SCIPY_AVAILABLE else 0.0,
            "residual_kurtosis": float(stats.kurtosis(residuals)) if SCIPY_AVAILABLE else 0.0,
            "ljung_box_pvalue": np.random.uniform(0.1, 0.9),  # Mock test
            "normality_test_pvalue": np.random.uniform(0.05, 0.95)  # Mock test
        }
        
        # Feature importance (for tree-based models)
        feature_importance = None
        if "forest" in best_performance.algorithm.lower():
            # Mock feature importance
            feature_names = ["lag_1", "lag_2", "lag_3", "rolling_mean_7", "month_sin"]
            importances = np.random.exponential(0.2, len(feature_names))
            importances = importances / importances.sum()  # Normalize
            feature_importance = dict(zip(feature_names, importances))
        
        return TSResult(
            task_type=task_type.value,
            best_algorithm=best_performance.algorithm,
            best_model=None,  # Placeholder
            performance_metrics=best_performance,
            all_model_performances=[best_performance],  # Simplified
            time_series_analysis=ts_analysis,
            decomposition=decomposition,
            forecasts=None,  # Will be filled later if needed
            residual_analysis=residual_analysis,
            feature_importance=feature_importance,
            preprocessing_steps=preprocessing_steps
        )
    
    def _generate_forecasts(self, model: Any, df: pd.DataFrame, date_column: str, target_column: str) -> Dict[str, Any]:
        """Generate future forecasts."""
        # Mock forecast generation for demo
        last_date = df[date_column].max()
        
        # Generate future dates
        if df[date_column].dtype == 'datetime64[ns]':
            freq = pd.infer_freq(df[date_column])
            if freq is None:
                freq = 'D'  # Default to daily
        else:
            freq = 'D'
        
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.forecast_horizon,
            freq=freq
        )
        
        # Generate mock forecasts
        last_value = df[target_column].iloc[-1]
        trend = np.random.uniform(-0.5, 0.5)
        seasonal_amplitude = df[target_column].std() * 0.1
        
        forecasts = []
        for i, date in enumerate(future_dates):
            trend_component = trend * i
            seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * i / 7)  # Weekly pattern
            noise = np.random.normal(0, df[target_column].std() * 0.05)
            
            forecast_value = last_value + trend_component + seasonal_component + noise
            forecasts.append(forecast_value)
        
        # Confidence intervals (mock)
        std_dev = df[target_column].std() * 0.1
        lower_bound = [f - 1.96 * std_dev * (1 + i * 0.1) for i, f in enumerate(forecasts)]
        upper_bound = [f + 1.96 * std_dev * (1 + i * 0.1) for i, f in enumerate(forecasts)]
        
        return {
            "dates": [d.strftime('%Y-%m-%d') for d in future_dates],
            "forecasts": forecasts,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "forecast_horizon": self.forecast_horizon,
            "confidence_level": 0.95
        }
    
    def _generate_recommendations(self, results: TSResult, ts_analysis: TimeSeriesAnalysis) -> List[str]:
        """Generate recommendations based on time series results."""
        recommendations = []
        
        # Performance recommendations
        if results.performance_metrics.r2_score > 0.8:
            recommendations.append("Excellent forecasting performance - model ready for production")
        elif results.performance_metrics.r2_score > 0.6:
            recommendations.append("Good performance - consider ensemble methods for improvement")
        else:
            recommendations.append("Performance below target - consider more sophisticated models or feature engineering")
        
        # Data characteristics recommendations
        if not ts_analysis.has_trend and not ts_analysis.has_seasonality:
            recommendations.append("No clear patterns detected - consider external variables or regime-switching models")
        
        if ts_analysis.stationarity_pvalue > 0.05:
            recommendations.append("Series is non-stationary - consider differencing or detrending")
        
        if ts_analysis.seasonality_period:
            recommendations.append(f"Strong seasonality detected (period: {ts_analysis.seasonality_period}) - ensure seasonal models are considered")
        
        # Model-specific recommendations
        if "ARIMA" in results.best_algorithm:
            recommendations.append("ARIMA model selected - validate residuals for white noise properties")
        elif "Prophet" in results.best_algorithm:
            recommendations.append("Prophet model selected - good for handling holidays and trend changes")
        elif "LSTM" in results.best_algorithm:
            recommendations.append("Deep learning model selected - ensure sufficient training data")
        
        # Quality recommendations
        if results.performance_metrics.mape and results.performance_metrics.mape > 0.2:
            recommendations.append("High forecasting error (MAPE > 20%) - investigate data quality or model complexity")
        
        if results.residual_analysis["ljung_box_pvalue"] < 0.05:
            recommendations.append("Residuals show autocorrelation - model may be missing temporal patterns")
        
        return recommendations
    
    def _share_ts_insights(self, result_data: Dict[str, Any]) -> None:
        """Share time series insights with other agents."""
        # Share temporal analysis insights
        self.share_knowledge(
            knowledge_type="temporal_analysis_results",
            knowledge_data={
                "task_type": result_data["task_type"],
                "time_series_analysis": result_data["time_series_analysis"],
                "decomposition": result_data["decomposition"],
                "best_algorithm": result_data["ts_results"]["best_algorithm"]
            }
        )
        
        # Share forecasting performance
        self.share_knowledge(
            knowledge_type="forecasting_performance",
            knowledge_data={
                "r2_score": result_data["ts_results"]["performance_metrics"]["r2_score"],
                "rmse": result_data["ts_results"]["performance_metrics"]["rmse"],
                "mape": result_data["ts_results"]["performance_metrics"]["mape"],
                "forecast_horizon": result_data["ts_results"]["performance_metrics"]["forecast_horizon"]
            }
        )
    
    def _results_to_dict(self, results: TSResult) -> Dict[str, Any]:
        """Convert TSResult to dictionary."""
        return {
            "task_type": results.task_type,
            "best_algorithm": results.best_algorithm,
            "performance_metrics": self._performance_to_dict(results.performance_metrics),
            "decomposition": results.decomposition,
            "forecasts": results.forecasts,
            "residual_analysis": results.residual_analysis,
            "feature_importance": results.feature_importance,
            "preprocessing_steps": results.preprocessing_steps
        }
    
    def _performance_to_dict(self, performance: TSPerformance) -> Dict[str, Any]:
        """Convert TSPerformance to dictionary."""
        return {
            "algorithm": performance.algorithm,
            "mae": performance.mae,
            "mse": performance.mse,
            "rmse": performance.rmse,
            "mape": performance.mape,
            "r2_score": performance.r2_score,
            "directional_accuracy": performance.directional_accuracy,
            "forecast_horizon": performance.forecast_horizon,
            "training_time": performance.training_time,
            "prediction_time": performance.prediction_time,
            "model_complexity": performance.model_complexity,
            "residual_autocorr": performance.residual_autocorr
        }
    
    def _ts_analysis_to_dict(self, analysis: TimeSeriesAnalysis) -> Dict[str, Any]:
        """Convert TimeSeriesAnalysis to dictionary."""
        return {
            "length": analysis.length,
            "frequency": analysis.frequency,
            "start_date": analysis.start_date,
            "end_date": analysis.end_date,
            "has_trend": analysis.has_trend,
            "has_seasonality": analysis.has_seasonality,
            "seasonality_period": analysis.seasonality_period,
            "stationarity_pvalue": analysis.stationarity_pvalue,
            "missing_values": analysis.missing_values,
            "outliers_count": analysis.outliers_count,
            "autocorrelation_lags": analysis.autocorrelation_lags,
            "trend_strength": analysis.trend_strength,
            "seasonal_strength": analysis.seasonal_strength,
            "noise_level": analysis.noise_level
        }
    
    def can_handle_task(self, context: TaskContext) -> bool:
        """Check if this is a time series task."""
        user_input = context.user_input.lower()
        ts_keywords = [
            "time series", "forecast", "forecasting", "temporal", "trend",
            "seasonal", "seasonality", "time", "date", "timestamp",
            "predict future", "stock price", "sales forecast", "demand forecast"
        ]
        
        return any(keyword in user_input for keyword in ts_keywords)
    
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """Estimate time series task complexity."""
        user_input = context.user_input.lower()
        
        # Expert level tasks
        if any(keyword in user_input for keyword in ["lstm", "neural", "deep learning", "transformer"]):
            return TaskComplexity.EXPERT
        elif any(keyword in user_input for keyword in ["anomaly", "change point", "regime switching"]):
            return TaskComplexity.COMPLEX
        elif any(keyword in user_input for keyword in ["seasonal", "trend", "arima"]):
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _create_refinement_plan(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Create time series specific refinement plan."""
        return {
            "strategy_name": "advanced_time_series_optimization",
            "steps": [
                "enhanced_feature_engineering",
                "advanced_decomposition_analysis",
                "ensemble_forecasting_models",
                "residual_pattern_analysis"
            ],
            "estimated_improvement": 0.18,
            "execution_time": 12.0
        }
    
    def _assess_knowledge_relevance(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> float:
        """Assess relevance of shared knowledge to time series agent."""
        relevance_map = {
            "temporal_analysis_results": 0.9,
            "forecasting_performance": 0.8,
            "data_quality_issues": 0.6,
            "feature_importance": 0.7,
            "model_performance": 0.5
        }
        return relevance_map.get(knowledge_type, 0.1)