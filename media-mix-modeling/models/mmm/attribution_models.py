#!/usr/bin/env python3
"""
Attribution Modeling for Media Mix Models
Advanced attribution algorithms and incrementality testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class AttributionModeler:
    """
    Advanced attribution modeling for media mix analysis
    
    Features:
    - Data-driven attribution modeling
    - First-touch, last-touch, and linear attribution
    - Time-decay attribution
    - Incrementality testing frameworks
    - Cross-channel journey analysis
    """
    
    def __init__(self, attribution_window: int = 30):
        """
        Initialize attribution modeler
        
        Args:
            attribution_window: Attribution window in days
        """
        self.attribution_window = attribution_window
        self.attribution_models = {}
        self.incrementality_results = {}
        
    def calculate_first_touch_attribution(self, 
                                        data: pd.DataFrame,
                                        touchpoint_columns: List[str],
                                        conversion_column: str = 'conversions',
                                        date_column: str = 'date') -> pd.DataFrame:
        """
        Calculate first-touch attribution
        
        Args:
            data: Input dataframe
            touchpoint_columns: List of touchpoint/channel columns
            conversion_column: Conversion events column
            date_column: Date column name
            
        Returns:
            DataFrame with first-touch attribution results
        """
        attribution_data = data.copy()
        attribution_results = []
        
        # Sort by date
        attribution_data = attribution_data.sort_values(date_column)
        
        for idx, row in attribution_data.iterrows():
            if row[conversion_column] > 0:
                # Look back within attribution window
                window_start = pd.to_datetime(row[date_column]) - pd.Timedelta(days=self.attribution_window)
                window_data = attribution_data[
                    (pd.to_datetime(attribution_data[date_column]) >= window_start) &
                    (pd.to_datetime(attribution_data[date_column]) <= pd.to_datetime(row[date_column]))
                ]
                
                # Find first touchpoint in window
                for touchpoint in touchpoint_columns:
                    touchpoint_data = window_data[window_data[touchpoint] > 0]
                    if len(touchpoint_data) > 0:
                        first_touch = touchpoint_data.iloc[0]
                        attribution_results.append({
                            'date': row[date_column],
                            'conversion_date': row[date_column],
                            'attributed_channel': touchpoint,
                            'attribution_value': row[conversion_column],
                            'attribution_method': 'first_touch',
                            'time_to_conversion': (pd.to_datetime(row[date_column]) - 
                                                 pd.to_datetime(first_touch[date_column])).days
                        })
                        break
        
        return pd.DataFrame(attribution_results)
    
    def calculate_last_touch_attribution(self,
                                       data: pd.DataFrame,
                                       touchpoint_columns: List[str],
                                       conversion_column: str = 'conversions',
                                       date_column: str = 'date') -> pd.DataFrame:
        """
        Calculate last-touch attribution
        
        Args:
            data: Input dataframe
            touchpoint_columns: List of touchpoint/channel columns
            conversion_column: Conversion events column
            date_column: Date column name
            
        Returns:
            DataFrame with last-touch attribution results
        """
        attribution_data = data.copy()
        attribution_results = []
        
        # Sort by date
        attribution_data = attribution_data.sort_values(date_column)
        
        for idx, row in attribution_data.iterrows():
            if row[conversion_column] > 0:
                # Look back within attribution window
                window_start = pd.to_datetime(row[date_column]) - pd.Timedelta(days=self.attribution_window)
                window_data = attribution_data[
                    (pd.to_datetime(attribution_data[date_column]) >= window_start) &
                    (pd.to_datetime(attribution_data[date_column]) <= pd.to_datetime(row[date_column]))
                ]
                
                # Find last touchpoint in window
                for touchpoint in reversed(touchpoint_columns):
                    touchpoint_data = window_data[window_data[touchpoint] > 0]
                    if len(touchpoint_data) > 0:
                        last_touch = touchpoint_data.iloc[-1]
                        attribution_results.append({
                            'date': row[date_column],
                            'conversion_date': row[date_column],
                            'attributed_channel': touchpoint,
                            'attribution_value': row[conversion_column],
                            'attribution_method': 'last_touch',
                            'time_to_conversion': (pd.to_datetime(row[date_column]) - 
                                                 pd.to_datetime(last_touch[date_column])).days
                        })
                        break
        
        return pd.DataFrame(attribution_results)
    
    def calculate_linear_attribution(self,
                                   data: pd.DataFrame,
                                   touchpoint_columns: List[str],
                                   conversion_column: str = 'conversions',
                                   date_column: str = 'date') -> pd.DataFrame:
        """
        Calculate linear (equal weight) attribution
        
        Args:
            data: Input dataframe
            touchpoint_columns: List of touchpoint/channel columns
            conversion_column: Conversion events column
            date_column: Date column name
            
        Returns:
            DataFrame with linear attribution results
        """
        attribution_data = data.copy()
        attribution_results = []
        
        # Sort by date
        attribution_data = attribution_data.sort_values(date_column)
        
        for idx, row in attribution_data.iterrows():
            if row[conversion_column] > 0:
                # Look back within attribution window
                window_start = pd.to_datetime(row[date_column]) - pd.Timedelta(days=self.attribution_window)
                window_data = attribution_data[
                    (pd.to_datetime(attribution_data[date_column]) >= window_start) &
                    (pd.to_datetime(attribution_data[date_column]) <= pd.to_datetime(row[date_column]))
                ]
                
                # Find all touchpoints in window
                active_touchpoints = []
                for touchpoint in touchpoint_columns:
                    if window_data[touchpoint].sum() > 0:
                        active_touchpoints.append(touchpoint)
                
                # Distribute conversion value equally among touchpoints
                if active_touchpoints:
                    attribution_per_touchpoint = row[conversion_column] / len(active_touchpoints)
                    
                    for touchpoint in active_touchpoints:
                        attribution_results.append({
                            'date': row[date_column],
                            'conversion_date': row[date_column],
                            'attributed_channel': touchpoint,
                            'attribution_value': attribution_per_touchpoint,
                            'attribution_method': 'linear',
                            'total_touchpoints': len(active_touchpoints)
                        })
        
        return pd.DataFrame(attribution_results)
    
    def calculate_time_decay_attribution(self,
                                       data: pd.DataFrame,
                                       touchpoint_columns: List[str],
                                       conversion_column: str = 'conversions',
                                       date_column: str = 'date',
                                       decay_rate: float = 0.5) -> pd.DataFrame:
        """
        Calculate time-decay attribution
        
        Args:
            data: Input dataframe
            touchpoint_columns: List of touchpoint/channel columns
            conversion_column: Conversion events column
            date_column: Date column name
            decay_rate: Decay rate for time-based weighting
            
        Returns:
            DataFrame with time-decay attribution results
        """
        attribution_data = data.copy()
        attribution_results = []
        
        # Sort by date
        attribution_data = attribution_data.sort_values(date_column)
        
        for idx, row in attribution_data.iterrows():
            if row[conversion_column] > 0:
                # Look back within attribution window
                window_start = pd.to_datetime(row[date_column]) - pd.Timedelta(days=self.attribution_window)
                window_data = attribution_data[
                    (pd.to_datetime(attribution_data[date_column]) >= window_start) &
                    (pd.to_datetime(attribution_data[date_column]) <= pd.to_datetime(row[date_column]))
                ]
                
                # Calculate time-decay weights for each touchpoint
                touchpoint_weights = {}
                total_weight = 0
                
                for _, touch_row in window_data.iterrows():
                    days_before = (pd.to_datetime(row[date_column]) - 
                                 pd.to_datetime(touch_row[date_column])).days
                    
                    for touchpoint in touchpoint_columns:
                        if touch_row[touchpoint] > 0:
                            # Calculate time decay weight
                            weight = np.exp(-decay_rate * days_before)
                            
                            if touchpoint not in touchpoint_weights:
                                touchpoint_weights[touchpoint] = 0
                            touchpoint_weights[touchpoint] += weight
                            total_weight += weight
                
                # Distribute conversion value based on weights
                if total_weight > 0:
                    for touchpoint, weight in touchpoint_weights.items():
                        attribution_value = row[conversion_column] * (weight / total_weight)
                        attribution_results.append({
                            'date': row[date_column],
                            'conversion_date': row[date_column],
                            'attributed_channel': touchpoint,
                            'attribution_value': attribution_value,
                            'attribution_method': 'time_decay',
                            'weight': weight / total_weight,
                            'decay_rate': decay_rate
                        })
        
        return pd.DataFrame(attribution_results)
    
    def calculate_data_driven_attribution(self,
                                        data: pd.DataFrame,
                                        touchpoint_columns: List[str],
                                        conversion_column: str = 'conversions',
                                        date_column: str = 'date') -> Dict[str, Any]:
        """
        Calculate data-driven attribution using machine learning
        
        Args:
            data: Input dataframe
            touchpoint_columns: List of touchpoint/channel columns
            conversion_column: Conversion events column
            date_column: Date column name
            
        Returns:
            Dictionary with attribution model and results
        """
        # Prepare features for machine learning
        feature_data = []
        
        # Sort by date
        sorted_data = data.sort_values(date_column)
        
        for idx, row in sorted_data.iterrows():
            # Look back within attribution window for features
            window_start = pd.to_datetime(row[date_column]) - pd.Timedelta(days=self.attribution_window)
            window_data = sorted_data[
                (pd.to_datetime(sorted_data[date_column]) >= window_start) &
                (pd.to_datetime(sorted_data[date_column]) <= pd.to_datetime(row[date_column]))
            ]
            
            # Create features: sum of touchpoints in window
            features = {}
            for touchpoint in touchpoint_columns:
                features[f"{touchpoint}_sum"] = window_data[touchpoint].sum()
                features[f"{touchpoint}_count"] = (window_data[touchpoint] > 0).sum()
                features[f"{touchpoint}_recency"] = len(window_data) - window_data[window_data[touchpoint] > 0].index.max() if (window_data[touchpoint] > 0).any() else self.attribution_window
            
            features['target'] = row[conversion_column]
            feature_data.append(features)
        
        feature_df = pd.DataFrame(feature_data)
        
        # Train Random Forest model for attribution
        feature_columns = [col for col in feature_df.columns if col != 'target']
        X = feature_df[feature_columns]
        y = feature_df['target']
        
        # Handle missing values
        X = X.fillna(0)
        
        if len(X) > 0 and y.sum() > 0:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            
            # Calculate feature importance (attribution weights)
            feature_importance = dict(zip(feature_columns, rf_model.feature_importances_))
            
            # Group by channel
            channel_attribution = {}
            for touchpoint in touchpoint_columns:
                channel_features = [col for col in feature_columns if col.startswith(touchpoint)]
                channel_attribution[touchpoint] = sum([feature_importance.get(feat, 0) for feat in channel_features])
            
            # Normalize to percentages
            total_attribution = sum(channel_attribution.values())
            if total_attribution > 0:
                channel_attribution = {ch: attr/total_attribution for ch, attr in channel_attribution.items()}
            
            return {
                'model': rf_model,
                'channel_attribution': channel_attribution,
                'feature_importance': feature_importance,
                'model_score': rf_model.score(X, y) if len(X) > 1 else 0,
                'total_conversions': y.sum()
            }
        else:
            return {
                'error': 'Insufficient data for data-driven attribution',
                'channel_attribution': {ch: 1/len(touchpoint_columns) for ch in touchpoint_columns}
            }
    
    def run_incrementality_test(self,
                              data: pd.DataFrame,
                              test_channel: str,
                              test_start_date: str,
                              test_end_date: str,
                              control_channels: List[str],
                              target_column: str = 'conversions') -> Dict[str, Any]:
        """
        Run incrementality test for a specific channel
        
        Args:
            data: Input dataframe
            test_channel: Channel being tested
            test_start_date: Test period start date
            test_end_date: Test period end date
            control_channels: List of control channels
            target_column: Target metric column
            
        Returns:
            Dictionary with incrementality test results
        """
        # Split data into pre-test, test, and post-test periods
        test_start = pd.to_datetime(test_start_date)
        test_end = pd.to_datetime(test_end_date)
        
        pre_test_data = data[pd.to_datetime(data['date']) < test_start]
        test_data = data[
            (pd.to_datetime(data['date']) >= test_start) & 
            (pd.to_datetime(data['date']) <= test_end)
        ]
        post_test_data = data[pd.to_datetime(data['date']) > test_end]
        
        # Calculate baseline performance using pre-test data
        if len(pre_test_data) > 0:
            baseline_target = pre_test_data[target_column].mean()
            baseline_test_channel = pre_test_data[test_channel].mean()
        else:
            baseline_target = 0
            baseline_test_channel = 0
        
        # Calculate test period performance
        test_target = test_data[target_column].mean() if len(test_data) > 0 else 0
        test_channel_spend = test_data[test_channel].mean() if len(test_data) > 0 else 0
        
        # Calculate control channels performance for comparison
        control_performance = {}
        for control_ch in control_channels:
            if control_ch in data.columns:
                pre_control = pre_test_data[control_ch].mean() if len(pre_test_data) > 0 else 0
                test_control = test_data[control_ch].mean() if len(test_data) > 0 else 0
                control_performance[control_ch] = {
                    'pre_test': pre_control,
                    'test_period': test_control,
                    'change_pct': (test_control - pre_control) / pre_control if pre_control > 0 else 0
                }
        
        # Calculate incrementality metrics
        incremental_target = test_target - baseline_target
        incremental_spend = test_channel_spend - baseline_test_channel
        
        # Calculate statistical significance (simplified)
        if len(pre_test_data) > 1 and len(test_data) > 1:
            pre_std = pre_test_data[target_column].std()
            test_std = test_data[target_column].std()
            pooled_std = np.sqrt((pre_std**2 + test_std**2) / 2)
            
            if pooled_std > 0:
                t_stat = abs(incremental_target) / (pooled_std * np.sqrt(1/len(pre_test_data) + 1/len(test_data)))
                significant = t_stat > 1.96  # Simplified 95% confidence
            else:
                significant = False
        else:
            significant = False
        
        return {
            'test_channel': test_channel,
            'test_period': f"{test_start_date} to {test_end_date}",
            'baseline_target': baseline_target,
            'test_target': test_target,
            'incremental_lift': incremental_target,
            'lift_percentage': incremental_target / baseline_target if baseline_target > 0 else 0,
            'baseline_spend': baseline_test_channel,
            'test_spend': test_channel_spend,
            'incremental_spend': incremental_spend,
            'incremental_roi': incremental_target / incremental_spend if incremental_spend > 0 else 0,
            'statistically_significant': significant,
            'control_channels_performance': control_performance,
            'test_periods': {
                'pre_test_observations': len(pre_test_data),
                'test_observations': len(test_data),
                'post_test_observations': len(post_test_data)
            }
        }
    
    def compare_attribution_methods(self,
                                  data: pd.DataFrame,
                                  touchpoint_columns: List[str],
                                  conversion_column: str = 'conversions',
                                  date_column: str = 'date') -> Dict[str, Any]:
        """
        Compare different attribution methods
        
        Args:
            data: Input dataframe
            touchpoint_columns: List of touchpoint/channel columns
            conversion_column: Conversion events column
            date_column: Date column name
            
        Returns:
            Dictionary with comparison results
        """
        methods_results = {}
        
        # Calculate all attribution methods
        methods_results['first_touch'] = self.calculate_first_touch_attribution(
            data, touchpoint_columns, conversion_column, date_column
        )
        
        methods_results['last_touch'] = self.calculate_last_touch_attribution(
            data, touchpoint_columns, conversion_column, date_column
        )
        
        methods_results['linear'] = self.calculate_linear_attribution(
            data, touchpoint_columns, conversion_column, date_column
        )
        
        methods_results['time_decay'] = self.calculate_time_decay_attribution(
            data, touchpoint_columns, conversion_column, date_column
        )
        
        methods_results['data_driven'] = self.calculate_data_driven_attribution(
            data, touchpoint_columns, conversion_column, date_column
        )
        
        # Summarize results by method and channel
        summary = {}
        
        for method_name, method_result in methods_results.items():
            if method_name == 'data_driven':
                # Handle data-driven differently
                summary[method_name] = method_result.get('channel_attribution', {})
            else:
                # Summarize attribution by channel
                if len(method_result) > 0:
                    channel_summary = method_result.groupby('attributed_channel')['attribution_value'].sum().to_dict()
                    total_attribution = sum(channel_summary.values())
                    
                    if total_attribution > 0:
                        summary[method_name] = {ch: attr/total_attribution for ch, attr in channel_summary.items()}
                    else:
                        summary[method_name] = {}
                else:
                    summary[method_name] = {}
        
        return {
            'detailed_results': methods_results,
            'attribution_summary': summary,
            'total_conversions': data[conversion_column].sum(),
            'attribution_window_days': self.attribution_window
        }