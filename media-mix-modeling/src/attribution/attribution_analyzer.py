"""
Attribution Analysis Engine for Media Mix Modeling
Implements multiple attribution methodologies for comprehensive channel evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AttributionAnalyzer:
    """
    Multi-methodology attribution analysis for marketing channels
    
    Supports:
    - Last-touch attribution
    - First-touch attribution  
    - Linear attribution
    - Time-decay attribution
    - Data-driven attribution (MMM-based)
    """
    
    def __init__(self, attribution_window_days: int = 30):
        """
        Initialize attribution analyzer
        
        Args:
            attribution_window_days: Attribution window in days
        """
        self.attribution_window_days = attribution_window_days
        self.attribution_methods = {
            'last_touch': self._last_touch_attribution,
            'first_touch': self._first_touch_attribution,
            'linear': self._linear_attribution,
            'time_decay': self._time_decay_attribution,
            'data_driven': self._data_driven_attribution
        }
    
    def run_attribution_analysis(self, data: pd.DataFrame, spend_columns: List[str], 
                                revenue_column: str, date_column: str = 'date') -> Dict[str, Any]:
        """
        Run comprehensive attribution analysis across multiple methods
        
        Args:
            data: Historical marketing data
            spend_columns: List of media spend column names
            revenue_column: Revenue column name
            date_column: Date column name
            
        Returns:
            Dictionary with attribution results from all methods
        """
        print(f"[ATTRIBUTION] Running attribution analysis for {len(spend_columns)} channels")
        
        # Validate data
        self._validate_data(data, spend_columns, revenue_column, date_column)
        
        # Prepare data
        prepared_data = self._prepare_attribution_data(data, spend_columns, revenue_column, date_column)
        
        # Run all attribution methods
        attribution_results = {}
        
        for method_name, method_func in self.attribution_methods.items():
            try:
                print(f"[ATTRIBUTION] Computing {method_name} attribution...")
                attribution = method_func(prepared_data, spend_columns, revenue_column)
                attribution_results[method_name] = attribution
                
                # Log summary
                total_attributed = sum(attribution.values())
                print(f"[ATTRIBUTION] {method_name}: {total_attributed:.1%} total attribution")
                
            except Exception as e:
                print(f"[ATTRIBUTION] Warning: {method_name} failed: {e}")
                attribution_results[method_name] = {channel: 0.0 for channel in spend_columns}
        
        # Calculate attribution comparison
        comparison = self._compare_attribution_methods(attribution_results, spend_columns)
        
        # Generate insights
        insights = self._generate_attribution_insights(attribution_results, spend_columns)
        
        return {
            'attribution_results': attribution_results,
            'comparison': comparison,
            'insights': insights,
            'metadata': {
                'channels': spend_columns,
                'attribution_window_days': self.attribution_window_days,
                'analysis_date': datetime.now().isoformat(),
                'data_period': {
                    'start': data[date_column].min(),
                    'end': data[date_column].max(),
                    'records': len(data)
                }
            }
        }
    
    def _validate_data(self, data: pd.DataFrame, spend_columns: List[str], 
                      revenue_column: str, date_column: str):
        """Validate input data for attribution analysis"""
        
        # Check required columns exist
        missing_columns = []
        for col in spend_columns + [revenue_column, date_column]:
            if col not in data.columns:
                missing_columns.append(col)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for null values
        null_columns = []
        for col in spend_columns + [revenue_column]:
            if data[col].isnull().any():
                null_columns.append(col)
        
        if null_columns:
            print(f"[ATTRIBUTION] Warning: Null values found in {null_columns}")
        
        # Check data length
        if len(data) < 7:
            raise ValueError("Insufficient data for attribution analysis (minimum 7 records)")
        
        print(f"[ATTRIBUTION] Data validation passed: {len(data)} records, {len(spend_columns)} channels")
    
    def _prepare_attribution_data(self, data: pd.DataFrame, spend_columns: List[str], 
                                 revenue_column: str, date_column: str) -> pd.DataFrame:
        """Prepare data for attribution analysis"""
        
        # Create working copy
        attribution_data = data.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(attribution_data[date_column]):
            attribution_data[date_column] = pd.to_datetime(attribution_data[date_column])
        
        # Sort by date
        attribution_data = attribution_data.sort_values(date_column).reset_index(drop=True)
        
        # Fill null values with 0 for spend columns
        for col in spend_columns:
            attribution_data[col] = attribution_data[col].fillna(0)
        
        # Fill null values with forward fill for revenue
        attribution_data[revenue_column] = attribution_data[revenue_column].fillna(method='ffill')
        
        # Calculate period-over-period changes
        attribution_data['revenue_change'] = attribution_data[revenue_column].diff()
        
        # Calculate spend ratios
        total_spend_per_period = attribution_data[spend_columns].sum(axis=1)
        for col in spend_columns:
            attribution_data[f'{col}_ratio'] = (
                attribution_data[col] / total_spend_per_period
            ).fillna(0)
        
        return attribution_data
    
    def _last_touch_attribution(self, data: pd.DataFrame, spend_columns: List[str], 
                               revenue_column: str) -> Dict[str, float]:
        """
        Last-touch attribution: 100% credit to the last non-zero spend channel
        """
        attribution = {channel: 0.0 for channel in spend_columns}
        total_revenue = data[revenue_column].sum()
        
        if total_revenue <= 0:
            return attribution
        
        # For each period, find the last channel with spend and give it full credit
        for idx, row in data.iterrows():
            period_revenue = row[revenue_column]
            if period_revenue > 0:
                # Find last channel with spend (moving backwards through columns)
                last_channel = None
                for channel in reversed(spend_columns):
                    if row[channel] > 0:
                        last_channel = channel
                        break
                
                if last_channel:
                    attribution[last_channel] += period_revenue
        
        # Normalize to percentages
        total_attributed = sum(attribution.values())
        if total_attributed > 0:
            attribution = {channel: value / total_attributed for channel, value in attribution.items()}
        
        return attribution
    
    def _first_touch_attribution(self, data: pd.DataFrame, spend_columns: List[str], 
                                revenue_column: str) -> Dict[str, float]:
        """
        First-touch attribution: 100% credit to the first non-zero spend channel
        """
        attribution = {channel: 0.0 for channel in spend_columns}
        total_revenue = data[revenue_column].sum()
        
        if total_revenue <= 0:
            return attribution
        
        # For each period, find the first channel with spend and give it full credit
        for idx, row in data.iterrows():
            period_revenue = row[revenue_column]
            if period_revenue > 0:
                # Find first channel with spend
                first_channel = None
                for channel in spend_columns:
                    if row[channel] > 0:
                        first_channel = channel
                        break
                
                if first_channel:
                    attribution[first_channel] += period_revenue
        
        # Normalize to percentages
        total_attributed = sum(attribution.values())
        if total_attributed > 0:
            attribution = {channel: value / total_attributed for channel, value in attribution.items()}
        
        return attribution
    
    def _linear_attribution(self, data: pd.DataFrame, spend_columns: List[str], 
                           revenue_column: str) -> Dict[str, float]:
        """
        Linear attribution: Equal credit distribution across active channels
        """
        attribution = {channel: 0.0 for channel in spend_columns}
        total_revenue = data[revenue_column].sum()
        
        if total_revenue <= 0:
            return attribution
        
        # For each period, distribute revenue equally among active channels
        for idx, row in data.iterrows():
            period_revenue = row[revenue_column]
            if period_revenue > 0:
                # Find active channels (with spend > 0)
                active_channels = [channel for channel in spend_columns if row[channel] > 0]
                
                if active_channels:
                    # Equal distribution
                    credit_per_channel = period_revenue / len(active_channels)
                    for channel in active_channels:
                        attribution[channel] += credit_per_channel
        
        # Normalize to percentages
        total_attributed = sum(attribution.values())
        if total_attributed > 0:
            attribution = {channel: value / total_attributed for channel, value in attribution.items()}
        
        return attribution
    
    def _time_decay_attribution(self, data: pd.DataFrame, spend_columns: List[str], 
                               revenue_column: str, decay_rate: float = 0.1) -> Dict[str, float]:
        """
        Time-decay attribution: Exponential decay based on recency
        """
        attribution = {channel: 0.0 for channel in spend_columns}
        total_revenue = data[revenue_column].sum()
        
        if total_revenue <= 0:
            return attribution
        
        # Calculate time weights using exponential decay
        data_with_weights = data.copy()
        data_with_weights['time_weight'] = np.exp(-decay_rate * np.arange(len(data)))
        
        # For each period, distribute revenue based on time-weighted spend
        for idx, row in data_with_weights.iterrows():
            period_revenue = row[revenue_column]
            time_weight = row['time_weight']
            
            if period_revenue > 0:
                # Calculate weighted spend for each channel
                weighted_spends = {}
                total_weighted_spend = 0
                
                for channel in spend_columns:
                    weighted_spend = row[channel] * time_weight
                    weighted_spends[channel] = weighted_spend
                    total_weighted_spend += weighted_spend
                
                # Distribute revenue proportionally to weighted spend
                if total_weighted_spend > 0:
                    for channel in spend_columns:
                        weight_ratio = weighted_spends[channel] / total_weighted_spend
                        attribution[channel] += period_revenue * weight_ratio
        
        # Normalize to percentages
        total_attributed = sum(attribution.values())
        if total_attributed > 0:
            attribution = {channel: value / total_attributed for channel, value in attribution.items()}
        
        return attribution
    
    def _data_driven_attribution(self, data: pd.DataFrame, spend_columns: List[str], 
                                revenue_column: str) -> Dict[str, float]:
        """
        Data-driven attribution: Based on statistical contribution analysis
        """
        attribution = {channel: 0.0 for channel in spend_columns}
        
        try:
            # Simple correlation-based attribution as baseline
            correlations = {}
            
            for channel in spend_columns:
                # Calculate correlation between spend and revenue
                correlation = data[channel].corr(data[revenue_column])
                correlations[channel] = max(0, correlation) if not pd.isna(correlation) else 0
            
            # Normalize correlations to sum to 1
            total_correlation = sum(correlations.values())
            if total_correlation > 0:
                attribution = {channel: corr / total_correlation for channel, corr in correlations.items()}
            else:
                # Fallback to equal distribution
                attribution = {channel: 1.0 / len(spend_columns) for channel in spend_columns}
            
        except Exception as e:
            print(f"[ATTRIBUTION] Data-driven attribution failed, using fallback: {e}")
            # Fallback to spend-based attribution
            total_spend = data[spend_columns].sum().sum()
            if total_spend > 0:
                for channel in spend_columns:
                    channel_spend = data[channel].sum()
                    attribution[channel] = channel_spend / total_spend
            else:
                attribution = {channel: 1.0 / len(spend_columns) for channel in spend_columns}
        
        return attribution
    
    def _compare_attribution_methods(self, attribution_results: Dict[str, Dict[str, float]], 
                                   spend_columns: List[str]) -> Dict[str, Any]:
        """Compare attribution results across different methods"""
        
        comparison = {
            'channel_comparison': {},
            'method_variance': {},
            'consensus_ranking': []
        }
        
        # Channel comparison across methods
        for channel in spend_columns:
            channel_attributions = {}
            for method, results in attribution_results.items():
                channel_attributions[method] = results.get(channel, 0.0)
            
            comparison['channel_comparison'][channel] = {
                'attributions': channel_attributions,
                'mean': np.mean(list(channel_attributions.values())),
                'std': np.std(list(channel_attributions.values())),
                'min': min(channel_attributions.values()),
                'max': max(channel_attributions.values())
            }
        
        # Method variance analysis
        for method in attribution_results.keys():
            method_values = list(attribution_results[method].values())
            comparison['method_variance'][method] = {
                'concentration': np.std(method_values),
                'entropy': -sum(p * np.log(p + 1e-10) for p in method_values if p > 0)
            }
        
        # Consensus ranking based on average attribution
        avg_attributions = {}
        for channel in spend_columns:
            channel_values = [attribution_results[method].get(channel, 0.0) 
                            for method in attribution_results.keys()]
            avg_attributions[channel] = np.mean(channel_values)
        
        comparison['consensus_ranking'] = sorted(
            avg_attributions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return comparison
    
    def _generate_attribution_insights(self, attribution_results: Dict[str, Dict[str, float]], 
                                     spend_columns: List[str]) -> Dict[str, Any]:
        """Generate actionable insights from attribution analysis"""
        
        insights = {
            'top_performers': [],
            'attribution_stability': {},
            'method_recommendations': {},
            'optimization_opportunities': []
        }
        
        # Identify top performing channels
        consensus_scores = {}
        for channel in spend_columns:
            scores = [attribution_results[method].get(channel, 0.0) 
                     for method in attribution_results.keys()]
            consensus_scores[channel] = np.mean(scores)
        
        insights['top_performers'] = sorted(
            consensus_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        # Attribution stability analysis
        for channel in spend_columns:
            channel_scores = [attribution_results[method].get(channel, 0.0) 
                            for method in attribution_results.keys()]
            insights['attribution_stability'][channel] = {
                'coefficient_of_variation': np.std(channel_scores) / (np.mean(channel_scores) + 1e-10),
                'consistent': np.std(channel_scores) < 0.1
            }
        
        # Method recommendations
        for method, results in attribution_results.items():
            top_channel = max(results.items(), key=lambda x: x[1])
            insights['method_recommendations'][method] = {
                'primary_channel': top_channel[0],
                'primary_attribution': top_channel[1],
                'use_case': self._get_method_use_case(method)
            }
        
        # Optimization opportunities
        for channel in spend_columns:
            avg_attribution = consensus_scores[channel]
            if avg_attribution < 0.1:  # Low attribution
                insights['optimization_opportunities'].append({
                    'channel': channel,
                    'issue': 'low_attribution',
                    'recommendation': 'Consider reallocating budget or improving targeting'
                })
            
            # Check for high variance (inconsistent attribution)
            channel_scores = [attribution_results[method].get(channel, 0.0) 
                            for method in attribution_results.keys()]
            if np.std(channel_scores) > 0.2:
                insights['optimization_opportunities'].append({
                    'channel': channel,
                    'issue': 'inconsistent_attribution',
                    'recommendation': 'Attribution varies significantly across methods - investigate data quality'
                })
        
        return insights
    
    def _get_method_use_case(self, method: str) -> str:
        """Get recommended use case for attribution method"""
        use_cases = {
            'last_touch': 'Performance marketing with clear conversion events',
            'first_touch': 'Brand awareness and top-of-funnel attribution',
            'linear': 'Multi-touch customer journeys with equal touchpoint value',
            'time_decay': 'Sales cycles where recent touchpoints matter more',
            'data_driven': 'Statistical analysis and predictive modeling'
        }
        return use_cases.get(method, 'General attribution analysis')

# Convenience functions for easy attribution analysis
def quick_attribution_analysis(data: pd.DataFrame, spend_columns: List[str], 
                             revenue_column: str) -> Dict[str, float]:
    """
    Quick attribution analysis using data-driven method
    
    Args:
        data: Marketing data DataFrame
        spend_columns: List of spend column names
        revenue_column: Revenue column name
        
    Returns:
        Dictionary with channel attribution percentages
    """
    analyzer = AttributionAnalyzer()
    results = analyzer.run_attribution_analysis(data, spend_columns, revenue_column)
    return results['attribution_results']['data_driven']

def compare_attribution_methods(data: pd.DataFrame, spend_columns: List[str], 
                               revenue_column: str) -> pd.DataFrame:
    """
    Compare all attribution methods and return results as DataFrame
    
    Args:
        data: Marketing data DataFrame
        spend_columns: List of spend column names  
        revenue_column: Revenue column name
        
    Returns:
        DataFrame comparing attribution across methods
    """
    analyzer = AttributionAnalyzer()
    results = analyzer.run_attribution_analysis(data, spend_columns, revenue_column)
    
    # Convert to DataFrame for easy comparison
    attribution_df = pd.DataFrame(results['attribution_results'])
    attribution_df.index.name = 'channel'
    
    return attribution_df

if __name__ == "__main__":
    print("Attribution Analysis for Media Mix Modeling")
    print("Usage: from src.attribution.attribution_analyzer import AttributionAnalyzer")