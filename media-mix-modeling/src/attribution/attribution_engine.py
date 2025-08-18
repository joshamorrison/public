"""
Attribution Engine for Media Mix Modeling
Orchestrates attribution analysis, model integration, and business insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .attribution_analyzer import AttributionAnalyzer

class AttributionEngine:
    """
    Comprehensive attribution engine that orchestrates multiple attribution methods
    and integrates with MMM models for enhanced insights
    
    Features:
    - Multi-methodology attribution analysis
    - MMM model integration
    - Business insights generation
    - Scenario planning and optimization
    - Performance tracking and reporting
    """
    
    def __init__(self, 
                 attribution_window_days: int = 30,
                 mmm_model: Optional[Any] = None,
                 enable_advanced_analytics: bool = True):
        """
        Initialize attribution engine
        
        Args:
            attribution_window_days: Attribution window in days
            mmm_model: Optional MMM model for enhanced attribution
            enable_advanced_analytics: Enable advanced analytics features
        """
        self.attribution_window_days = attribution_window_days
        self.mmm_model = mmm_model
        self.enable_advanced_analytics = enable_advanced_analytics
        
        # Initialize attribution analyzer
        self.analyzer = AttributionAnalyzer(attribution_window_days)
        
        # Engine state
        self.last_analysis_date = None
        self.attribution_history = []
        self.performance_trends = {}
        self.business_insights = {}
        
        print(f"[ATTRIBUTION ENGINE] Initialized with {attribution_window_days}-day window")
    
    def run_comprehensive_attribution(self, 
                                    data: pd.DataFrame,
                                    spend_columns: List[str],
                                    revenue_column: str,
                                    date_column: str = 'date',
                                    include_mmm_insights: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive attribution analysis with all available methods
        
        Args:
            data: Historical marketing data
            spend_columns: List of media spend column names
            revenue_column: Revenue column name
            date_column: Date column name
            include_mmm_insights: Whether to include MMM model insights
            
        Returns:
            Comprehensive attribution analysis results
        """
        print(f"[ATTRIBUTION ENGINE] Running comprehensive attribution analysis")
        
        # Run standard attribution analysis
        attribution_results = self.analyzer.run_attribution_analysis(
            data, spend_columns, revenue_column, date_column
        )
        
        # Add MMM-based insights if model is available
        if include_mmm_insights and self.mmm_model is not None:
            mmm_insights = self._integrate_mmm_insights(data, spend_columns, revenue_column)
            attribution_results['mmm_insights'] = mmm_insights
        
        # Generate advanced analytics if enabled
        if self.enable_advanced_analytics:
            advanced_analytics = self._generate_advanced_analytics(
                data, spend_columns, revenue_column, attribution_results
            )
            attribution_results['advanced_analytics'] = advanced_analytics
        
        # Update attribution history
        self._update_attribution_history(attribution_results)
        
        # Generate business insights
        business_insights = self._generate_business_insights(attribution_results, spend_columns)
        attribution_results['business_insights'] = business_insights
        
        # Update last analysis date
        self.last_analysis_date = datetime.now()
        
        print(f"[ATTRIBUTION ENGINE] Analysis completed with {len(attribution_results)} components")
        
        return attribution_results
    
    def _integrate_mmm_insights(self, 
                               data: pd.DataFrame,
                               spend_columns: List[str],
                               revenue_column: str) -> Dict[str, Any]:
        """Integrate MMM model insights with attribution analysis"""
        
        mmm_insights = {
            'mmm_attribution': {},
            'incrementality_analysis': {},
            'saturation_analysis': {},
            'adstock_effects': {}
        }
        
        try:
            # Get MMM attribution results if available
            if hasattr(self.mmm_model, 'attribution_results'):
                mmm_insights['mmm_attribution'] = self.mmm_model.attribution_results
            
            # Calculate incrementality using MMM predictions
            if hasattr(self.mmm_model, 'predict'):
                incrementality = self._calculate_incrementality(data, spend_columns, revenue_column)
                mmm_insights['incrementality_analysis'] = incrementality
            
            # Analyze saturation effects
            if hasattr(self.mmm_model, 'saturation_params'):
                saturation_analysis = self._analyze_saturation_effects(data, spend_columns)
                mmm_insights['saturation_analysis'] = saturation_analysis
            
            # Analyze adstock effects
            if hasattr(self.mmm_model, 'adstock_params'):
                adstock_analysis = self._analyze_adstock_effects(data, spend_columns)
                mmm_insights['adstock_effects'] = adstock_analysis
            
            print(f"[MMM INTEGRATION] Generated insights for {len(spend_columns)} channels")
            
        except Exception as e:
            print(f"[MMM INTEGRATION] Error integrating MMM insights: {e}")
            mmm_insights['error'] = str(e)
        
        return mmm_insights
    
    def _calculate_incrementality(self, 
                                 data: pd.DataFrame,
                                 spend_columns: List[str],
                                 revenue_column: str) -> Dict[str, Any]:
        """Calculate incrementality for each channel using MMM model"""
        
        incrementality = {}
        
        for channel in spend_columns:
            try:
                # Create scenario with channel spend set to zero
                zero_spend_data = data.copy()
                zero_spend_data[channel] = 0
                
                # Get predictions for baseline and zero spend scenarios
                baseline_predictions = self.mmm_model.predict(data)
                zero_spend_predictions = self.mmm_model.predict(zero_spend_data)
                
                # Calculate incremental revenue
                incremental_revenue = baseline_predictions.sum() - zero_spend_predictions.sum()
                total_spend = data[channel].sum()
                
                incrementality[channel] = {
                    'incremental_revenue': incremental_revenue,
                    'total_spend': total_spend,
                    'incremental_roi': incremental_revenue / total_spend if total_spend > 0 else 0,
                    'incrementality_percentage': incremental_revenue / baseline_predictions.sum() if baseline_predictions.sum() > 0 else 0
                }
                
            except Exception as e:
                print(f"[INCREMENTALITY] Error calculating for {channel}: {e}")
                incrementality[channel] = {'error': str(e)}
        
        return incrementality
    
    def _analyze_saturation_effects(self, 
                                   data: pd.DataFrame,
                                   spend_columns: List[str]) -> Dict[str, Any]:
        """Analyze saturation effects for each channel"""
        
        saturation_analysis = {}
        
        for channel in spend_columns:
            try:
                # Get saturation parameters if available
                saturation_param = getattr(self.mmm_model, 'saturation_params', {}).get(f"{channel}_saturated", {})
                
                # Calculate current saturation level
                current_spend = data[channel].iloc[-1] if len(data) > 0 else 0
                max_spend = data[channel].max()
                
                # Estimate saturation point (simplified)
                estimated_saturation_point = max_spend * 1.5  # 150% of historical max
                current_saturation_level = current_spend / estimated_saturation_point if estimated_saturation_point > 0 else 0
                
                saturation_analysis[channel] = {
                    'saturation_parameter': saturation_param.get('param', self.mmm_model.saturation_param),
                    'current_saturation_level': min(current_saturation_level, 1.0),
                    'estimated_saturation_point': estimated_saturation_point,
                    'room_for_growth': max(0, 1.0 - current_saturation_level),
                    'saturation_status': self._get_saturation_status(current_saturation_level)
                }
                
            except Exception as e:
                saturation_analysis[channel] = {'error': str(e)}
        
        return saturation_analysis
    
    def _analyze_adstock_effects(self, 
                               data: pd.DataFrame,
                               spend_columns: List[str]) -> Dict[str, Any]:
        """Analyze adstock (carryover) effects for each channel"""
        
        adstock_analysis = {}
        
        for channel in spend_columns:
            try:
                # Get adstock parameters
                adstock_rate = getattr(self.mmm_model, 'adstock_params', {}).get(channel, self.mmm_model.adstock_rate)
                
                # Calculate carryover duration (half-life)
                half_life_periods = np.log(0.5) / np.log(adstock_rate) if adstock_rate > 0 and adstock_rate != 1 else 0
                
                # Calculate current carryover value
                recent_spend = data[channel].iloc[-7:] if len(data) >= 7 else data[channel]  # Last 7 periods
                carryover_weights = [adstock_rate ** i for i in range(len(recent_spend))]
                current_carryover = sum(spend * weight for spend, weight in zip(recent_spend, reversed(carryover_weights)))
                
                adstock_analysis[channel] = {
                    'adstock_rate': adstock_rate,
                    'half_life_periods': half_life_periods,
                    'current_carryover_value': current_carryover,
                    'carryover_strength': self._get_carryover_strength(adstock_rate),
                    'impact_duration': f"{int(half_life_periods * 2)} periods" if half_life_periods > 0 else "No carryover"
                }
                
            except Exception as e:
                adstock_analysis[channel] = {'error': str(e)}
        
        return adstock_analysis
    
    def _generate_advanced_analytics(self, 
                                   data: pd.DataFrame,
                                   spend_columns: List[str],
                                   revenue_column: str,
                                   attribution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advanced analytics and insights"""
        
        advanced_analytics = {
            'channel_efficiency_analysis': {},
            'trend_analysis': {},
            'seasonality_patterns': {},
            'correlation_analysis': {},
            'optimization_recommendations': []
        }
        
        try:
            # Channel efficiency analysis
            advanced_analytics['channel_efficiency_analysis'] = self._analyze_channel_efficiency(
                data, spend_columns, revenue_column
            )
            
            # Trend analysis
            advanced_analytics['trend_analysis'] = self._analyze_trends(
                data, spend_columns, revenue_column
            )
            
            # Seasonality patterns
            if 'date' in data.columns:
                advanced_analytics['seasonality_patterns'] = self._analyze_seasonality(
                    data, spend_columns, revenue_column
                )
            
            # Correlation analysis
            advanced_analytics['correlation_analysis'] = self._analyze_correlations(
                data, spend_columns, revenue_column
            )
            
            # Generate optimization recommendations
            advanced_analytics['optimization_recommendations'] = self._generate_optimization_recommendations(
                attribution_results, advanced_analytics
            )
            
        except Exception as e:
            print(f"[ADVANCED ANALYTICS] Error generating analytics: {e}")
            advanced_analytics['error'] = str(e)
        
        return advanced_analytics
    
    def _analyze_channel_efficiency(self, 
                                  data: pd.DataFrame,
                                  spend_columns: List[str],
                                  revenue_column: str) -> Dict[str, Any]:
        """Analyze efficiency metrics for each channel"""
        
        efficiency_analysis = {}
        
        for channel in spend_columns:
            total_spend = data[channel].sum()
            total_revenue = data[revenue_column].sum()
            
            # Calculate basic efficiency metrics
            if total_spend > 0:
                roi = total_revenue / total_spend
                spend_share = total_spend / data[spend_columns].sum().sum()
                
                # Calculate efficiency trend (last 30% vs first 30% of data)
                split_point = int(len(data) * 0.7)
                early_roi = (data[revenue_column][:split_point].sum() / data[channel][:split_point].sum()) if data[channel][:split_point].sum() > 0 else 0
                recent_roi = (data[revenue_column][split_point:].sum() / data[channel][split_point:].sum()) if data[channel][split_point:].sum() > 0 else 0
                
                efficiency_analysis[channel] = {
                    'roi': roi,
                    'spend_share': spend_share,
                    'efficiency_trend': recent_roi - early_roi,
                    'efficiency_status': 'improving' if recent_roi > early_roi else 'declining',
                    'cost_per_acquisition': total_spend / len(data) if len(data) > 0 else 0
                }
            else:
                efficiency_analysis[channel] = {
                    'roi': 0,
                    'spend_share': 0,
                    'efficiency_trend': 0,
                    'efficiency_status': 'no_spend',
                    'cost_per_acquisition': 0
                }
        
        return efficiency_analysis
    
    def _analyze_trends(self, 
                       data: pd.DataFrame,
                       spend_columns: List[str],
                       revenue_column: str) -> Dict[str, Any]:
        """Analyze trends in spend and revenue"""
        
        trend_analysis = {}
        
        # Calculate moving averages for trend detection
        window_size = min(7, len(data) // 4)  # 7-period or 25% of data
        
        for channel in spend_columns + [revenue_column]:
            if len(data) >= window_size:
                moving_avg = data[channel].rolling(window=window_size).mean()
                trend_slope = np.polyfit(range(len(moving_avg.dropna())), moving_avg.dropna(), 1)[0]
                
                trend_analysis[channel] = {
                    'trend_slope': trend_slope,
                    'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
                    'trend_strength': abs(trend_slope),
                    'volatility': data[channel].std() / data[channel].mean() if data[channel].mean() > 0 else 0
                }
            else:
                trend_analysis[channel] = {
                    'trend_slope': 0,
                    'trend_direction': 'insufficient_data',
                    'trend_strength': 0,
                    'volatility': 0
                }
        
        return trend_analysis
    
    def _analyze_seasonality(self, 
                           data: pd.DataFrame,
                           spend_columns: List[str],
                           revenue_column: str) -> Dict[str, Any]:
        """Analyze seasonal patterns in data"""
        
        seasonality_analysis = {}
        
        try:
            # Convert date column to datetime if needed
            if 'date' in data.columns:
                data_copy = data.copy()
                data_copy['date'] = pd.to_datetime(data_copy['date'])
                data_copy['month'] = data_copy['date'].dt.month
                data_copy['day_of_week'] = data_copy['date'].dt.dayofweek
                
                # Monthly patterns
                monthly_patterns = {}
                for channel in spend_columns + [revenue_column]:
                    monthly_avg = data_copy.groupby('month')[channel].mean()
                    monthly_patterns[channel] = {
                        'peak_month': monthly_avg.idxmax(),
                        'low_month': monthly_avg.idxmin(),
                        'seasonality_ratio': monthly_avg.max() / monthly_avg.min() if monthly_avg.min() > 0 else 0
                    }
                
                seasonality_analysis['monthly_patterns'] = monthly_patterns
                
                # Day of week patterns
                dow_patterns = {}
                for channel in spend_columns + [revenue_column]:
                    dow_avg = data_copy.groupby('day_of_week')[channel].mean()
                    dow_patterns[channel] = {
                        'peak_day': dow_avg.idxmax(),
                        'low_day': dow_avg.idxmin(),
                        'weekday_vs_weekend': dow_avg[:5].mean() / dow_avg[5:].mean() if dow_avg[5:].mean() > 0 else 0
                    }
                
                seasonality_analysis['day_of_week_patterns'] = dow_patterns
                
        except Exception as e:
            seasonality_analysis['error'] = str(e)
        
        return seasonality_analysis
    
    def _analyze_correlations(self, 
                            data: pd.DataFrame,
                            spend_columns: List[str],
                            revenue_column: str) -> Dict[str, Any]:
        """Analyze correlations between channels and revenue"""
        
        correlation_analysis = {
            'revenue_correlations': {},
            'channel_correlations': {},
            'cross_channel_synergies': {}
        }
        
        # Revenue correlations
        for channel in spend_columns:
            correlation = data[channel].corr(data[revenue_column])
            correlation_analysis['revenue_correlations'][channel] = {
                'correlation': correlation if not pd.isna(correlation) else 0,
                'correlation_strength': self._get_correlation_strength(correlation if not pd.isna(correlation) else 0)
            }
        
        # Channel correlations
        channel_corr_matrix = data[spend_columns].corr()
        for i, channel1 in enumerate(spend_columns):
            for channel2 in spend_columns[i+1:]:
                correlation = channel_corr_matrix.loc[channel1, channel2]
                if not pd.isna(correlation):
                    correlation_analysis['channel_correlations'][f"{channel1}_{channel2}"] = {
                        'correlation': correlation,
                        'relationship_type': self._get_relationship_type(correlation)
                    }
        
        # Cross-channel synergies (simplified)
        for i, channel1 in enumerate(spend_columns):
            for channel2 in spend_columns[i+1:]:
                # Calculate combined effect vs individual effects
                combined_spend = data[channel1] + data[channel2]
                combined_correlation = combined_spend.corr(data[revenue_column])
                individual_correlations = [
                    data[channel1].corr(data[revenue_column]),
                    data[channel2].corr(data[revenue_column])
                ]
                
                synergy_score = combined_correlation - max(individual_correlations) if not any(pd.isna(individual_correlations)) else 0
                correlation_analysis['cross_channel_synergies'][f"{channel1}_{channel2}"] = {
                    'synergy_score': synergy_score if not pd.isna(synergy_score) else 0,
                    'synergy_strength': self._get_synergy_strength(synergy_score if not pd.isna(synergy_score) else 0)
                }
        
        return correlation_analysis
    
    def _generate_optimization_recommendations(self, 
                                             attribution_results: Dict[str, Any],
                                             advanced_analytics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable optimization recommendations"""
        
        recommendations = []
        
        try:
            # Get attribution insights
            insights = attribution_results.get('insights', {})
            top_performers = insights.get('top_performers', [])
            optimization_opportunities = insights.get('optimization_opportunities', [])
            
            # Add recommendations based on attribution analysis
            for opportunity in optimization_opportunities:
                recommendations.append({
                    'type': 'attribution_based',
                    'priority': 'high' if opportunity.get('issue') == 'low_attribution' else 'medium',
                    'recommendation': opportunity.get('recommendation', ''),
                    'channel': opportunity.get('channel', ''),
                    'rationale': f"Attribution analysis identified {opportunity.get('issue', 'issue')}"
                })
            
            # Add efficiency-based recommendations
            efficiency_analysis = advanced_analytics.get('channel_efficiency_analysis', {})
            for channel, efficiency in efficiency_analysis.items():
                if efficiency.get('efficiency_status') == 'declining':
                    recommendations.append({
                        'type': 'efficiency_based',
                        'priority': 'medium',
                        'recommendation': f'Investigate declining efficiency for {channel}',
                        'channel': channel,
                        'rationale': f'ROI trend is negative: {efficiency.get("efficiency_trend", 0):.3f}'
                    })
                elif efficiency.get('roi', 0) > 3.0:  # High ROI
                    recommendations.append({
                        'type': 'scaling_opportunity',
                        'priority': 'high',
                        'recommendation': f'Consider increasing budget allocation to {channel}',
                        'channel': channel,
                        'rationale': f'High ROI of {efficiency.get("roi", 0):.1f} suggests scaling opportunity'
                    })
            
            # Add trend-based recommendations
            trend_analysis = advanced_analytics.get('trend_analysis', {})
            for channel, trend in trend_analysis.items():
                if trend.get('trend_direction') == 'decreasing' and trend.get('trend_strength', 0) > 0.1:
                    recommendations.append({
                        'type': 'trend_based',
                        'priority': 'medium',
                        'recommendation': f'Address declining trend in {channel}',
                        'channel': channel,
                        'rationale': f'Strong declining trend detected (slope: {trend.get("trend_slope", 0):.3f})'
                    })
            
        except Exception as e:
            recommendations.append({
                'type': 'error',
                'priority': 'low',
                'recommendation': 'Error generating recommendations',
                'channel': 'system',
                'rationale': f'Error: {e}'
            })
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _update_attribution_history(self, attribution_results: Dict[str, Any]):
        """Update attribution history for trend tracking"""
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'attribution_results': attribution_results.get('attribution_results', {}),
            'performance_summary': {
                'total_channels': len(attribution_results.get('metadata', {}).get('channels', [])),
                'analysis_method': 'comprehensive'
            }
        }
        
        self.attribution_history.append(history_entry)
        
        # Keep only last 50 entries
        if len(self.attribution_history) > 50:
            self.attribution_history = self.attribution_history[-50:]
    
    def _generate_business_insights(self, 
                                  attribution_results: Dict[str, Any],
                                  spend_columns: List[str]) -> Dict[str, Any]:
        """Generate business-focused insights and recommendations"""
        
        business_insights = {
            'executive_summary': {},
            'budget_recommendations': {},
            'performance_alerts': [],
            'strategic_insights': []
        }
        
        try:
            # Executive summary
            attribution_data = attribution_results.get('attribution_results', {})
            if attribution_data:
                # Get consensus attribution (data-driven method)
                consensus_attribution = attribution_data.get('data_driven', {})
                top_channel = max(consensus_attribution.items(), key=lambda x: x[1]) if consensus_attribution else ('unknown', 0)
                
                business_insights['executive_summary'] = {
                    'top_performing_channel': top_channel[0],
                    'top_channel_attribution': top_channel[1],
                    'total_channels_analyzed': len(spend_columns),
                    'attribution_confidence': self._calculate_attribution_confidence(attribution_results),
                    'key_finding': f"{top_channel[0]} drives {top_channel[1]:.1%} of attributed revenue"
                }
            
            # Budget recommendations
            if 'insights' in attribution_results:
                optimization_opportunities = attribution_results['insights'].get('optimization_opportunities', [])
                budget_recommendations = {}
                
                for opportunity in optimization_opportunities:
                    channel = opportunity.get('channel', '')
                    if opportunity.get('issue') == 'low_attribution':
                        budget_recommendations[channel] = 'decrease'
                    elif opportunity.get('issue') == 'high_roi':
                        budget_recommendations[channel] = 'increase'
                
                business_insights['budget_recommendations'] = budget_recommendations
            
            # Performance alerts
            comparison = attribution_results.get('comparison', {})
            if 'channel_comparison' in comparison:
                for channel, channel_data in comparison['channel_comparison'].items():
                    std_dev = channel_data.get('std', 0)
                    if std_dev > 0.2:  # High variance across methods
                        business_insights['performance_alerts'].append({
                            'channel': channel,
                            'alert_type': 'high_variance',
                            'message': f'{channel} shows inconsistent attribution across methods',
                            'severity': 'medium'
                        })
            
            # Strategic insights
            business_insights['strategic_insights'] = [
                'Focus on data-driven attribution for strategic decisions',
                'Monitor attribution stability across different methods',
                'Consider incrementality testing for high-attribution channels',
                'Implement continuous attribution monitoring'
            ]
            
        except Exception as e:
            business_insights['error'] = str(e)
        
        return business_insights
    
    def _calculate_attribution_confidence(self, attribution_results: Dict[str, Any]) -> float:
        """Calculate confidence score for attribution results"""
        
        try:
            comparison = attribution_results.get('comparison', {})
            channel_comparison = comparison.get('channel_comparison', {})
            
            if not channel_comparison:
                return 0.5
            
            # Calculate average standard deviation across channels
            std_devs = [channel_data.get('std', 1.0) for channel_data in channel_comparison.values()]
            avg_std = np.mean(std_devs)
            
            # Convert to confidence score (lower std = higher confidence)
            confidence = max(0, 1 - avg_std)
            
            return confidence
            
        except Exception:
            return 0.5
    
    # Utility methods for categorization
    def _get_saturation_status(self, saturation_level: float) -> str:
        """Get saturation status category"""
        if saturation_level < 0.3:
            return 'low'
        elif saturation_level < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def _get_carryover_strength(self, adstock_rate: float) -> str:
        """Get carryover strength category"""
        if adstock_rate < 0.2:
            return 'weak'
        elif adstock_rate < 0.6:
            return 'moderate'
        else:
            return 'strong'
    
    def _get_correlation_strength(self, correlation: float) -> str:
        """Get correlation strength category"""
        abs_corr = abs(correlation)
        if abs_corr < 0.3:
            return 'weak'
        elif abs_corr < 0.7:
            return 'moderate'
        else:
            return 'strong'
    
    def _get_relationship_type(self, correlation: float) -> str:
        """Get relationship type between channels"""
        if correlation > 0.5:
            return 'complementary'
        elif correlation < -0.5:
            return 'competitive'
        else:
            return 'independent'
    
    def _get_synergy_strength(self, synergy_score: float) -> str:
        """Get synergy strength category"""
        if synergy_score > 0.1:
            return 'positive'
        elif synergy_score < -0.1:
            return 'negative'
        else:
            return 'neutral'

# Convenience functions for easy usage
def run_attribution_engine_analysis(data: pd.DataFrame, 
                                   spend_columns: List[str],
                                   revenue_column: str,
                                   mmm_model: Optional[Any] = None) -> Dict[str, Any]:
    """
    Run comprehensive attribution analysis using the Attribution Engine
    
    Args:
        data: Marketing data DataFrame
        spend_columns: List of spend column names
        revenue_column: Revenue column name
        mmm_model: Optional MMM model for enhanced insights
        
    Returns:
        Comprehensive attribution analysis results
    """
    engine = AttributionEngine(mmm_model=mmm_model)
    return engine.run_comprehensive_attribution(data, spend_columns, revenue_column)

def create_attribution_dashboard_data(attribution_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create dashboard-ready data from attribution results
    
    Args:
        attribution_results: Results from attribution engine
        
    Returns:
        Dashboard-ready data structure
    """
    dashboard_data = {
        'summary': attribution_results.get('business_insights', {}).get('executive_summary', {}),
        'attribution_chart_data': attribution_results.get('attribution_results', {}),
        'recommendations': attribution_results.get('business_insights', {}).get('budget_recommendations', {}),
        'alerts': attribution_results.get('business_insights', {}).get('performance_alerts', []),
        'insights': attribution_results.get('insights', {})
    }
    
    return dashboard_data

if __name__ == "__main__":
    print("Attribution Engine for Media Mix Modeling")
    print("Usage: from src.attribution.attribution_engine import AttributionEngine")