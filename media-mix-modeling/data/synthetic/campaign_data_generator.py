#!/usr/bin/env python3
"""
Campaign Data Generator
Advanced synthetic marketing campaign data with realistic MMM effects
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class CampaignDataGenerator:
    """
    Generate realistic synthetic marketing campaign data
    
    Features:
    - Multi-channel campaign simulation
    - Seasonality and trend effects
    - Adstock and saturation built into data
    - Cross-channel synergy effects
    - Realistic budget constraints and performance
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize campaign data generator
        
        Args:
            random_seed: Random seed for reproducible results
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Default channel configurations
        self.channel_configs = {
            'tv': {
                'base_spend': 45000,
                'effectiveness': 0.65,
                'saturation_point': 80000,
                'adstock_rate': 0.7,
                'seasonality_factor': 1.4,
                'cpm': 28,
                'reach_multiplier': 25
            },
            'digital': {
                'base_spend': 30000,
                'effectiveness': 0.85,
                'saturation_point': 50000,
                'adstock_rate': 0.3,
                'seasonality_factor': 1.1,
                'cpm': 2.5,
                'reach_multiplier': 2.5
            },
            'radio': {
                'base_spend': 18000,
                'effectiveness': 0.45,
                'saturation_point': 35000,
                'adstock_rate': 0.5,
                'seasonality_factor': 1.2,
                'cpm': 16,
                'reach_multiplier': 16
            },
            'print': {
                'base_spend': 10000,
                'effectiveness': 0.35,
                'saturation_point': 20000,
                'adstock_rate': 0.8,
                'seasonality_factor': 0.9,
                'cpm': 12,
                'reach_multiplier': 12
            },
            'social': {
                'base_spend': 15000,
                'effectiveness': 0.55,
                'saturation_point': 25000,
                'adstock_rate': 0.2,
                'seasonality_factor': 1.3,
                'cpm': 8,
                'reach_multiplier': 10
            },
            'ooh': {
                'base_spend': 12000,
                'effectiveness': 0.40,
                'saturation_point': 30000,
                'adstock_rate': 0.9,
                'seasonality_factor': 1.0,
                'cpm': 25,
                'reach_multiplier': 25
            }
        }
    
    def generate_campaign_data(self,
                             start_date: str = '2023-01-01',
                             end_date: str = '2024-06-30',
                             frequency: str = 'W',
                             channels: Optional[List[str]] = None,
                             base_revenue: float = 85000,
                             market_growth_rate: float = 0.02) -> pd.DataFrame:
        """
        Generate comprehensive campaign data
        
        Args:
            start_date: Campaign start date
            end_date: Campaign end date  
            frequency: Data frequency ('D', 'W', 'M')
            channels: List of channels to include (all if None)
            base_revenue: Base weekly revenue
            market_growth_rate: Annual market growth rate
            
        Returns:
            DataFrame with campaign data
        """
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        n_periods = len(dates)
        
        print(f"[SYNTHETIC] Generating {n_periods} periods of campaign data")
        print(f"[PERIOD] {start_date} to {end_date} ({frequency} frequency)")
        
        # Use all channels if none specified
        if channels is None:
            channels = list(self.channel_configs.keys())
        
        # Initialize dataframe
        campaign_data = pd.DataFrame({'date': dates})
        
        # Generate time-based effects
        seasonality_effects = self._generate_seasonality_effects(dates)
        trend_effects = self._generate_trend_effects(dates, market_growth_rate)
        
        # Generate spend data for each channel
        total_spend = 0
        channel_effects = {}
        
        for channel in channels:
            if channel in self.channel_configs:
                config = self.channel_configs[channel]
                
                # Generate spend with seasonality and noise
                spend_data = self._generate_channel_spend(
                    n_periods, 
                    config, 
                    seasonality_effects
                )
                
                campaign_data[f'{channel}_spend'] = spend_data
                total_spend += spend_data.sum()
                
                # Generate performance metrics
                impressions = self._generate_impressions(spend_data, config)
                campaign_data[f'{channel}_impressions'] = impressions
                
                if channel == 'digital':
                    campaign_data[f'{channel}_clicks'] = impressions
                elif channel in ['radio', 'tv']:
                    campaign_data[f'{channel}_reach'] = impressions
                elif channel == 'print':
                    campaign_data[f'{channel}_circulation'] = impressions
                elif channel == 'social':
                    campaign_data[f'{channel}_engagement'] = impressions
                elif channel == 'ooh':
                    campaign_data[f'{channel}_impressions'] = impressions
                
                # Calculate channel effects with adstock and saturation
                channel_effect = self._calculate_channel_effect(spend_data, config)
                channel_effects[channel] = channel_effect
        
        # Generate cross-channel synergy effects
        synergy_effects = self._generate_synergy_effects(channel_effects, channels)
        
        # Calculate total revenue
        revenue = self._generate_revenue(
            base_revenue,
            channel_effects,
            synergy_effects,
            seasonality_effects,
            trend_effects,
            n_periods
        )
        
        campaign_data['revenue'] = revenue
        campaign_data['conversions'] = (revenue / 42).astype(int)
        
        # Calculate summary statistics
        total_revenue = revenue.sum()
        roi = (total_revenue - total_spend) / total_spend if total_spend > 0 else 0
        
        print(f"[CHANNELS] Generated data for {len(channels)} channels: {channels}")
        print(f"[SPEND] Total media spend: ${total_spend:,.0f}")
        print(f"[REVENUE] Total revenue: ${total_revenue:,.0f}")
        print(f"[ROI] Campaign ROI: {roi:.1%}")
        
        return campaign_data
    
    def _generate_seasonality_effects(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate seasonality effects"""
        # Annual seasonality (Q4 peak)
        day_of_year = dates.dayofyear
        annual_effect = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
        
        # Weekly seasonality (weekend effect)
        day_of_week = dates.dayofweek
        weekly_effect = 1 + 0.1 * np.sin(2 * np.pi * day_of_week / 7)
        
        # Holiday effects (simplified)
        holiday_effect = np.ones(len(dates))
        for i, date in enumerate(dates):
            # Black Friday boost
            if date.month == 11 and date.day >= 23 and date.day <= 29:
                holiday_effect[i] = 1.5
            # Christmas season
            elif date.month == 12:
                holiday_effect[i] = 1.3
            # Summer dip
            elif date.month in [7, 8]:
                holiday_effect[i] = 0.9
        
        return annual_effect * weekly_effect * holiday_effect
    
    def _generate_trend_effects(self, dates: pd.DatetimeIndex, growth_rate: float) -> np.ndarray:
        """Generate trend effects"""
        # Linear growth trend
        days_from_start = (dates - dates[0]).days
        annual_growth = (1 + growth_rate) ** (days_from_start / 365.25)
        
        # Add some noise to make it realistic
        noise = np.random.normal(1, 0.02, len(dates))
        
        return annual_growth * noise
    
    def _generate_channel_spend(self, 
                              n_periods: int, 
                              config: Dict[str, Any], 
                              seasonality: np.ndarray) -> np.ndarray:
        """Generate realistic spend data for a channel"""
        base_spend = config['base_spend']
        seasonality_factor = config['seasonality_factor']
        
        # Apply seasonality
        seasonal_spend = base_spend * (1 + (seasonality - 1) * seasonality_factor)
        
        # Add random variation
        noise = np.random.gamma(2, 0.15, n_periods)  # Gamma for positive skew
        spend_variation = seasonal_spend * noise
        
        # Add budget constraints (occasional budget cuts/increases)
        budget_events = np.random.binomial(1, 0.05, n_periods)  # 5% chance of budget event
        budget_multipliers = np.where(
            budget_events,
            np.random.choice([0.7, 1.4], n_periods),  # 30% cut or 40% increase
            1.0
        )
        
        final_spend = spend_variation * budget_multipliers
        
        # Ensure no negative spend
        return np.maximum(final_spend, 0)
    
    def _generate_impressions(self, spend: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Generate impressions based on spend and CPM"""
        cpm = config['cpm']
        reach_multiplier = config['reach_multiplier']
        
        # Basic impressions calculation
        impressions = spend * reach_multiplier
        
        # Add efficiency variations (media buying effectiveness)
        efficiency_variation = np.random.normal(1, 0.1, len(spend))
        impressions *= efficiency_variation
        
        return np.maximum(impressions.astype(int), 0)
    
    def _calculate_channel_effect(self, spend: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Calculate channel effect with adstock and saturation"""
        effectiveness = config['effectiveness']
        saturation_point = config['saturation_point']
        adstock_rate = config['adstock_rate']
        
        # Apply saturation curve (diminishing returns)
        normalized_spend = spend / saturation_point
        saturated_effect = normalized_spend ** 0.6 / (normalized_spend ** 0.6 + 0.5 ** 0.6)
        saturated_spend = saturated_effect * saturation_point
        
        # Apply adstock (carryover effects)
        adstocked_effect = []
        carryover = 0
        
        for period_spend in saturated_spend:
            total_effect = period_spend + carryover * adstock_rate
            adstocked_effect.append(total_effect)
            carryover = total_effect
        
        # Scale by effectiveness
        return np.array(adstocked_effect) * effectiveness / 1000  # Scale for revenue impact
    
    def _generate_synergy_effects(self, 
                                channel_effects: Dict[str, np.ndarray], 
                                channels: List[str]) -> np.ndarray:
        """Generate cross-channel synergy effects"""
        n_periods = len(list(channel_effects.values())[0])
        synergy_effect = np.zeros(n_periods)
        
        # TV + Digital synergy
        if 'tv' in channel_effects and 'digital' in channel_effects:
            tv_digital_synergy = 0.05 * np.sqrt(
                channel_effects['tv'] * channel_effects['digital']
            )
            synergy_effect += tv_digital_synergy
        
        # Social + Digital synergy
        if 'social' in channel_effects and 'digital' in channel_effects:
            social_digital_synergy = 0.03 * np.sqrt(
                channel_effects['social'] * channel_effects['digital']
            )
            synergy_effect += social_digital_synergy
        
        # Print + Radio synergy (traditional media)
        if 'print' in channel_effects and 'radio' in channel_effects:
            traditional_synergy = 0.02 * np.sqrt(
                channel_effects['print'] * channel_effects['radio']
            )
            synergy_effect += traditional_synergy
        
        return synergy_effect
    
    def _generate_revenue(self,
                        base_revenue: float,
                        channel_effects: Dict[str, np.ndarray],
                        synergy_effects: np.ndarray,
                        seasonality_effects: np.ndarray,
                        trend_effects: np.ndarray,
                        n_periods: int) -> np.ndarray:
        """Generate total revenue combining all effects"""
        # Base revenue with trend and seasonality
        base_revenue_series = base_revenue * seasonality_effects * trend_effects
        
        # Sum all channel effects
        total_channel_effect = np.zeros(n_periods)
        for channel_effect in channel_effects.values():
            total_channel_effect += channel_effect
        
        # Add competitive and external factors
        external_noise = np.random.normal(0, 0.05, n_periods)  # 5% noise
        competitive_effect = np.random.normal(1, 0.02, n_periods)  # 2% competitive variation
        
        # Calculate final revenue
        total_revenue = (
            base_revenue_series + 
            total_channel_effect + 
            synergy_effects +
            base_revenue_series * external_noise
        ) * competitive_effect
        
        # Ensure no negative revenue
        return np.maximum(total_revenue, base_revenue * 0.5)
    
    def generate_test_scenarios(self,
                              baseline_data: pd.DataFrame,
                              scenario_configs: Dict[str, Dict[str, float]]) -> Dict[str, pd.DataFrame]:
        """
        Generate test scenarios based on baseline data
        
        Args:
            baseline_data: Baseline campaign data
            scenario_configs: Dict of scenario_name -> {channel: budget_multiplier}
            
        Returns:
            Dictionary of scenario DataFrames
        """
        scenarios = {}
        
        for scenario_name, config in scenario_configs.items():
            scenario_data = baseline_data.copy()
            
            # Apply scenario modifications
            for channel, multiplier in config.items():
                spend_column = f"{channel}_spend"
                if spend_column in scenario_data.columns:
                    scenario_data[spend_column] *= multiplier
                    
                    # Recalculate impressions
                    if channel in self.channel_configs:
                        channel_config = self.channel_configs[channel]
                        new_impressions = self._generate_impressions(
                            scenario_data[spend_column].values, 
                            channel_config
                        )
                        
                        if f"{channel}_impressions" in scenario_data.columns:
                            scenario_data[f"{channel}_impressions"] = new_impressions
                        elif f"{channel}_clicks" in scenario_data.columns:
                            scenario_data[f"{channel}_clicks"] = new_impressions
                        elif f"{channel}_reach" in scenario_data.columns:
                            scenario_data[f"{channel}_reach"] = new_impressions
            
            scenarios[scenario_name] = scenario_data
        
        return scenarios
    
    def get_channel_summary(self) -> pd.DataFrame:
        """Get summary of channel configurations"""
        summary_data = []
        
        for channel, config in self.channel_configs.items():
            summary_data.append({
                'channel': channel,
                'base_weekly_spend': config['base_spend'],
                'effectiveness': config['effectiveness'],
                'saturation_point': config['saturation_point'],
                'adstock_rate': config['adstock_rate'],
                'seasonality_factor': config['seasonality_factor'],
                'cpm': config['cpm']
            })
        
        return pd.DataFrame(summary_data)