#!/usr/bin/env python3
"""
Media Data Client - Multi-source data integration
Supports Kaggle, HuggingFace, and synthetic data with intelligent fallbacks
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class MediaDataClient:
    """
    Multi-source media data client with progressive enhancement
    1st: Kaggle Marketing Analytics (enterprise quality)
    2nd: HuggingFace Advertising (professional quality)  
    3rd: Synthetic data (always works)
    """
    
    def __init__(self, 
                 kaggle_username: Optional[str] = None,
                 kaggle_key: Optional[str] = None,
                 hf_token: Optional[str] = None,
                 cache_dir: str = "./cache/data"):
        """
        Initialize multi-source data client
        
        Args:
            kaggle_username: Kaggle API username
            kaggle_key: Kaggle API key
            hf_token: HuggingFace token
            cache_dir: Directory for caching downloaded data
        """
        self.kaggle_username = kaggle_username or os.getenv('KAGGLE_USERNAME')
        self.kaggle_key = kaggle_key or os.getenv('KAGGLE_KEY')
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Track data source capabilities
        self.data_sources = {
            'kaggle': self._check_kaggle_availability(),
            'huggingface': self._check_huggingface_availability(),
            'synthetic': True  # Always available
        }
    
    def _check_kaggle_availability(self) -> bool:
        """Check if Kaggle API is available and configured"""
        try:
            import kaggle
            if self.kaggle_username and self.kaggle_key:
                return True
            return False
        except ImportError:
            return False
        except Exception:
            return False
    
    def _check_huggingface_availability(self) -> bool:
        """Check if HuggingFace datasets is available"""
        try:
            import datasets
            return True
        except ImportError:
            return False
    
    def get_best_available_data(self) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
        """
        Get the best available marketing dataset with intelligent fallbacks
        
        Returns:
            Tuple of (data, source_info, source_type)
        """
        print("[DATA] Checking available data sources...")
        
        # Try Kaggle first (highest quality)
        if self.data_sources['kaggle']:
            try:
                data, source_info = self.load_kaggle_marketing_data()
                if data is not None and len(data) > 0:
                    print(f"[SUCCESS] Using Kaggle: {source_info['description']}")
                    return data, source_info, 'KAGGLE'
            except Exception as e:
                print(f"[ERROR] Kaggle failed: {str(e)[:50]}")
        
        # Try HuggingFace second (good quality)
        if self.data_sources['huggingface']:
            try:
                data, source_info = self.load_huggingface_advertising_data()
                if data is not None and len(data) > 0:
                    print(f"[SUCCESS] Using HuggingFace: {source_info['description']}")
                    return data, source_info, 'HUGGINGFACE'
            except Exception as e:
                print(f"[ERROR] HuggingFace failed: {str(e)[:50]}")
        
        # Fallback to synthetic (always works)
        print("[FALLBACK] Using synthetic marketing data")
        data, source_info = self.create_synthetic_marketing_data()
        print(f"[SUCCESS] Generated: {source_info['description']}")
        return data, source_info, 'SYNTHETIC'
    
    def create_synthetic_marketing_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create realistic synthetic marketing data
        
        Returns:
            Tuple of (dataframe, source_info)
        """
        print("[SYNTHETIC] Generating realistic marketing mix data...")
        
        # Create 18 months of weekly data
        dates = pd.date_range('2023-01-01', '2024-06-30', freq='W')
        np.random.seed(42)  # For reproducible results
        
        # Simulate realistic marketing mix with seasonality
        n_weeks = len(dates)
        
        # Add seasonality (higher spend in Q4, lower in Q1)
        seasonality = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_weeks) / 52 - np.pi/2)
        
        # Channel spend with realistic budgets and correlation
        tv_base = 45000 * seasonality
        digital_base = 30000 * seasonality  
        radio_base = 18000 * seasonality
        print_base = 10000 * seasonality
        social_base = 15000 * seasonality
        
        marketing_data = pd.DataFrame({
            'date': dates,
            'tv_spend': tv_base + np.random.normal(0, 6000, n_weeks),
            'digital_spend': digital_base + np.random.normal(0, 4000, n_weeks),
            'radio_spend': radio_base + np.random.normal(0, 2500, n_weeks),
            'print_spend': print_base + np.random.normal(0, 1500, n_weeks),
            'social_spend': social_base + np.random.normal(0, 2000, n_weeks)
        })
        
        # Ensure no negative spend
        spend_cols = ['tv_spend', 'digital_spend', 'radio_spend', 'print_spend', 'social_spend']
        marketing_data[spend_cols] = marketing_data[spend_cols].clip(lower=0)
        
        # Generate realistic performance metrics with diminishing returns
        marketing_data['tv_impressions'] = (marketing_data['tv_spend'] * 28).astype(int)
        marketing_data['digital_clicks'] = (marketing_data['digital_spend'] * 2.5).astype(int)
        marketing_data['radio_reach'] = (marketing_data['radio_spend'] * 16).astype(int)
        marketing_data['print_circulation'] = (marketing_data['print_spend'] * 12).astype(int)
        marketing_data['social_engagement'] = (marketing_data['social_spend'] * 10).astype(int)
        
        # Revenue with media mix effects and diminishing returns (square root for saturation)
        base_revenue = 85000
        tv_effect = 0.65 * np.sqrt(marketing_data['tv_spend'] / 1000)
        digital_effect = 0.85 * np.sqrt(marketing_data['digital_spend'] / 1000)
        radio_effect = 0.45 * np.sqrt(marketing_data['radio_spend'] / 1000)
        print_effect = 0.35 * np.sqrt(marketing_data['print_spend'] / 1000)
        social_effect = 0.55 * np.sqrt(marketing_data['social_spend'] / 1000)
        
        # Add cross-channel synergy effects
        cross_channel_effect = 0.1 * np.sqrt(
            (marketing_data['tv_spend'] + marketing_data['digital_spend']) / 2000
        )
        
        marketing_data['revenue'] = (
            base_revenue + tv_effect + digital_effect + radio_effect + 
            print_effect + social_effect + cross_channel_effect +
            np.random.normal(0, 6000, n_weeks)
        ).clip(lower=0)
        
        marketing_data['conversions'] = (marketing_data['revenue'] / 42).astype(int)
        
        # Calculate summary statistics
        total_spend = marketing_data[spend_cols].sum().sum()
        total_revenue = marketing_data['revenue'].sum()
        roi = (total_revenue - total_spend) / total_spend
        
        print(f"[STATS] Generated {n_weeks} weeks of data")
        print(f"[SPEND] Total media spend: ${total_spend:,.0f}")
        print(f"[REVENUE] Total revenue: ${total_revenue:,.0f}")
        print(f"[ROI] Baseline ROI: {roi:.1%}")
        
        source_info = {
            'description': f'Synthetic Marketing Data ({n_weeks} weeks, ${total_spend/1000000:.1f}M spend)',
            'channels': ['TV', 'Digital', 'Radio', 'Print', 'Social'],
            'metrics': ['Spend', 'Impressions/Reach', 'Conversions', 'Revenue'],
            'quality': 'DEMO',
            'size': len(marketing_data),
            'total_spend': total_spend,
            'total_revenue': total_revenue,
            'baseline_roi': roi,
            'source': 'synthetic'
        }
        
        return marketing_data, source_info