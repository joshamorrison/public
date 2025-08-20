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
    
    def load_kaggle_marketing_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load real marketing data from Kaggle datasets
        
        Returns:
            Tuple of (dataframe, source_info)
        """
        try:
            import kaggle
            
            # Try to download popular marketing datasets from Kaggle
            datasets_to_try = [
                ('carrie1/ecommerce-data', 'E-commerce customer data with marketing channels'),
                ('olistbr/brazilian-ecommerce', 'Brazilian e-commerce marketing data'),
                ('mkechinov/ecommerce-events-history-in-cosmetics-shop', 'Cosmetics shop marketing events'),
                ('retailrocket/ecommerce-dataset', 'E-commerce behavior with marketing attribution')
            ]
            
            for dataset_id, description in datasets_to_try:
                try:
                    print(f"[KAGGLE] Attempting to download: {dataset_id}")
                    
                    # Download to cache directory
                    kaggle.api.dataset_download_files(
                        dataset_id, 
                        path=self.cache_dir, 
                        unzip=True, 
                        quiet=False
                    )
                    
                    # Find CSV files in the downloaded data
                    import glob
                    csv_files = glob.glob(f"{self.cache_dir}/*.csv")
                    
                    if csv_files:
                        # Load the first CSV file
                        df = pd.read_csv(csv_files[0])
                        
                        # Basic validation - ensure it has date-like and numeric columns
                        if len(df) > 100 and any('date' in col.lower() or 'time' in col.lower() for col in df.columns):
                            print(f"[SUCCESS] Loaded Kaggle dataset: {len(df)} rows from {csv_files[0]}")
                            
                            source_info = {
                                'description': f'Kaggle: {description}',
                                'dataset_id': dataset_id,
                                'file_path': csv_files[0],
                                'size': len(df),
                                'quality': 'REAL',
                                'source': 'kaggle'
                            }
                            
                            return df, source_info
                            
                except Exception as e:
                    print(f"[ERROR] Failed to load {dataset_id}: {str(e)[:50]}")
                    continue
            
            # If no datasets worked, return None
            return None, {}
            
        except Exception as e:
            print(f"[ERROR] Kaggle setup failed: {str(e)}")
            return None, {}
    
    def load_huggingface_advertising_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load real advertising data from HuggingFace datasets
        
        Returns:
            Tuple of (dataframe, source_info)
        """
        try:
            from datasets import load_dataset
            
            # Try real advertising/marketing datasets from HuggingFace
            datasets_to_try = [
                ('RafaM97/marketing_social_media', 'Social media marketing campaigns with industry and channel data'),
                ('dianalogan/Marketing-Budget-and-Actual-Sales-Dataset', 'Marketing budget and actual sales correlation data'),
                ('PeterBrendan/Ads_Creative_Text_Programmatic', 'Programmatic advertising creative text (1000 samples)'),
                ('dvilasuero/marketing', 'Marketing dataset generated using Magpie alignment')
            ]
            
            for dataset_id, description in datasets_to_try:
                try:
                    print(f"[HUGGINGFACE] Attempting to load: {dataset_id}")
                    
                    # Load dataset
                    dataset = load_dataset(dataset_id, split='train')
                    df = dataset.to_pandas()
                    
                    # Basic validation - check for minimum data size
                    if len(df) > 50:
                        print(f"[SUCCESS] Loaded HuggingFace dataset: {len(df)} rows")
                        
                        source_info = {
                            'description': f'HuggingFace: {description}',
                            'dataset_id': dataset_id,
                            'size': len(df),
                            'columns': list(df.columns),
                            'quality': 'REAL',
                            'source': 'huggingface'
                        }
                        
                        return df, source_info
                        
                except Exception as e:
                    print(f"[ERROR] Failed to load {dataset_id}: {str(e)[:50]}")
                    continue
            
            # If no datasets worked, return None
            return None, {}
            
        except Exception as e:
            print(f"[ERROR] HuggingFace setup failed: {str(e)}")
            return None, {}
    
    def fetch_campaign_performance(self, start_date, end_date, channels, campaigns=None, granularity='daily'):
        """Fetch campaign performance data for API endpoints"""
        try:
            # Get the best available data
            data, source_info, source_type = self.get_best_available_data()
            
            # Filter by date range if date column exists
            date_cols = [col for col in data.columns if 'date' in col.lower()]
            if date_cols:
                data[date_cols[0]] = pd.to_datetime(data[date_cols[0]])
                mask = (data[date_cols[0]] >= start_date) & (data[date_cols[0]] <= end_date)
                data = data[mask]
            
            return data.to_dict('records')
            
        except Exception as e:
            raise Exception(f"Failed to fetch campaign performance: {str(e)}")
    
    def fetch_journey_data(self, start_date, end_date, channels):
        """Fetch customer journey data for attribution analysis"""
        try:
            # For now, return synthetic journey data
            # In production, this would connect to customer journey tracking systems
            import random
            
            journeys = []
            for i in range(100):
                customer_id = f"customer_{i:04d}"
                touchpoints = random.randint(1, 5)
                
                for j in range(touchpoints):
                    journeys.append({
                        'customer_id': customer_id,
                        'touchpoint_date': start_date + timedelta(days=random.randint(0, (end_date - start_date).days)),
                        'channel': random.choice(channels),
                        'touchpoint_order': j + 1,
                        'campaign_id': f"campaign_{random.randint(1, 10):02d}",
                        'converted': 1 if j == touchpoints - 1 and random.random() < 0.15 else 0
                    })
            
            return journeys
            
        except Exception as e:
            raise Exception(f"Failed to fetch journey data: {str(e)}")
    
    def fetch_performance_data(self, channels, days):
        """Fetch historical performance data for optimization"""
        try:
            data, source_info, source_type = self.get_best_available_data()
            return data.to_dict('records')
        except Exception as e:
            raise Exception(f"Failed to fetch performance data: {str(e)}")
    
    def fetch_saturation_data(self, channels, days):
        """Fetch data for saturation analysis"""
        try:
            data, source_info, source_type = self.get_best_available_data()
            return data.to_dict('records')
        except Exception as e:
            raise Exception(f"Failed to fetch saturation data: {str(e)}")