"""
FRED (Federal Reserve Economic Data) API client for fetching economic time series data.
"""

import pandas as pd
import numpy as np
from fredapi import Fred
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class FredDataClient:
    """Client for fetching economic data from FRED API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize FRED client with API key."""
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED API key required. Set FRED_API_KEY environment variable.")
        
        self.fred = Fred(api_key=self.api_key)
        
        # Economic indicators mapping
        self.indicators = {
            'gdp': 'GDPC1',  # Real GDP
            'unemployment': 'UNRATE',  # Unemployment Rate
            'inflation': 'CPIAUCSL',  # Consumer Price Index
            'interest_rate': 'DGS10',  # 10-Year Treasury Rate
            'consumer_confidence': 'UMCSENT',  # University of Michigan Consumer Sentiment
            'housing_starts': 'HOUST',  # Housing Starts
            'industrial_production': 'INDPRO',  # Industrial Production Index
            'retail_sales': 'RSAFS',  # Retail Sales
        }
    
    def fetch_indicator(self, indicator: str, start_date: str = '2000-01-01', 
                       end_date: Optional[str] = None) -> pd.Series:
        """
        Fetch a single economic indicator.
        
        Args:
            indicator: Indicator name (key from self.indicators) or FRED series ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
        
        Returns:
            pandas Series with time series data
        """
        # Use indicator mapping if available, otherwise use as series ID
        series_id = self.indicators.get(indicator, indicator)
        
        try:
            data = self.fred.get_series(
                series_id, 
                start=start_date, 
                end=end_date
            )
            data.name = indicator
            logger.info(f"Fetched {len(data)} observations for {indicator}")
            return data
        except Exception as e:
            logger.error(f"Error fetching {indicator}: {e}")
            raise
    
    def fetch_multiple_indicators(self, indicators: List[str], 
                                start_date: str = '2000-01-01',
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch multiple economic indicators and combine into DataFrame.
        
        Args:
            indicators: List of indicator names or FRED series IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with indicators as columns
        """
        data_dict = {}
        
        for indicator in indicators:
            try:
                series = self.fetch_indicator(indicator, start_date, end_date)
                data_dict[indicator] = series
            except Exception as e:
                logger.warning(f"Failed to fetch {indicator}: {e}")
                continue
        
        if not data_dict:
            raise ValueError("No indicators were successfully fetched")
        
        # Combine into DataFrame
        df = pd.DataFrame(data_dict)
        
        # Forward fill missing values for irregular series
        df = df.fillna(method='ffill')
        
        logger.info(f"Combined dataset shape: {df.shape}")
        return df
    
    def get_economic_dashboard_data(self, start_date: str = '2000-01-01') -> pd.DataFrame:
        """
        Fetch key economic indicators for dashboard analysis.
        
        Returns:
            DataFrame with major economic indicators
        """
        dashboard_indicators = [
            'gdp', 'unemployment', 'inflation', 'interest_rate', 
            'consumer_confidence', 'housing_starts'
        ]
        
        return self.fetch_multiple_indicators(dashboard_indicators, start_date)
    
    def search_series(self, search_text: str, limit: int = 10) -> pd.DataFrame:
        """
        Search for FRED series by text.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
        
        Returns:
            DataFrame with series information
        """
        try:
            results = self.fred.search(search_text, limit=limit)
            return results
        except Exception as e:
            logger.error(f"Error searching series: {e}")
            raise
    
    def get_series_info(self, series_id: str) -> Dict:
        """Get metadata information for a FRED series."""
        try:
            info = self.fred.get_series_info(series_id)
            return info.to_dict()
        except Exception as e:
            logger.error(f"Error getting series info: {e}")
            raise


def validate_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data quality and return metrics.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'total_observations': len(df),
        'date_range': (df.index.min(), df.index.max()),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'columns': df.columns.tolist(),
        'frequency': pd.infer_freq(df.index),
    }
    
    # Check for sufficient data points
    quality_report['sufficient_data'] = len(df) >= 50
    
    # Check for excessive missing values
    quality_report['excessive_missing'] = (df.isnull().sum() / len(df) > 0.2).any()
    
    return quality_report


if __name__ == "__main__":
    # Example usage
    client = FredDataClient()
    
    # Fetch dashboard data
    data = client.get_economic_dashboard_data(start_date='2010-01-01')
    print(f"Fetched data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Validate data quality
    quality = validate_data_quality(data)
    print(f"Data quality report: {quality}")