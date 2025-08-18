"""
Data handling and multi-source integration.
Real data from Kaggle, HuggingFace, and synthetic generation.
"""

from .media_data_client import MediaDataClient
from .synthetic.campaign_data_generator import CampaignDataGenerator

__all__ = ["MediaDataClient", "CampaignDataGenerator"]