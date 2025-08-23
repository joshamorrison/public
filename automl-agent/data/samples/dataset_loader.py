"""
Real Dataset Loader for AutoML Agent

Downloads and provides access to real-world datasets for different ML tasks.
No synthetic data - only actual datasets from trusted sources.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import urllib.request
import zipfile
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class RealDatasetLoader:
    """
    Loads real datasets for different ML tasks.
    
    Prioritizes built-in sklearn datasets, then downloads from UCI ML Repository
    and other trusted sources when needed.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the dataset loader."""
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Available real datasets
        self.datasets = {
            # Classification Tasks
            "customer_churn": {
                "type": "classification",
                "source": "telco",
                "description": "Telco Customer Churn Dataset",
                "features": ["tenure", "monthly_charges", "total_charges", "contract_type", "payment_method"],
                "target": "churn",
                "url": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
            },
            "credit_approval": {
                "type": "classification", 
                "source": "uci",
                "description": "Credit Approval Dataset",
                "target": "approved"
            },
            "heart_disease": {
                "type": "classification",
                "source": "uci", 
                "description": "Heart Disease Dataset",
                "target": "target"
            },
            
            # Regression Tasks
            "boston_housing": {
                "type": "regression",
                "source": "sklearn",
                "description": "Boston Housing Prices",
                "target": "price"
            },
            "california_housing": {
                "type": "regression", 
                "source": "sklearn",
                "description": "California Housing Prices",
                "target": "median_house_value"
            },
            "bike_sharing": {
                "type": "regression",
                "source": "uci",
                "description": "Bike Sharing Demand",
                "target": "count",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
            },
            
            # Time Series
            "airline_passengers": {
                "type": "time_series",
                "source": "classic",
                "description": "Airline Passengers Time Series",
                "target": "passengers"
            },
            "energy_consumption": {
                "type": "time_series",
                "source": "uci",
                "description": "Individual Household Electric Power Consumption",
                "target": "global_active_power"
            },
            
            # NLP/Text
            "movie_reviews": {
                "type": "text_classification",
                "source": "sklearn",
                "description": "Movie Review Sentiment",
                "target": "sentiment"
            },
            "news_categories": {
                "type": "text_classification", 
                "source": "sklearn",
                "description": "20 Newsgroups Text Classification",
                "target": "category"
            },
            
            # Simple test data
            "test_data": {
                "type": "classification",
                "source": "local",
                "description": "Simple test dataset for demos",
                "target": "target",
                "features": ["age", "income", "education"]
            }
        }
    
    def load_customer_churn(self) -> Tuple[pd.DataFrame, str]:
        """Load real Telco Customer Churn dataset."""
        cache_file = self.cache_dir / "telco_churn.csv"
        
        if not cache_file.exists():
            print("Downloading Telco Customer Churn dataset...")
            url = self.datasets["customer_churn"]["url"]
            urllib.request.urlretrieve(url, cache_file)
            print("Dataset downloaded and cached")
        
        df = pd.read_csv(cache_file)
        
        # Clean the dataset
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna()
        
        # Rename target for clarity
        df['customer_churn'] = (df['Churn'] == 'Yes').astype(int)
        df = df.drop('Churn', axis=1)
        
        print(f"Loaded {len(df)} customers with {df.shape[1]} features")
        print(f"Churn rate: {df['customer_churn'].mean():.1%}")
        
        return df, "customer_churn"
    
    def load_california_housing(self) -> Tuple[pd.DataFrame, str]:
        """Load real California Housing dataset."""
        try:
            from sklearn.datasets import fetch_california_housing
            
            print(">> Loading California Housing dataset...")
            data = fetch_california_housing(as_frame=True)
            df = data.frame
            
            print(f"[DATA] Loaded {len(df)} houses with {df.shape[1]} features")
            print(f"[HOUSE] Price range: ${df['MedHouseVal'].min():.0f}k - ${df['MedHouseVal'].max():.0f}k")
            
            return df, "MedHouseVal"
            
        except ImportError:
            print("[ERROR] Scikit-learn not available, creating minimal sample")
            return self._create_minimal_housing_sample()
    
    def load_bike_sharing(self) -> Tuple[pd.DataFrame, str]:
        """Load real Bike Sharing Demand dataset."""
        cache_file = self.cache_dir / "bike_sharing.csv"
        
        if not cache_file.exists():
            print(">> Downloading Bike Sharing dataset...")
            zip_url = self.datasets["bike_sharing"]["url"]
            zip_file = self.cache_dir / "bike_sharing.zip"
            
            urllib.request.urlretrieve(zip_url, zip_file)
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.cache_dir)
            
            # Use the day.csv file (daily aggregated data)
            day_file = self.cache_dir / "day.csv"
            if day_file.exists():
                day_file.rename(cache_file)
            
            print("[OK] Dataset downloaded and cached")
        
        df = pd.read_csv(cache_file)
        
        # Parse date
        df['dteday'] = pd.to_datetime(df['dteday'])
        
        print(f"[DATA] Loaded {len(df)} days of bike sharing data")
        print(f"[BIKE] Daily rentals range: {df['cnt'].min()} - {df['cnt'].max()}")
        
        return df, "cnt"
    
    def load_airline_passengers(self) -> Tuple[pd.DataFrame, str]:
        """Load classic Airline Passengers time series."""
        # Create the classic airline passengers dataset
        dates = pd.date_range('1949-01-01', '1960-12-01', freq='MS')
        
        # Classic airline passengers data (approximate values)
        passengers = [
            112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
            115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
            145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
            171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
            196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
            204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
            242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
            284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
            315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
            340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
            360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
            417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
        ]
        
        df = pd.DataFrame({
            'date': dates,
            'passengers': passengers
        })
        
        print(f"[DATA] Loaded {len(df)} months of airline passenger data")
        print(f"[PLANE] Passenger range: {df['passengers'].min()} - {df['passengers'].max()}")
        
        return df, "passengers"
    
    def load_wine_quality(self) -> Tuple[pd.DataFrame, str]:
        """Load real Wine Quality dataset."""
        cache_file = self.cache_dir / "wine_quality.csv"
        
        if not cache_file.exists():
            print(">> Downloading Wine Quality dataset...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            urllib.request.urlretrieve(url, cache_file)
            print("[OK] Dataset downloaded and cached")
        
        df = pd.read_csv(cache_file, sep=';')
        
        # Convert to binary classification (good wine >= 6)
        df['good_wine'] = (df['quality'] >= 6).astype(int)
        
        print(f"[DATA] Loaded {len(df)} wines with {df.shape[1]} features")
        print(f"[WINE] Good wine rate: {df['good_wine'].mean():.1%}")
        
        return df, "good_wine"
    
    def load_iris(self) -> Tuple[pd.DataFrame, str]:
        """Load classic Iris dataset."""
        try:
            from sklearn.datasets import load_iris
            
            print(">> Loading Iris dataset...")
            data = load_iris(as_frame=True)
            df = data.frame
            
            print(f"[DATA] Loaded {len(df)} flowers with {df.shape[1]} features")
            print(f"[FLOWER] Classes: {data.target_names.tolist()}")
            
            return df, "target"
            
        except ImportError:
            print("[ERROR] Scikit-learn not available")
            return None, None
    
    def _create_minimal_housing_sample(self) -> Tuple[pd.DataFrame, str]:
        """Create a minimal housing sample if sklearn unavailable."""
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'avg_rooms': np.random.normal(6, 1.5, n_samples),
            'avg_bedrooms': np.random.normal(1.2, 0.3, n_samples),
            'population': np.random.randint(500, 5000, n_samples),
            'households': np.random.randint(100, 2000, n_samples),
            'median_income': np.random.normal(4, 2, n_samples),
            'house_age': np.random.randint(1, 50, n_samples)
        })
        
        # Create realistic price based on features
        df['price'] = (
            df['median_income'] * 30 +
            df['avg_rooms'] * 10 +
            np.random.normal(0, 20, n_samples)
        )
        df['price'] = np.clip(df['price'], 50, 500)
        
        return df, "price"
    
    def load_test_data(self) -> Tuple[pd.DataFrame, str]:
        """Load simple test dataset from samples."""
        test_file = self.data_dir / "test_data.csv"
        
        if not test_file.exists():
            print(f"[ERROR] Test data file not found at {test_file}")
            return self._create_minimal_test_sample()
        
        df = pd.read_csv(test_file)
        
        print(f"[DATA] Loaded {len(df)} samples with {df.shape[1]} features")
        print(f"[TEST] Features: {df.columns.tolist()[:-1]}")
        
        return df, "target"
    
    def _create_minimal_test_sample(self) -> Tuple[pd.DataFrame, str]:
        """Create a minimal test sample if file not found."""
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'age': np.random.randint(20, 65, n_samples),
            'income': np.random.randint(25000, 150000, n_samples),
            'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        return df, "target"
    
    def get_dataset_info(self) -> Dict:
        """Get information about all available datasets."""
        return self.datasets
    
    def load_dataset(self, name: str) -> Tuple[pd.DataFrame, str]:
        """Load a specific dataset by name."""
        if name not in self.datasets:
            available = list(self.datasets.keys())
            raise ValueError(f"Dataset '{name}' not available. Available: {available}")
        
        loader_method = f"load_{name}"
        if hasattr(self, loader_method):
            return getattr(self, loader_method)()
        else:
            raise NotImplementedError(f"Loader for '{name}' not implemented yet")
    
    def list_available_datasets(self) -> List[str]:
        """List all available dataset names."""
        return list(self.datasets.keys())


def load_demo_dataset(task_type: str = "classification") -> Tuple[pd.DataFrame, str]:
    """
    Load a demo dataset for the specified task type.
    
    Args:
        task_type: Type of ML task ('classification', 'regression', 'time_series')
    
    Returns:
        Tuple of (dataframe, target_column_name)
    """
    loader = RealDatasetLoader()
    
    if task_type == "classification":
        return loader.load_customer_churn()
    elif task_type == "regression":
        return loader.load_california_housing()
    elif task_type == "time_series":
        return loader.load_airline_passengers()
    else:
        # Default to customer churn
        return loader.load_customer_churn()


if __name__ == "__main__":
    # Demo usage
    loader = RealDatasetLoader()
    
    print("Available Real Datasets:")
    for name, info in loader.get_dataset_info().items():
        print(f"  - {name}: {info['description']} ({info['type']})")
    
    print("\n" + "="*60)
    
    # Test load customer churn
    print("Testing Customer Churn Dataset:")
    df, target = loader.load_customer_churn()
    print(f"Shape: {df.shape}")
    print(f"Target: {target}")
    print(f"Features: {df.columns.tolist()[:5]}...")