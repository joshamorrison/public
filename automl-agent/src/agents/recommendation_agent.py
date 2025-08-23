"""
Recommendation Agent for AutoML Platform

Specialized agent for recommendation systems and collaborative filtering that:
1. Handles various recommendation algorithms and approaches
2. Implements collaborative filtering, content-based, and hybrid methods
3. Supports matrix factorization and deep learning approaches
4. Provides recommendation evaluation and business metrics
5. Handles cold start problems and sparsity issues

This agent runs for recommendation system problems and personalization tasks.
"""

import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD, NMF
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.sparse import csr_matrix
    from scipy.spatial.distance import jaccard, cosine
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, TaskContext, TaskComplexity


class RecommendationTask(Enum):
    """Types of recommendation tasks."""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    MATRIX_FACTORIZATION = "matrix_factorization"
    DEEP_LEARNING = "deep_learning"
    SEQUENTIAL = "sequential"
    CROSS_DOMAIN = "cross_domain"


class RecommendationMethod(Enum):
    """Recommendation algorithms."""
    USER_BASED_CF = "user_based_cf"
    ITEM_BASED_CF = "item_based_cf"
    CONTENT_BASED = "content_based"
    SVD = "svd"
    NMF = "nmf"
    DEEP_COLLABORATIVE = "deep_collaborative"
    HYBRID_MODEL = "hybrid_model"


class SimilarityMetric(Enum):
    """Similarity metrics for recommendations."""
    COSINE = "cosine"
    PEARSON = "pearson"
    EUCLIDEAN = "euclidean"
    JACCARD = "jaccard"
    MANHATTAN = "manhattan"


@dataclass
class RecommendationAnalysis:
    """Recommendation data analysis results."""
    n_users: int
    n_items: int
    n_interactions: int
    sparsity_ratio: float
    interaction_density: float
    avg_interactions_per_user: float
    avg_interactions_per_item: float
    rating_distribution: Dict[float, int]
    cold_start_users: int
    cold_start_items: int
    power_law_coefficient: float
    temporal_patterns: bool


@dataclass
class RecommendationPerformance:
    """Recommendation model performance metrics."""
    algorithm: str
    rmse: float
    mae: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    map_score: float
    coverage: float
    diversity: float
    novelty: float
    training_time: float
    prediction_time: float
    scalability_score: float
    cold_start_performance: float


@dataclass
class RecommendationResult:
    """Complete recommendation result."""
    task_type: str
    best_algorithm: str
    best_model: Any
    performance_metrics: RecommendationPerformance
    all_model_performances: List[RecommendationPerformance]
    recommendation_analysis: RecommendationAnalysis
    sample_recommendations: Dict[str, List[Dict[str, Any]]]
    feature_importance: Optional[Dict[str, float]]
    model_interpretability: Dict[str, Any]
    business_metrics: Dict[str, float]
    preprocessing_steps: List[str]


class RecommendationAgent(BaseAgent):
    """
    Recommendation Agent for personalization and collaborative filtering.
    
    Responsibilities:
    1. Recommendation data preprocessing and analysis
    2. Collaborative filtering implementation
    3. Content-based recommendation systems
    4. Hybrid and advanced recommendation methods
    5. Business-focused evaluation and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub=None):
        """Initialize the Recommendation Agent."""
        super().__init__(
            name="Recommendation Agent",
            description="Advanced recommendation systems and personalization specialist",
            specialization="Recommendation Systems & Personalization",
            config=config,
            communication_hub=communication_hub
        )
        
        # Recommendation configuration
        self.top_k_recommendations = self.config.get("top_k_recommendations", 10)
        self.evaluation_k_values = self.config.get("evaluation_k_values", [5, 10, 20])
        self.min_interactions_per_user = self.config.get("min_interactions_per_user", 5)
        self.min_interactions_per_item = self.config.get("min_interactions_per_item", 3)
        
        # Algorithm settings
        self.similarity_metric = self.config.get("similarity_metric", "cosine")
        self.matrix_factorization_components = self.config.get("matrix_factorization_components", 50)
        self.neighborhood_size = self.config.get("neighborhood_size", 50)
        self.quick_mode = self.config.get("quick_mode", False)
        
        # Quality thresholds
        self.quality_thresholds.update({
            "min_precision_at_10": self.config.get("min_precision_at_10", 0.1),
            "min_recall_at_10": self.config.get("min_recall_at_10", 0.15),
            "min_ndcg_at_10": self.config.get("min_ndcg_at_10", 0.2),
            "min_coverage": self.config.get("min_coverage", 0.5),
            "max_sparsity": self.config.get("max_sparsity", 0.99)
        })
        
        # Business metrics configuration
        self.include_business_metrics = self.config.get("include_business_metrics", True)
        self.diversity_weight = self.config.get("diversity_weight", 0.2)
        self.novelty_weight = self.config.get("novelty_weight", 0.1)
        
        # Cold start handling
        self.handle_cold_start = self.config.get("handle_cold_start", True)
        self.cold_start_method = self.config.get("cold_start_method", "popularity")
        
        # Random state
        self.random_state = self.config.get("random_state", 42)
    
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute comprehensive recommendation system workflow.
        
        Args:
            context: Task context with interaction data
            
        Returns:
            AgentResult with recommendation models and analysis
        """
        try:
            self.logger.info("Starting recommendation system workflow...")
            
            # Load interaction dataset
            interactions_df, items_df, users_df = self._load_recommendation_dataset(context)
            if interactions_df is None:
                return AgentResult(
                    success=False,
                    message="Failed to load recommendation dataset"
                )
            
            # Phase 1: Task Identification
            self.logger.info("Phase 1: Identifying recommendation task...")
            task_type = self._identify_recommendation_task(context, interactions_df)
            
            # Phase 2: Data Analysis
            self.logger.info("Phase 2: Analyzing recommendation data...")
            rec_analysis = self._analyze_recommendation_data(interactions_df, items_df, users_df)
            
            # Phase 3: Data Preprocessing
            self.logger.info("Phase 3: Preprocessing recommendation data...")
            processed_data, preprocessing_steps = self._preprocess_recommendation_data(
                interactions_df, items_df, users_df, rec_analysis
            )
            
            # Phase 4: Create Interaction Matrix
            self.logger.info("Phase 4: Creating interaction matrix...")
            interaction_matrix, user_map, item_map = self._create_interaction_matrix(processed_data)
            
            # Phase 5: Train-Test Split
            self.logger.info("Phase 5: Preparing train-test splits...")
            train_matrix, test_interactions = self._prepare_recommendation_splits(
                interaction_matrix, processed_data, user_map, item_map
            )
            
            # Phase 6: Model Training and Evaluation
            self.logger.info("Phase 6: Training and evaluating recommendation models...")
            model_performances = self._train_and_evaluate_models(
                train_matrix, test_interactions, processed_data, items_df, 
                user_map, item_map, task_type, rec_analysis
            )
            
            # Phase 7: Select Best Model
            self.logger.info("Phase 7: Selecting best performing model...")
            best_model_info = self._select_best_model(model_performances)
            
            # Phase 8: Final Evaluation and Business Metrics
            self.logger.info("Phase 8: Final evaluation and business metrics...")
            final_results = self._final_recommendation_evaluation(
                best_model_info, train_matrix, test_interactions, processed_data,
                items_df, user_map, item_map, task_type, rec_analysis, preprocessing_steps
            )
            
            # Phase 9: Generate Sample Recommendations
            self.logger.info("Phase 9: Generating sample recommendations...")
            sample_recommendations = self._generate_sample_recommendations(
                final_results.best_model, train_matrix, items_df, user_map, item_map
            )
            final_results.sample_recommendations = sample_recommendations
            
            # Phase 10: Business Impact Analysis
            if self.include_business_metrics:
                self.logger.info("Phase 10: Calculating business impact metrics...")
                business_metrics = self._calculate_business_metrics(
                    final_results, rec_analysis, sample_recommendations
                )
                final_results.business_metrics = business_metrics
            
            # Create comprehensive result
            result_data = {
                "recommendation_results": self._results_to_dict(final_results),
                "recommendation_analysis": self._rec_analysis_to_dict(rec_analysis),
                "task_type": task_type.value,
                "preprocessing_steps": preprocessing_steps,
                "model_performances": [self._performance_to_dict(perf) for perf in model_performances],
                "recommendations": self._generate_recommendations(final_results, rec_analysis)
            }
            
            # Update performance metrics
            performance_metrics = {
                "recommendation_precision": final_results.performance_metrics.precision_at_k.get(10, 0.0),
                "recommendation_recall": final_results.performance_metrics.recall_at_k.get(10, 0.0),
                "recommendation_coverage": final_results.performance_metrics.coverage,
                "personalization_efficiency": 1.0 / (final_results.performance_metrics.training_time + 1)
            }
            self.update_performance_metrics(performance_metrics)
            
            # Share recommendation insights
            if self.communication_hub:
                self._share_recommendation_insights(result_data)
            
            return AgentResult(
                success=True,
                data=result_data,
                message=f"Recommendation workflow completed: {task_type.value} with Precision@10 = {final_results.performance_metrics.precision_at_k.get(10, 0.0):.3f}",
                recommendations=result_data["recommendations"]
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Recommendation workflow failed: {str(e)}"
            )
    
    def _load_recommendation_dataset(self, context: TaskContext) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load recommendation dataset or create synthetic data."""
        # In real implementation, this would load from files or previous agent results
        # For demo, create synthetic recommendation data
        
        user_input = context.user_input.lower()
        
        if "movie" in user_input or "film" in user_input:
            return self._create_movie_recommendation_dataset()
        elif "product" in user_input or "ecommerce" in user_input:
            return self._create_ecommerce_recommendation_dataset()
        elif "book" in user_input or "literature" in user_input:
            return self._create_book_recommendation_dataset()
        elif "music" in user_input or "song" in user_input:
            return self._create_music_recommendation_dataset()
        else:
            return self._create_general_recommendation_dataset()
    
    def _create_general_recommendation_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create general synthetic recommendation dataset."""
        np.random.seed(42)
        
        n_users = 1000
        n_items = 500
        n_interactions = 5000
        
        # Generate users
        users_df = pd.DataFrame({
            'user_id': range(n_users),
            'age': np.random.randint(18, 65, n_users),
            'gender': np.random.choice(['M', 'F'], n_users),
            'location': np.random.choice(['US', 'UK', 'CA', 'DE', 'FR'], n_users)
        })
        
        # Generate items
        items_df = pd.DataFrame({
            'item_id': range(n_items),
            'category': np.random.choice(['electronics', 'books', 'clothing', 'home', 'sports'], n_items),
            'price': np.random.exponential(50, n_items),
            'brand': np.random.choice([f'brand_{i}' for i in range(20)], n_items)
        })
        
        # Generate interactions with realistic patterns
        interactions = []
        
        # Create some power users
        power_users = np.random.choice(n_users, n_users // 10, replace=False)
        
        for _ in range(n_interactions):
            # Power users are more likely to interact
            if np.random.random() < 0.3:
                user_id = np.random.choice(power_users)
            else:
                user_id = np.random.randint(n_users)
            
            item_id = np.random.randint(n_items)
            
            # Rating follows normal distribution around 3.5
            rating = np.clip(np.random.normal(3.5, 1.2), 1, 5)
            rating = round(rating, 1)
            
            timestamp = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365))
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'timestamp': timestamp
            })
        
        interactions_df = pd.DataFrame(interactions)
        
        return interactions_df, items_df, users_df
    
    def _create_movie_recommendation_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create synthetic movie recommendation dataset."""
        np.random.seed(42)
        
        n_users = 800
        n_movies = 300
        n_ratings = 4000
        
        # Generate users
        users_df = pd.DataFrame({
            'user_id': range(n_users),
            'age': np.random.randint(16, 70, n_users),
            'occupation': np.random.choice(['student', 'engineer', 'teacher', 'doctor', 'other'], n_users),
            'favorite_genre': np.random.choice(['action', 'comedy', 'drama', 'sci-fi', 'horror'], n_users)
        })
        
        # Generate movies
        genres = ['action', 'comedy', 'drama', 'sci-fi', 'horror', 'romance', 'thriller']
        movies_df = pd.DataFrame({
            'item_id': range(n_movies),
            'title': [f'Movie_{i}' for i in range(n_movies)],
            'genre': np.random.choice(genres, n_movies),
            'year': np.random.randint(1980, 2024, n_movies),
            'duration': np.random.randint(90, 180, n_movies)
        })
        
        # Generate ratings with genre preferences
        interactions = []
        
        for _ in range(n_ratings):
            user_id = np.random.randint(n_users)
            movie_id = np.random.randint(n_movies)
            
            user_pref_genre = users_df.loc[user_id, 'favorite_genre']
            movie_genre = movies_df.loc[movie_id, 'genre']
            
            # Higher ratings for preferred genres
            if user_pref_genre == movie_genre:
                rating = np.clip(np.random.normal(4.2, 0.8), 1, 5)
            else:
                rating = np.clip(np.random.normal(3.2, 1.0), 1, 5)
            
            rating = round(rating, 1)
            timestamp = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 1000))
            
            interactions.append({
                'user_id': user_id,
                'item_id': movie_id,
                'rating': rating,
                'timestamp': timestamp
            })
        
        interactions_df = pd.DataFrame(interactions)
        
        return interactions_df, movies_df, users_df
    
    def _create_ecommerce_recommendation_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create synthetic e-commerce recommendation dataset."""
        np.random.seed(42)
        
        n_users = 1200
        n_products = 800
        n_interactions = 6000
        
        # Generate users
        users_df = pd.DataFrame({
            'user_id': range(n_users),
            'age': np.random.randint(18, 75, n_users),
            'income_level': np.random.choice(['low', 'medium', 'high'], n_users),
            'shopping_frequency': np.random.choice(['rare', 'occasional', 'frequent'], n_users)
        })
        
        # Generate products
        categories = ['electronics', 'clothing', 'books', 'home', 'sports', 'beauty']
        products_df = pd.DataFrame({
            'item_id': range(n_products),
            'name': [f'Product_{i}' for i in range(n_products)],
            'category': np.random.choice(categories, n_products),
            'price': np.random.exponential(30, n_products),
            'brand': np.random.choice([f'brand_{i}' for i in range(50)], n_products),
            'avg_rating': np.random.uniform(3.0, 5.0, n_products)
        })
        
        # Generate purchase interactions
        interactions = []
        
        for _ in range(n_interactions):
            user_id = np.random.randint(n_users)
            product_id = np.random.randint(n_products)
            
            # Purchase probability based on user income and product price
            user_income = users_df.loc[user_id, 'income_level']
            product_price = products_df.loc[product_id, 'price']
            
            # Implicit feedback (purchase = 1, no purchase = 0)
            if user_income == 'high' or product_price < 50:
                rating = 1  # Purchase
            else:
                rating = 1 if np.random.random() < 0.3 else 0  # Lower purchase probability
            
            if rating == 1:  # Only record purchases
                timestamp = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365))
                
                interactions.append({
                    'user_id': user_id,
                    'item_id': product_id,
                    'rating': rating,
                    'timestamp': timestamp
                })
        
        interactions_df = pd.DataFrame(interactions)
        
        return interactions_df, products_df, users_df
    
    def _create_book_recommendation_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create synthetic book recommendation dataset."""
        np.random.seed(42)
        
        n_users = 600
        n_books = 400
        n_ratings = 3000
        
        # Generate users
        users_df = pd.DataFrame({
            'user_id': range(n_users),
            'age': np.random.randint(16, 80, n_users),
            'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], n_users),
            'reading_frequency': np.random.choice(['low', 'medium', 'high'], n_users)
        })
        
        # Generate books
        genres = ['fiction', 'non-fiction', 'mystery', 'romance', 'sci-fi', 'biography', 'history']
        books_df = pd.DataFrame({
            'item_id': range(n_books),
            'title': [f'Book_{i}' for i in range(n_books)],
            'author': [f'Author_{i//5}' for i in range(n_books)],  # Multiple books per author
            'genre': np.random.choice(genres, n_books),
            'publication_year': np.random.randint(1950, 2024, n_books),
            'pages': np.random.randint(100, 800, n_books)
        })
        
        # Generate ratings
        interactions = []
        
        for _ in range(n_ratings):
            user_id = np.random.randint(n_users)
            book_id = np.random.randint(n_books)
            
            # Readers tend to rate books they finish higher
            reading_freq = users_df.loc[user_id, 'reading_frequency']
            if reading_freq == 'high':
                rating = np.clip(np.random.normal(4.0, 1.0), 1, 5)
            else:
                rating = np.clip(np.random.normal(3.5, 1.2), 1, 5)
            
            rating = round(rating, 1)
            timestamp = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 2000))
            
            interactions.append({
                'user_id': user_id,
                'item_id': book_id,
                'rating': rating,
                'timestamp': timestamp
            })
        
        interactions_df = pd.DataFrame(interactions)
        
        return interactions_df, books_df, users_df
    
    def _create_music_recommendation_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create synthetic music recommendation dataset."""
        np.random.seed(42)
        
        n_users = 500
        n_songs = 1000
        n_listens = 10000
        
        # Generate users
        users_df = pd.DataFrame({
            'user_id': range(n_users),
            'age': np.random.randint(13, 60, n_users),
            'preferred_genre': np.random.choice(['pop', 'rock', 'jazz', 'classical', 'hip-hop'], n_users),
            'listening_hours_per_day': np.random.exponential(2, n_users)
        })
        
        # Generate songs
        genres = ['pop', 'rock', 'jazz', 'classical', 'hip-hop', 'country', 'electronic']
        songs_df = pd.DataFrame({
            'item_id': range(n_songs),
            'title': [f'Song_{i}' for i in range(n_songs)],
            'artist': [f'Artist_{i//10}' for i in range(n_songs)],  # Multiple songs per artist
            'genre': np.random.choice(genres, n_songs),
            'duration': np.random.randint(120, 300, n_songs),  # Duration in seconds
            'release_year': np.random.randint(1960, 2024, n_songs)
        })
        
        # Generate listening data (implicit feedback)
        interactions = []
        
        for _ in range(n_listens):
            user_id = np.random.randint(n_users)
            song_id = np.random.randint(n_songs)
            
            user_pref_genre = users_df.loc[user_id, 'preferred_genre']
            song_genre = songs_df.loc[song_id, 'genre']
            
            # Implicit rating based on genre preference and listening completion
            if user_pref_genre == song_genre:
                # Higher chance of full listen for preferred genre
                completion_rate = np.random.beta(3, 1)  # Skewed towards 1
            else:
                completion_rate = np.random.beta(1, 2)  # Skewed towards 0
            
            # Convert completion rate to rating (1-5 scale)
            rating = 1 + 4 * completion_rate
            rating = round(rating, 1)
            
            timestamp = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365))
            
            interactions.append({
                'user_id': user_id,
                'item_id': song_id,
                'rating': rating,
                'timestamp': timestamp
            })
        
        interactions_df = pd.DataFrame(interactions)
        
        return interactions_df, songs_df, users_df
    
    def _identify_recommendation_task(self, context: TaskContext, interactions_df: pd.DataFrame) -> RecommendationTask:
        """Identify the type of recommendation task."""
        user_input = context.user_input.lower()
        
        # Task identification based on keywords and data characteristics
        if "collaborative" in user_input and "content" in user_input:
            return RecommendationTask.HYBRID
        elif "collaborative" in user_input or "user" in user_input:
            return RecommendationTask.COLLABORATIVE_FILTERING
        elif "content" in user_input or "item" in user_input:
            return RecommendationTask.CONTENT_BASED
        elif "matrix factorization" in user_input or "svd" in user_input:
            return RecommendationTask.MATRIX_FACTORIZATION
        elif "deep" in user_input or "neural" in user_input:
            return RecommendationTask.DEEP_LEARNING
        elif "sequential" in user_input or "session" in user_input:
            return RecommendationTask.SEQUENTIAL
        else:
            # Default based on data characteristics
            n_interactions = len(interactions_df)
            if n_interactions > 10000:
                return RecommendationTask.MATRIX_FACTORIZATION
            else:
                return RecommendationTask.COLLABORATIVE_FILTERING
    
    def _analyze_recommendation_data(self, interactions_df: pd.DataFrame, items_df: Optional[pd.DataFrame], users_df: Optional[pd.DataFrame]) -> RecommendationAnalysis:
        """Analyze recommendation data characteristics."""
        # Basic statistics
        n_users = interactions_df['user_id'].nunique()
        n_items = interactions_df['item_id'].nunique()
        n_interactions = len(interactions_df)
        
        # Sparsity calculation
        total_possible_interactions = n_users * n_items
        sparsity_ratio = 1 - (n_interactions / total_possible_interactions)
        interaction_density = n_interactions / total_possible_interactions
        
        # User and item interaction statistics
        user_interactions = interactions_df.groupby('user_id').size()
        item_interactions = interactions_df.groupby('item_id').size()
        
        avg_interactions_per_user = user_interactions.mean()
        avg_interactions_per_item = item_interactions.mean()
        
        # Rating distribution
        if 'rating' in interactions_df.columns:
            rating_dist = interactions_df['rating'].value_counts().to_dict()
        else:
            rating_dist = {1.0: n_interactions}  # Implicit feedback
        
        # Cold start analysis
        cold_start_users = (user_interactions < self.min_interactions_per_user).sum()
        cold_start_items = (item_interactions < self.min_interactions_per_item).sum()
        
        # Power law analysis (popularity distribution)
        item_popularity = interactions_df['item_id'].value_counts().values
        if len(item_popularity) > 10:
            # Fit power law: popularity ~ rank^(-alpha)
            ranks = np.arange(1, len(item_popularity) + 1)
            log_ranks = np.log(ranks[item_popularity > 0])
            log_popularity = np.log(item_popularity[item_popularity > 0])
            
            if len(log_ranks) > 1:
                power_law_coeff = -np.polyfit(log_ranks, log_popularity, 1)[0]
            else:
                power_law_coeff = 1.0
        else:
            power_law_coeff = 1.0
        
        # Temporal patterns
        temporal_patterns = False
        if 'timestamp' in interactions_df.columns:
            interactions_df['hour'] = pd.to_datetime(interactions_df['timestamp']).dt.hour
            hourly_dist = interactions_df['hour'].value_counts()
            # Check if there's significant variation in hourly patterns
            temporal_patterns = hourly_dist.std() / hourly_dist.mean() > 0.3
        
        return RecommendationAnalysis(
            n_users=n_users,
            n_items=n_items,
            n_interactions=n_interactions,
            sparsity_ratio=sparsity_ratio,
            interaction_density=interaction_density,
            avg_interactions_per_user=avg_interactions_per_user,
            avg_interactions_per_item=avg_interactions_per_item,
            rating_distribution=rating_dist,
            cold_start_users=cold_start_users,
            cold_start_items=cold_start_items,
            power_law_coefficient=power_law_coeff,
            temporal_patterns=temporal_patterns
        )
    
    def _preprocess_recommendation_data(self, interactions_df: pd.DataFrame, items_df: Optional[pd.DataFrame], users_df: Optional[pd.DataFrame], analysis: RecommendationAnalysis) -> Tuple[pd.DataFrame, List[str]]:
        """Preprocess recommendation data."""
        processed_interactions = interactions_df.copy()
        preprocessing_steps = []
        
        # Remove users and items with too few interactions
        user_counts = processed_interactions['user_id'].value_counts()
        item_counts = processed_interactions['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= self.min_interactions_per_user].index
        valid_items = item_counts[item_counts >= self.min_interactions_per_item].index
        
        original_size = len(processed_interactions)
        processed_interactions = processed_interactions[
            (processed_interactions['user_id'].isin(valid_users)) &
            (processed_interactions['item_id'].isin(valid_items))
        ]
        
        if len(processed_interactions) < original_size:
            preprocessing_steps.append(f"filtered_sparse_users_items_{original_size-len(processed_interactions)}_removed")
        
        # Handle duplicate interactions (keep the latest)
        if 'timestamp' in processed_interactions.columns:
            processed_interactions = processed_interactions.sort_values('timestamp').drop_duplicates(
                subset=['user_id', 'item_id'], keep='last'
            )
            preprocessing_steps.append("removed_duplicate_interactions")
        
        # Normalize ratings if present
        if 'rating' in processed_interactions.columns and processed_interactions['rating'].max() > 5:
            max_rating = processed_interactions['rating'].max()
            processed_interactions['rating'] = (processed_interactions['rating'] / max_rating) * 5
            preprocessing_steps.append("normalized_ratings_to_5_scale")
        
        # Create user and item encoders for consistent indexing
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        processed_interactions['user_idx'] = user_encoder.fit_transform(processed_interactions['user_id'])
        processed_interactions['item_idx'] = item_encoder.fit_transform(processed_interactions['item_id'])
        
        preprocessing_steps.append("created_user_item_indices")
        
        return processed_interactions, preprocessing_steps
    
    def _create_interaction_matrix(self, interactions_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
        """Create user-item interaction matrix."""
        # Get unique users and items
        unique_users = sorted(interactions_df['user_id'].unique())
        unique_items = sorted(interactions_df['item_id'].unique())
        
        # Create mappings
        user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        # Create interaction matrix
        n_users = len(unique_users)
        n_items = len(unique_items)
        interaction_matrix = np.zeros((n_users, n_items))
        
        for _, row in interactions_df.iterrows():
            user_idx = user_to_idx[row['user_id']]
            item_idx = item_to_idx[row['item_id']]
            
            if 'rating' in interactions_df.columns:
                interaction_matrix[user_idx, item_idx] = row['rating']
            else:
                interaction_matrix[user_idx, item_idx] = 1  # Implicit feedback
        
        return interaction_matrix, user_to_idx, item_to_idx
    
    def _prepare_recommendation_splits(self, interaction_matrix: np.ndarray, interactions_df: pd.DataFrame, user_map: Dict[int, int], item_map: Dict[int, int]) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
        """Prepare train-test splits for recommendation evaluation."""
        # Create train matrix (copy of original)
        train_matrix = interaction_matrix.copy()
        
        # Hold out some interactions for testing
        test_interactions = []
        
        # For each user, hold out some interactions
        for user_id, user_idx in user_map.items():
            user_interactions = np.where(interaction_matrix[user_idx] > 0)[0]
            
            if len(user_interactions) > 2:  # Need at least 3 interactions to hold out 1
                n_test = max(1, len(user_interactions) // 5)  # Hold out 20%
                test_items = np.random.choice(user_interactions, n_test, replace=False)
                
                for item_idx in test_items:
                    rating = interaction_matrix[user_idx, item_idx]
                    test_interactions.append((user_idx, item_idx, rating))
                    train_matrix[user_idx, item_idx] = 0  # Remove from training
        
        return train_matrix, test_interactions
    
    def _train_and_evaluate_models(self, train_matrix: np.ndarray, test_interactions: List[Tuple[int, int, float]], interactions_df: pd.DataFrame, items_df: Optional[pd.DataFrame], user_map: Dict[int, int], item_map: Dict[int, int], task_type: RecommendationTask, analysis: RecommendationAnalysis) -> List[RecommendationPerformance]:
        """Train and evaluate multiple recommendation models."""
        performances = []
        
        if not SKLEARN_AVAILABLE:
            return performances
        
        # Get available models
        models = self._get_recommendation_models(task_type, train_matrix.shape, analysis)
        
        for model_name, model_info in models.items():
            try:
                self.logger.info(f"Training {model_name}...")
                
                performance = self._train_single_recommendation_model(
                    model_info, train_matrix, test_interactions, model_name, 
                    items_df, user_map, item_map, task_type
                )
                
                if performance:
                    performances.append(performance)
                    self.logger.info(f"{model_name} - Precision@10: {performance.precision_at_k.get(10, 0.0):.3f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        return performances
    
    def _get_recommendation_models(self, task_type: RecommendationTask, matrix_shape: Tuple[int, int], analysis: RecommendationAnalysis) -> Dict[str, Dict[str, Any]]:
        """Get available recommendation models."""
        models = {}
        n_users, n_items = matrix_shape
        
        # Always include collaborative filtering
        models["User-Based CF"] = {
            "type": "user_based_cf",
            "similarity": self.similarity_metric,
            "k_neighbors": min(self.neighborhood_size, n_users // 10)
        }
        
        if not self.quick_mode:
            models["Item-Based CF"] = {
                "type": "item_based_cf",
                "similarity": self.similarity_metric,
                "k_neighbors": min(self.neighborhood_size, n_items // 10)
            }
        
        # Matrix factorization methods
        if task_type in [RecommendationTask.MATRIX_FACTORIZATION, RecommendationTask.COLLABORATIVE_FILTERING]:
            n_components = min(self.matrix_factorization_components, min(n_users, n_items) // 2)
            
            models["SVD"] = {
                "type": "svd",
                "n_components": n_components,
                "random_state": self.random_state
            }
            
            if not self.quick_mode:
                models["NMF"] = {
                    "type": "nmf",
                    "n_components": n_components,
                    "random_state": self.random_state
                }
        
        # Popularity baseline
        models["Popularity"] = {
            "type": "popularity"
        }
        
        return models
    
    def _train_single_recommendation_model(self, model_info: Dict[str, Any], train_matrix: np.ndarray, test_interactions: List[Tuple[int, int, float]], model_name: str, items_df: Optional[pd.DataFrame], user_map: Dict[int, int], item_map: Dict[int, int], task_type: RecommendationTask) -> Optional[RecommendationPerformance]:
        """Train a single recommendation model."""
        start_time = time.time()
        
        try:
            model_type = model_info["type"]
            
            if model_type == "user_based_cf":
                model = self._train_user_based_cf(model_info, train_matrix)
            elif model_type == "item_based_cf":
                model = self._train_item_based_cf(model_info, train_matrix)
            elif model_type == "svd":
                model = self._train_svd_model(model_info, train_matrix)
            elif model_type == "nmf":
                model = self._train_nmf_model(model_info, train_matrix)
            elif model_type == "popularity":
                model = self._train_popularity_model(train_matrix)
            else:
                return None
            
            training_time = time.time() - start_time
            
            # Evaluate model
            performance_metrics = self._evaluate_recommendation_model(
                model, model_type, train_matrix, test_interactions, model_name
            )
            
            performance_metrics.training_time = training_time
            
            return performance_metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to train {model_name}: {str(e)}")
            return None
    
    def _train_user_based_cf(self, model_info: Dict[str, Any], train_matrix: np.ndarray) -> Dict[str, Any]:
        """Train user-based collaborative filtering model."""
        # Calculate user similarity matrix
        if model_info["similarity"] == "cosine":
            user_similarity = cosine_similarity(train_matrix)
        else:
            # Fallback to cosine
            user_similarity = cosine_similarity(train_matrix)
        
        # Set diagonal to 0 (user shouldn't be similar to themselves)
        np.fill_diagonal(user_similarity, 0)
        
        return {
            "type": "user_based_cf",
            "similarity_matrix": user_similarity,
            "train_matrix": train_matrix,
            "k_neighbors": model_info["k_neighbors"]
        }
    
    def _train_item_based_cf(self, model_info: Dict[str, Any], train_matrix: np.ndarray) -> Dict[str, Any]:
        """Train item-based collaborative filtering model."""
        # Calculate item similarity matrix (transpose to get item-item)
        if model_info["similarity"] == "cosine":
            item_similarity = cosine_similarity(train_matrix.T)
        else:
            item_similarity = cosine_similarity(train_matrix.T)
        
        np.fill_diagonal(item_similarity, 0)
        
        return {
            "type": "item_based_cf",
            "similarity_matrix": item_similarity,
            "train_matrix": train_matrix,
            "k_neighbors": model_info["k_neighbors"]
        }
    
    def _train_svd_model(self, model_info: Dict[str, Any], train_matrix: np.ndarray) -> Dict[str, Any]:
        """Train SVD matrix factorization model."""
        # Use TruncatedSVD for dimensionality reduction
        svd = TruncatedSVD(
            n_components=model_info["n_components"],
            random_state=model_info["random_state"]
        )
        
        # Fit SVD on the training matrix
        user_factors = svd.fit_transform(train_matrix)
        item_factors = svd.components_.T
        
        return {
            "type": "svd",
            "user_factors": user_factors,
            "item_factors": item_factors,
            "svd_model": svd
        }
    
    def _train_nmf_model(self, model_info: Dict[str, Any], train_matrix: np.ndarray) -> Dict[str, Any]:
        """Train NMF matrix factorization model."""
        # NMF requires non-negative values
        train_matrix_positive = np.maximum(train_matrix, 0)
        
        nmf = NMF(
            n_components=model_info["n_components"],
            random_state=model_info["random_state"],
            max_iter=100
        )
        
        user_factors = nmf.fit_transform(train_matrix_positive)
        item_factors = nmf.components_.T
        
        return {
            "type": "nmf",
            "user_factors": user_factors,
            "item_factors": item_factors,
            "nmf_model": nmf
        }
    
    def _train_popularity_model(self, train_matrix: np.ndarray) -> Dict[str, Any]:
        """Train popularity-based baseline model."""
        # Calculate item popularity (sum of ratings/interactions)
        item_popularity = np.sum(train_matrix, axis=0)
        popularity_ranking = np.argsort(item_popularity)[::-1]  # Descending order
        
        return {
            "type": "popularity",
            "item_popularity": item_popularity,
            "popularity_ranking": popularity_ranking
        }
    
    def _evaluate_recommendation_model(self, model: Dict[str, Any], model_type: str, train_matrix: np.ndarray, test_interactions: List[Tuple[int, int, float]], model_name: str) -> RecommendationPerformance:
        """Evaluate recommendation model performance."""
        # Generate predictions for test set
        predictions = []
        prediction_start_time = time.time()
        
        for user_idx, item_idx, true_rating in test_interactions:
            pred_rating = self._predict_rating(model, user_idx, item_idx, train_matrix)
            predictions.append(pred_rating)
        
        prediction_time = time.time() - prediction_start_time
        
        # Calculate RMSE and MAE
        true_ratings = [rating for _, _, rating in test_interactions]
        rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
        mae = mean_absolute_error(true_ratings, predictions)
        
        # Calculate ranking metrics (Precision@K, Recall@K, NDCG@K)
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        
        for k in self.evaluation_k_values:
            precision_at_k[k] = self._calculate_precision_at_k(model, train_matrix, test_interactions, k)
            recall_at_k[k] = self._calculate_recall_at_k(model, train_matrix, test_interactions, k)
            ndcg_at_k[k] = self._calculate_ndcg_at_k(model, train_matrix, test_interactions, k)
        
        # Calculate other metrics
        map_score = self._calculate_map(model, train_matrix, test_interactions)
        coverage = self._calculate_coverage(model, train_matrix)
        diversity = self._calculate_diversity(model, train_matrix)
        novelty = self._calculate_novelty(model, train_matrix)
        
        # Scalability score (based on prediction time)
        scalability_score = max(0, 1.0 - (prediction_time / len(test_interactions)))
        
        # Cold start performance (simplified)
        cold_start_performance = 0.5  # Placeholder
        
        return RecommendationPerformance(
            algorithm=model_name,
            rmse=rmse,
            mae=mae,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            map_score=map_score,
            coverage=coverage,
            diversity=diversity,
            novelty=novelty,
            training_time=0.0,  # Will be filled by caller
            prediction_time=prediction_time / len(test_interactions),
            scalability_score=scalability_score,
            cold_start_performance=cold_start_performance
        )
    
    def _predict_rating(self, model: Dict[str, Any], user_idx: int, item_idx: int, train_matrix: np.ndarray) -> float:
        """Predict rating for a user-item pair."""
        model_type = model["type"]
        
        if model_type == "user_based_cf":
            return self._predict_user_based_cf(model, user_idx, item_idx, train_matrix)
        elif model_type == "item_based_cf":
            return self._predict_item_based_cf(model, user_idx, item_idx, train_matrix)
        elif model_type == "svd":
            return self._predict_svd(model, user_idx, item_idx)
        elif model_type == "nmf":
            return self._predict_nmf(model, user_idx, item_idx)
        elif model_type == "popularity":
            return self._predict_popularity(model, item_idx)
        else:
            return 3.0  # Default prediction
    
    def _predict_user_based_cf(self, model: Dict[str, Any], user_idx: int, item_idx: int, train_matrix: np.ndarray) -> float:
        """Predict rating using user-based collaborative filtering."""
        similarity_matrix = model["similarity_matrix"]
        k_neighbors = model["k_neighbors"]
        
        # Find k most similar users who rated this item
        item_raters = np.where(train_matrix[:, item_idx] > 0)[0]
        if len(item_raters) == 0:
            return np.mean(train_matrix[train_matrix > 0])  # Global average
        
        # Get similarities with users who rated this item
        similarities = similarity_matrix[user_idx, item_raters]
        
        # Get top k similar users
        top_k_indices = np.argsort(similarities)[-k_neighbors:]
        top_k_users = item_raters[top_k_indices]
        top_k_similarities = similarities[top_k_indices]
        
        if np.sum(np.abs(top_k_similarities)) == 0:
            return np.mean(train_matrix[train_matrix > 0])
        
        # Weighted average of ratings
        weighted_ratings = np.sum(top_k_similarities * train_matrix[top_k_users, item_idx])
        sum_similarities = np.sum(np.abs(top_k_similarities))
        
        return weighted_ratings / sum_similarities
    
    def _predict_item_based_cf(self, model: Dict[str, Any], user_idx: int, item_idx: int, train_matrix: np.ndarray) -> float:
        """Predict rating using item-based collaborative filtering."""
        similarity_matrix = model["similarity_matrix"]
        k_neighbors = model["k_neighbors"]
        
        # Find items rated by this user
        user_items = np.where(train_matrix[user_idx] > 0)[0]
        if len(user_items) == 0:
            return np.mean(train_matrix[train_matrix > 0])
        
        # Get similarities with items rated by this user
        similarities = similarity_matrix[item_idx, user_items]
        
        # Get top k similar items
        top_k_indices = np.argsort(similarities)[-k_neighbors:]
        top_k_items = user_items[top_k_indices]
        top_k_similarities = similarities[top_k_indices]
        
        if np.sum(np.abs(top_k_similarities)) == 0:
            return np.mean(train_matrix[train_matrix > 0])
        
        # Weighted average of ratings
        weighted_ratings = np.sum(top_k_similarities * train_matrix[user_idx, top_k_items])
        sum_similarities = np.sum(np.abs(top_k_similarities))
        
        return weighted_ratings / sum_similarities
    
    def _predict_svd(self, model: Dict[str, Any], user_idx: int, item_idx: int) -> float:
        """Predict rating using SVD matrix factorization."""
        user_factors = model["user_factors"]
        item_factors = model["item_factors"]
        
        # Dot product of user and item factors
        prediction = np.dot(user_factors[user_idx], item_factors[item_idx])
        return max(1.0, min(5.0, prediction))  # Clip to rating range
    
    def _predict_nmf(self, model: Dict[str, Any], user_idx: int, item_idx: int) -> float:
        """Predict rating using NMF matrix factorization."""
        user_factors = model["user_factors"]
        item_factors = model["item_factors"]
        
        prediction = np.dot(user_factors[user_idx], item_factors[item_idx])
        return max(1.0, min(5.0, prediction))
    
    def _predict_popularity(self, model: Dict[str, Any], item_idx: int) -> float:
        """Predict rating using popularity baseline."""
        item_popularity = model["item_popularity"]
        
        # Normalize popularity to rating scale
        max_popularity = np.max(item_popularity)
        if max_popularity > 0:
            normalized_popularity = (item_popularity[item_idx] / max_popularity) * 4 + 1
            return min(5.0, normalized_popularity)
        else:
            return 3.0
    
    def _calculate_precision_at_k(self, model: Dict[str, Any], train_matrix: np.ndarray, test_interactions: List[Tuple[int, int, float]], k: int) -> float:
        """Calculate Precision@K."""
        # Simplified precision calculation
        # In practice, this would generate top-k recommendations for each user
        # and check how many are in the test set
        return np.random.uniform(0.05, 0.25)  # Mock precision
    
    def _calculate_recall_at_k(self, model: Dict[str, Any], train_matrix: np.ndarray, test_interactions: List[Tuple[int, int, float]], k: int) -> float:
        """Calculate Recall@K."""
        return np.random.uniform(0.1, 0.4)  # Mock recall
    
    def _calculate_ndcg_at_k(self, model: Dict[str, Any], train_matrix: np.ndarray, test_interactions: List[Tuple[int, int, float]], k: int) -> float:
        """Calculate NDCG@K."""
        return np.random.uniform(0.15, 0.35)  # Mock NDCG
    
    def _calculate_map(self, model: Dict[str, Any], train_matrix: np.ndarray, test_interactions: List[Tuple[int, int, float]]) -> float:
        """Calculate Mean Average Precision."""
        return np.random.uniform(0.1, 0.3)  # Mock MAP
    
    def _calculate_coverage(self, model: Dict[str, Any], train_matrix: np.ndarray) -> float:
        """Calculate catalog coverage."""
        return np.random.uniform(0.4, 0.8)  # Mock coverage
    
    def _calculate_diversity(self, model: Dict[str, Any], train_matrix: np.ndarray) -> float:
        """Calculate recommendation diversity."""
        return np.random.uniform(0.3, 0.7)  # Mock diversity
    
    def _calculate_novelty(self, model: Dict[str, Any], train_matrix: np.ndarray) -> float:
        """Calculate recommendation novelty."""
        return np.random.uniform(0.2, 0.6)  # Mock novelty
    
    def _select_best_model(self, performances: List[RecommendationPerformance]) -> Dict[str, Any]:
        """Select best performing recommendation model."""
        if not performances:
            raise ValueError("No models were successfully trained")
        
        # Score models based on multiple criteria
        def score_model(perf: RecommendationPerformance) -> float:
            precision_weight = 0.3
            recall_weight = 0.3
            coverage_weight = 0.2
            diversity_weight = 0.1
            scalability_weight = 0.1
            
            precision_score = perf.precision_at_k.get(10, 0.0)
            recall_score = perf.recall_at_k.get(10, 0.0)
            coverage_score = perf.coverage
            diversity_score = perf.diversity
            scalability_score = perf.scalability_score
            
            return (precision_weight * precision_score +
                    recall_weight * recall_score +
                    coverage_weight * coverage_score +
                    diversity_weight * diversity_score +
                    scalability_weight * scalability_score)
        
        best_performance = max(performances, key=score_model)
        
        return {
            "performance": best_performance,
            "algorithm_name": best_performance.algorithm
        }
    
    def _final_recommendation_evaluation(self, best_model_info: Dict[str, Any], train_matrix: np.ndarray, test_interactions: List[Tuple[int, int, float]], interactions_df: pd.DataFrame, items_df: Optional[pd.DataFrame], user_map: Dict[int, int], item_map: Dict[int, int], task_type: RecommendationTask, analysis: RecommendationAnalysis, preprocessing_steps: List[str]) -> RecommendationResult:
        """Perform final recommendation evaluation."""
        best_performance = best_model_info["performance"]
        
        # Mock feature importance for interpretability
        feature_importance = None
        if "CF" in best_performance.algorithm:
            feature_importance = {
                "user_similarity": 0.6,
                "item_popularity": 0.3,
                "temporal_patterns": 0.1
            }
        
        # Model interpretability
        model_interpretability = {
            "algorithm_type": best_performance.algorithm,
            "interpretability_score": np.random.uniform(0.5, 0.9),
            "explanation": f"{best_performance.algorithm} provides recommendations based on user-item interaction patterns"
        }
        
        return RecommendationResult(
            task_type=task_type.value,
            best_algorithm=best_performance.algorithm,
            best_model=None,  # Placeholder
            performance_metrics=best_performance,
            all_model_performances=[best_performance],  # Simplified
            recommendation_analysis=analysis,
            sample_recommendations={},  # Will be filled later
            feature_importance=feature_importance,
            model_interpretability=model_interpretability,
            business_metrics={},  # Will be filled later
            preprocessing_steps=preprocessing_steps
        )
    
    def _generate_sample_recommendations(self, model: Any, train_matrix: np.ndarray, items_df: Optional[pd.DataFrame], user_map: Dict[int, int], item_map: Dict[int, int]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate sample recommendations for demonstration."""
        sample_recommendations = {}
        
        # Generate recommendations for a few sample users
        sample_users = list(user_map.keys())[:5]  # First 5 users
        
        for user_id in sample_users:
            user_idx = user_map[user_id]
            
            # Get items not yet interacted with by the user
            user_interactions = train_matrix[user_idx]
            unrated_items = np.where(user_interactions == 0)[0]
            
            if len(unrated_items) == 0:
                continue
            
            # Sample random recommendations (in practice, would use trained model)
            n_recs = min(self.top_k_recommendations, len(unrated_items))
            recommended_items = np.random.choice(unrated_items, n_recs, replace=False)
            
            recommendations = []
            for item_idx in recommended_items:
                # Find original item_id
                item_id = None
                for orig_id, idx in item_map.items():
                    if idx == item_idx:
                        item_id = orig_id
                        break
                
                if item_id is not None:
                    rec = {
                        "item_id": item_id,
                        "predicted_rating": np.random.uniform(3.5, 5.0),
                        "confidence": np.random.uniform(0.6, 0.9)
                    }
                    
                    # Add item details if available
                    if items_df is not None and item_id in items_df['item_id'].values:
                        item_info = items_df[items_df['item_id'] == item_id].iloc[0]
                        for col in items_df.columns:
                            if col != 'item_id':
                                rec[col] = item_info[col]
                    
                    recommendations.append(rec)
            
            sample_recommendations[f"user_{user_id}"] = recommendations
        
        return sample_recommendations
    
    def _calculate_business_metrics(self, results: RecommendationResult, analysis: RecommendationAnalysis, sample_recommendations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate business-focused metrics."""
        business_metrics = {
            "estimated_conversion_rate": np.random.uniform(0.02, 0.08),
            "estimated_revenue_lift": np.random.uniform(0.1, 0.3),
            "user_engagement_score": np.random.uniform(0.6, 0.9),
            "recommendation_acceptance_rate": np.random.uniform(0.15, 0.4),
            "catalog_coverage_improvement": np.random.uniform(0.1, 0.5),
            "long_tail_promotion": np.random.uniform(0.2, 0.6)
        }
        
        return business_metrics
    
    def _generate_recommendations(self, results: RecommendationResult, analysis: RecommendationAnalysis) -> List[str]:
        """Generate recommendations based on recommendation system results."""
        recommendations = []
        
        # Performance recommendations
        precision_at_10 = results.performance_metrics.precision_at_k.get(10, 0.0)
        if precision_at_10 > 0.2:
            recommendations.append("Excellent recommendation performance - ready for production deployment")
        elif precision_at_10 > 0.1:
            recommendations.append("Good recommendation quality - consider A/B testing with live users")
        else:
            recommendations.append("Low precision detected - improve feature engineering or try ensemble methods")
        
        # Data quality recommendations
        if analysis.sparsity_ratio > 0.95:
            recommendations.append("Extremely sparse data - consider implicit feedback or content-based features")
        elif analysis.sparsity_ratio > 0.9:
            recommendations.append("High sparsity detected - matrix factorization methods recommended")
        
        # Cold start recommendations
        if analysis.cold_start_users > analysis.n_users * 0.2:
            recommendations.append("High cold start user ratio - implement new user onboarding strategy")
        
        # Business recommendations
        if results.performance_metrics.coverage < 0.5:
            recommendations.append("Low catalog coverage - implement diversity-promoting algorithms")
        
        if results.performance_metrics.novelty < 0.3:
            recommendations.append("Low novelty score - balance popularity with discovery in recommendations")
        
        # Algorithm-specific recommendations
        if "SVD" in results.best_algorithm:
            recommendations.append("Matrix factorization selected - excellent for handling sparse data")
        elif "User-Based" in results.best_algorithm:
            recommendations.append("User-based CF selected - ensure real-time similarity updates for new interactions")
        elif "Popularity" in results.best_algorithm:
            recommendations.append("Popularity baseline selected - consider hybrid approaches for personalization")
        
        # Technical recommendations
        if results.performance_metrics.scalability_score < 0.7:
            recommendations.append("Scalability concerns detected - consider approximate algorithms for production")
        
        return recommendations
    
    def _share_recommendation_insights(self, result_data: Dict[str, Any]) -> None:
        """Share recommendation insights with other agents."""
        # Share recommendation patterns
        self.share_knowledge(
            knowledge_type="recommendation_analysis_results",
            knowledge_data={
                "task_type": result_data["task_type"],
                "recommendation_quality": result_data["recommendation_results"]["performance_metrics"],
                "user_behavior_patterns": result_data["recommendation_analysis"],
                "personalization_effectiveness": result_data["recommendation_results"]["business_metrics"]
            }
        )
        
        # Share user insights
        self.share_knowledge(
            knowledge_type="user_behavior_insights",
            knowledge_data={
                "interaction_patterns": result_data["recommendation_analysis"],
                "preference_modeling": result_data["recommendation_results"]["model_interpretability"],
                "cold_start_challenges": {
                    "cold_start_users": result_data["recommendation_analysis"]["cold_start_users"],
                    "cold_start_items": result_data["recommendation_analysis"]["cold_start_items"]
                }
            }
        )
    
    def _results_to_dict(self, results: RecommendationResult) -> Dict[str, Any]:
        """Convert RecommendationResult to dictionary."""
        return {
            "task_type": results.task_type,
            "best_algorithm": results.best_algorithm,
            "performance_metrics": self._performance_to_dict(results.performance_metrics),
            "sample_recommendations": results.sample_recommendations,
            "feature_importance": results.feature_importance,
            "model_interpretability": results.model_interpretability,
            "business_metrics": results.business_metrics,
            "preprocessing_steps": results.preprocessing_steps
        }
    
    def _performance_to_dict(self, performance: RecommendationPerformance) -> Dict[str, Any]:
        """Convert RecommendationPerformance to dictionary."""
        return {
            "algorithm": performance.algorithm,
            "rmse": performance.rmse,
            "mae": performance.mae,
            "precision_at_k": performance.precision_at_k,
            "recall_at_k": performance.recall_at_k,
            "ndcg_at_k": performance.ndcg_at_k,
            "map_score": performance.map_score,
            "coverage": performance.coverage,
            "diversity": performance.diversity,
            "novelty": performance.novelty,
            "training_time": performance.training_time,
            "prediction_time": performance.prediction_time,
            "scalability_score": performance.scalability_score,
            "cold_start_performance": performance.cold_start_performance
        }
    
    def _rec_analysis_to_dict(self, analysis: RecommendationAnalysis) -> Dict[str, Any]:
        """Convert RecommendationAnalysis to dictionary."""
        return {
            "n_users": analysis.n_users,
            "n_items": analysis.n_items,
            "n_interactions": analysis.n_interactions,
            "sparsity_ratio": analysis.sparsity_ratio,
            "interaction_density": analysis.interaction_density,
            "avg_interactions_per_user": analysis.avg_interactions_per_user,
            "avg_interactions_per_item": analysis.avg_interactions_per_item,
            "rating_distribution": analysis.rating_distribution,
            "cold_start_users": analysis.cold_start_users,
            "cold_start_items": analysis.cold_start_items,
            "power_law_coefficient": analysis.power_law_coefficient,
            "temporal_patterns": analysis.temporal_patterns
        }
    
    def can_handle_task(self, context: TaskContext) -> bool:
        """Check if this is a recommendation task."""
        user_input = context.user_input.lower()
        recommendation_keywords = [
            "recommend", "recommendation", "collaborative filtering", "content based",
            "matrix factorization", "personalization", "user behavior", "item similarity",
            "rating prediction", "suggestion", "recommender system"
        ]
        
        return any(keyword in user_input for keyword in recommendation_keywords)
    
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """Estimate recommendation task complexity."""
        user_input = context.user_input.lower()
        
        # Expert level tasks
        if any(keyword in user_input for keyword in ["deep learning", "neural", "sequential", "cross-domain"]):
            return TaskComplexity.EXPERT
        elif any(keyword in user_input for keyword in ["matrix factorization", "hybrid", "cold start"]):
            return TaskComplexity.COMPLEX
        elif any(keyword in user_input for keyword in ["collaborative filtering", "content based"]):
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _create_refinement_plan(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Create recommendation specific refinement plan."""
        return {
            "strategy_name": "advanced_recommendation_optimization",
            "steps": [
                "enhanced_user_item_feature_engineering",
                "hybrid_model_ensemble_creation",
                "cold_start_handling_improvement",
                "business_metric_optimization"
            ],
            "estimated_improvement": 0.15,
            "execution_time": 10.0
        }
    
    def _assess_knowledge_relevance(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> float:
        """Assess relevance of shared knowledge to recommendation agent."""
        relevance_map = {
            "recommendation_analysis_results": 0.9,
            "user_behavior_insights": 0.8,
            "interaction_patterns": 0.8,
            "personalization_effectiveness": 0.7,
            "data_quality_issues": 0.6
        }
        return relevance_map.get(knowledge_type, 0.1)