"""
Clustering Agent for AutoML Platform

Specialized agent for unsupervised learning and clustering analysis that:
1. Handles various clustering algorithms and distance metrics
2. Implements cluster validation and optimal cluster selection
3. Supports dimensionality reduction and visualization
4. Provides cluster analysis and interpretation
5. Handles different data types and preprocessing

This agent runs for unsupervised learning problems and pattern discovery tasks.
"""

import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from sklearn.cluster import (
        KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
        MeanShift, OPTICS, Birch
    )
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        silhouette_score, calinski_harabasz_score, davies_bouldin_score,
        adjusted_rand_score, normalized_mutual_info_score
    )
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, TaskContext, TaskComplexity


class ClusteringTask(Enum):
    """Types of clustering tasks."""
    DISCOVERY = "discovery"
    SEGMENTATION = "segmentation"
    ANOMALY_DETECTION = "anomaly_detection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    MARKET_BASKET = "market_basket"
    TIME_SERIES_CLUSTERING = "time_series_clustering"


class ClusteringMethod(Enum):
    """Clustering algorithms."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    SPECTRAL = "spectral"
    MEANSHIFT = "meanshift"
    OPTICS = "optics"
    BIRCH = "birch"


class DistanceMetric(Enum):
    """Distance metrics for clustering."""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    CORRELATION = "correlation"
    HAMMING = "hamming"
    JACCARD = "jaccard"


@dataclass
class ClusteringAnalysis:
    """Clustering data analysis results."""
    n_samples: int
    n_features: int
    feature_types: Dict[str, str]
    data_density: float
    dimensionality_score: float
    correlation_matrix_rank: int
    missing_values_ratio: float
    outlier_ratio: float
    optimal_preprocessing: List[str]
    recommended_k_range: Tuple[int, int]


@dataclass
class ClusterPerformance:
    """Clustering model performance metrics."""
    algorithm: str
    n_clusters: int
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    inertia: Optional[float]
    cluster_sizes: List[int]
    cluster_density: List[float]
    training_time: float
    scalability_score: float
    interpretability_score: float


@dataclass
class ClusterResult:
    """Complete clustering result."""
    task_type: str
    best_algorithm: str
    best_model: Any
    performance_metrics: ClusterPerformance
    all_model_performances: List[ClusterPerformance]
    clustering_analysis: ClusteringAnalysis
    cluster_labels: List[int]
    cluster_centers: Optional[List[List[float]]]
    cluster_profiles: Dict[str, Any]
    dimensionality_reduction: Optional[Dict[str, Any]]
    visualization_data: Optional[Dict[str, Any]]
    preprocessing_steps: List[str]


class ClusteringAgent(BaseAgent):
    """
    Clustering Agent for unsupervised learning and pattern discovery.
    
    Responsibilities:
    1. Data preprocessing for clustering
    2. Clustering algorithm selection and training
    3. Cluster validation and optimization
    4. Dimensionality reduction and visualization
    5. Cluster interpretation and profiling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub=None):
        """Initialize the Clustering Agent."""
        super().__init__(
            name="Clustering Agent",
            description="Advanced unsupervised learning and clustering specialist",
            specialization="Clustering & Pattern Discovery",
            config=config,
            communication_hub=communication_hub
        )
        
        # Clustering configuration
        self.max_clusters = self.config.get("max_clusters", 10)
        self.min_clusters = self.config.get("min_clusters", 2)
        self.auto_select_k = self.config.get("auto_select_k", True)
        self.include_visualization = self.config.get("include_visualization", True)
        
        # Preprocessing settings
        self.apply_scaling = self.config.get("apply_scaling", True)
        self.scaling_method = self.config.get("scaling_method", "standard")
        self.handle_categorical = self.config.get("handle_categorical", True)
        self.remove_outliers = self.config.get("remove_outliers", False)
        
        # Algorithm settings
        self.try_multiple_algorithms = self.config.get("try_multiple_algorithms", True)
        self.quick_mode = self.config.get("quick_mode", False)
        self.random_state = self.config.get("random_state", 42)
        
        # Quality thresholds
        self.quality_thresholds.update({
            "min_silhouette_score": self.config.get("min_silhouette_score", 0.3),
            "min_calinski_harabasz": self.config.get("min_calinski_harabasz", 10.0),
            "max_davies_bouldin": self.config.get("max_davies_bouldin", 2.0),
            "min_cluster_size": self.config.get("min_cluster_size", 5)
        })
        
        # Dimensionality reduction settings
        self.apply_dim_reduction = self.config.get("apply_dim_reduction", True)
        self.max_features_for_tsne = self.config.get("max_features_for_tsne", 50)
        self.pca_variance_threshold = self.config.get("pca_variance_threshold", 0.95)
        
        # Scaler
        self.scaler = None
    
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute comprehensive clustering workflow.
        
        Args:
            context: Task context with dataset
            
        Returns:
            AgentResult with clustering models and analysis
        """
        try:
            self.logger.info("Starting clustering analysis workflow...")
            
            # Load dataset
            df = self._load_clustering_dataset(context)
            if df is None:
                return AgentResult(
                    success=False,
                    message="Failed to load clustering dataset"
                )
            
            # Phase 1: Task Identification
            self.logger.info("Phase 1: Identifying clustering task...")
            task_type = self._identify_clustering_task(context, df)
            
            # Phase 2: Data Analysis
            self.logger.info("Phase 2: Analyzing data characteristics...")
            clustering_analysis = self._analyze_clustering_data(df)
            
            # Phase 3: Data Preprocessing
            self.logger.info("Phase 3: Preprocessing data for clustering...")
            df_processed, preprocessing_steps = self._preprocess_clustering_data(
                df, clustering_analysis
            )
            
            # Phase 4: Dimensionality Reduction (if needed)
            dim_reduction_result = None
            if self.apply_dim_reduction and df_processed.shape[1] > 10:
                self.logger.info("Phase 4: Applying dimensionality reduction...")
                df_processed, dim_reduction_result = self._apply_dimensionality_reduction(
                    df_processed, clustering_analysis
                )
            
            # Phase 5: Optimal Cluster Number Selection
            self.logger.info("Phase 5: Determining optimal number of clusters...")
            optimal_k = self._determine_optimal_clusters(df_processed, clustering_analysis)
            
            # Phase 6: Model Training and Evaluation
            self.logger.info("Phase 6: Training and evaluating clustering models...")
            model_performances = self._train_and_evaluate_models(
                df_processed, optimal_k, task_type, clustering_analysis
            )
            
            # Phase 7: Select Best Model
            self.logger.info("Phase 7: Selecting best performing model...")
            best_model_info = self._select_best_model(model_performances)
            
            # Phase 8: Final Clustering and Analysis
            self.logger.info("Phase 8: Final clustering and analysis...")
            final_results = self._final_clustering_analysis(
                best_model_info, df_processed, df, task_type, clustering_analysis,
                dim_reduction_result, preprocessing_steps
            )
            
            # Phase 9: Cluster Profiling
            self.logger.info("Phase 9: Generating cluster profiles...")
            cluster_profiles = self._generate_cluster_profiles(df, final_results.cluster_labels)
            final_results.cluster_profiles = cluster_profiles
            
            # Phase 10: Visualization
            if self.include_visualization:
                self.logger.info("Phase 10: Creating visualizations...")
                visualization_data = self._create_visualizations(
                    df_processed, final_results.cluster_labels, dim_reduction_result
                )
                final_results.visualization_data = visualization_data
            
            # Create comprehensive result
            result_data = {
                "clustering_results": self._results_to_dict(final_results),
                "clustering_analysis": self._clustering_analysis_to_dict(clustering_analysis),
                "task_type": task_type.value,
                "preprocessing_steps": preprocessing_steps,
                "dimensionality_reduction": dim_reduction_result,
                "model_performances": [self._performance_to_dict(perf) for perf in model_performances],
                "recommendations": self._generate_recommendations(final_results, clustering_analysis)
            }
            
            # Update performance metrics
            performance_metrics = {
                "clustering_silhouette": final_results.performance_metrics.silhouette_score,
                "clustering_quality": (final_results.performance_metrics.silhouette_score + 
                                     (2.0 - final_results.performance_metrics.davies_bouldin_score)) / 2,
                "cluster_interpretability": final_results.performance_metrics.interpretability_score,
                "pattern_discovery_efficiency": 1.0 / (final_results.performance_metrics.training_time + 1)
            }
            self.update_performance_metrics(performance_metrics)
            
            # Share clustering insights
            if self.communication_hub:
                self._share_clustering_insights(result_data)
            
            return AgentResult(
                success=True,
                data=result_data,
                message=f"Clustering workflow completed: {task_type.value} with {final_results.performance_metrics.n_clusters} clusters (silhouette: {final_results.performance_metrics.silhouette_score:.3f})",
                recommendations=result_data["recommendations"]
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Clustering workflow failed: {str(e)}"
            )
    
    def _load_clustering_dataset(self, context: TaskContext) -> Optional[pd.DataFrame]:
        """Load clustering dataset or create synthetic data."""
        # In real implementation, this would load from files or previous agent results
        # For demo, create synthetic clustering data
        
        user_input = context.user_input.lower()
        
        if "customer" in user_input or "segment" in user_input:
            return self._create_customer_segmentation_dataset()
        elif "market" in user_input or "basket" in user_input:
            return self._create_market_basket_dataset()
        elif "gene" in user_input or "biology" in user_input:
            return self._create_genomic_dataset()
        elif "document" in user_input or "text" in user_input:
            return self._create_document_clustering_dataset()
        else:
            return self._create_general_clustering_dataset()
    
    def _create_general_clustering_dataset(self) -> pd.DataFrame:
        """Create general synthetic clustering dataset."""
        np.random.seed(42)
        
        # Create 3 natural clusters
        n_samples_per_cluster = 100
        n_features = 5
        
        # Cluster 1: High values
        cluster1 = np.random.multivariate_normal(
            mean=[8, 7, 6, 5, 4],
            cov=np.eye(n_features) * 1.5,
            size=n_samples_per_cluster
        )
        
        # Cluster 2: Medium values
        cluster2 = np.random.multivariate_normal(
            mean=[3, 4, 5, 6, 7],
            cov=np.eye(n_features) * 1.0,
            size=n_samples_per_cluster
        )
        
        # Cluster 3: Low values
        cluster3 = np.random.multivariate_normal(
            mean=[1, 2, 1, 2, 1],
            cov=np.eye(n_features) * 0.8,
            size=n_samples_per_cluster
        )
        
        # Combine clusters
        data = np.vstack([cluster1, cluster2, cluster3])
        true_labels = np.hstack([
            np.zeros(n_samples_per_cluster),
            np.ones(n_samples_per_cluster),
            np.full(n_samples_per_cluster, 2)
        ])
        
        # Create DataFrame
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(data, columns=feature_names)
        df['true_cluster'] = true_labels  # For validation purposes
        
        return df
    
    def _create_customer_segmentation_dataset(self) -> pd.DataFrame:
        """Create synthetic customer segmentation dataset."""
        np.random.seed(42)
        
        n_customers = 500
        
        # Generate customer data with natural segments
        # Segment 1: High-value customers
        high_value = pd.DataFrame({
            'annual_spend': np.random.normal(5000, 1000, n_customers // 3),
            'frequency': np.random.normal(25, 5, n_customers // 3),
            'recency_days': np.random.normal(15, 5, n_customers // 3),
            'avg_order_value': np.random.normal(200, 50, n_customers // 3),
            'customer_age': np.random.normal(45, 10, n_customers // 3),
            'segment': 'high_value'
        })
        
        # Segment 2: Regular customers
        regular = pd.DataFrame({
            'annual_spend': np.random.normal(2000, 500, n_customers // 3),
            'frequency': np.random.normal(12, 3, n_customers // 3),
            'recency_days': np.random.normal(30, 10, n_customers // 3),
            'avg_order_value': np.random.normal(100, 25, n_customers // 3),
            'customer_age': np.random.normal(35, 8, n_customers // 3),
            'segment': 'regular'
        })
        
        # Segment 3: Occasional customers
        occasional = pd.DataFrame({
            'annual_spend': np.random.normal(500, 200, n_customers - 2 * (n_customers // 3)),
            'frequency': np.random.normal(3, 1, n_customers - 2 * (n_customers // 3)),
            'recency_days': np.random.normal(90, 30, n_customers - 2 * (n_customers // 3)),
            'avg_order_value': np.random.normal(50, 15, n_customers - 2 * (n_customers // 3)),
            'customer_age': np.random.normal(28, 5, n_customers - 2 * (n_customers // 3)),
            'segment': 'occasional'
        })
        
        # Combine segments
        df = pd.concat([high_value, regular, occasional], ignore_index=True)
        
        # Add some categorical features
        df['preferred_category'] = np.random.choice(['electronics', 'clothing', 'books', 'home'], len(df))
        df['acquisition_channel'] = np.random.choice(['online', 'store', 'referral', 'ad'], len(df))
        
        return df
    
    def _create_market_basket_dataset(self) -> pd.DataFrame:
        """Create synthetic market basket analysis dataset."""
        np.random.seed(42)
        
        n_transactions = 1000
        items = ['bread', 'milk', 'eggs', 'butter', 'cheese', 'yogurt', 'apples', 'bananas', 
                'chicken', 'beef', 'rice', 'pasta', 'tomatoes', 'onions', 'lettuce']
        
        # Create binary matrix for market basket analysis
        data = np.zeros((n_transactions, len(items)))
        
        # Create some common patterns
        for i in range(n_transactions):
            # Random basket size
            basket_size = np.random.poisson(5) + 1
            
            # Common associations
            if np.random.random() < 0.3:  # Breakfast items
                data[i, [0, 1, 2]] = 1  # bread, milk, eggs
            if np.random.random() < 0.2:  # Dairy cluster
                data[i, [1, 3, 4, 5]] = 1  # milk, butter, cheese, yogurt
            if np.random.random() < 0.25:  # Fruit cluster
                data[i, [6, 7]] = 1  # apples, bananas
            
            # Random additional items
            remaining_items = basket_size - data[i].sum()
            if remaining_items > 0:
                random_indices = np.random.choice(len(items), 
                                                 min(int(remaining_items), len(items)), 
                                                 replace=False)
                data[i, random_indices] = 1
        
        df = pd.DataFrame(data, columns=items)
        return df
    
    def _create_genomic_dataset(self) -> pd.DataFrame:
        """Create synthetic genomic expression dataset."""
        np.random.seed(42)
        
        n_samples = 200
        n_genes = 50
        
        # Create gene expression data with different cell types
        # Cell type 1: Neural
        neural_data = np.random.lognormal(mean=2, sigma=1, size=(n_samples // 3, n_genes))
        neural_data[:, :10] *= 3  # Upregulate first 10 genes
        
        # Cell type 2: Muscle
        muscle_data = np.random.lognormal(mean=1.5, sigma=0.8, size=(n_samples // 3, n_genes))
        muscle_data[:, 10:20] *= 4  # Upregulate genes 10-20
        
        # Cell type 3: Immune
        immune_data = np.random.lognormal(mean=1.8, sigma=1.2, size=(n_samples - 2 * (n_samples // 3), n_genes))
        immune_data[:, 20:30] *= 5  # Upregulate genes 20-30
        
        # Combine data
        data = np.vstack([neural_data, muscle_data, immune_data])
        
        # Create DataFrame
        gene_names = [f'gene_{i+1}' for i in range(n_genes)]
        df = pd.DataFrame(data, columns=gene_names)
        
        # Add metadata
        cell_types = ['neural'] * (n_samples // 3) + ['muscle'] * (n_samples // 3) + ['immune'] * (n_samples - 2 * (n_samples // 3))
        df['cell_type'] = cell_types
        
        return df
    
    def _create_document_clustering_dataset(self) -> pd.DataFrame:
        """Create synthetic document clustering dataset (simplified TF-IDF)."""
        np.random.seed(42)
        
        n_documents = 300
        n_terms = 20
        
        # Create document-term matrix with topic clusters
        # Topic 1: Technology
        tech_docs = np.random.poisson(2, (n_documents // 3, n_terms))
        tech_docs[:, :5] += np.random.poisson(5, (n_documents // 3, 5))  # High tech terms
        
        # Topic 2: Sports
        sports_docs = np.random.poisson(2, (n_documents // 3, n_terms))
        sports_docs[:, 5:10] += np.random.poisson(5, (n_documents // 3, 5))  # High sports terms
        
        # Topic 3: Politics
        politics_docs = np.random.poisson(2, (n_documents - 2 * (n_documents // 3), n_terms))
        politics_docs[:, 10:15] += np.random.poisson(5, (n_documents - 2 * (n_documents // 3), 5))  # High politics terms
        
        # Combine documents
        data = np.vstack([tech_docs, sports_docs, politics_docs])
        
        # Create DataFrame
        term_names = [f'term_{i+1}' for i in range(n_terms)]
        df = pd.DataFrame(data, columns=term_names)
        
        # Add document metadata
        topics = ['technology'] * (n_documents // 3) + ['sports'] * (n_documents // 3) + ['politics'] * (n_documents - 2 * (n_documents // 3))
        df['topic'] = topics
        
        return df
    
    def _identify_clustering_task(self, context: TaskContext, df: pd.DataFrame) -> ClusteringTask:
        """Identify the type of clustering task."""
        user_input = context.user_input.lower()
        
        # Task identification based on keywords
        if "segment" in user_input or "customer" in user_input:
            return ClusteringTask.SEGMENTATION
        elif "anomaly" in user_input or "outlier" in user_input:
            return ClusteringTask.ANOMALY_DETECTION
        elif "dimension" in user_input or "reduce" in user_input:
            return ClusteringTask.DIMENSIONALITY_REDUCTION
        elif "market" in user_input and "basket" in user_input:
            return ClusteringTask.MARKET_BASKET
        elif "time series" in user_input or "temporal" in user_input:
            return ClusteringTask.TIME_SERIES_CLUSTERING
        else:
            # Default to pattern discovery
            return ClusteringTask.DISCOVERY
    
    def _analyze_clustering_data(self, df: pd.DataFrame) -> ClusteringAnalysis:
        """Analyze data characteristics for clustering."""
        # Basic properties
        n_samples, n_features = df.shape
        
        # Exclude non-numeric columns for analysis
        numeric_df = df.select_dtypes(include=[np.number])
        n_numeric_features = numeric_df.shape[1]
        
        # Feature types
        feature_types = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                feature_types[col] = 'numeric'
            elif df[col].dtype == 'object':
                feature_types[col] = 'categorical'
            else:
                feature_types[col] = 'other'
        
        # Data density (ratio of non-zero values)
        if n_numeric_features > 0:
            non_zero_ratio = (numeric_df != 0).sum().sum() / (n_samples * n_numeric_features)
            data_density = non_zero_ratio
        else:
            data_density = 1.0
        
        # Dimensionality score (curse of dimensionality indicator)
        dimensionality_score = min(1.0, n_samples / (n_numeric_features * 10)) if n_numeric_features > 0 else 1.0
        
        # Correlation analysis
        correlation_matrix_rank = 0
        if n_numeric_features > 1:
            corr_matrix = numeric_df.corr()
            correlation_matrix_rank = np.linalg.matrix_rank(corr_matrix.fillna(0))
        
        # Missing values
        missing_values_ratio = df.isnull().sum().sum() / (n_samples * n_features)
        
        # Outlier detection (simplified)
        outlier_ratio = 0.0
        if n_numeric_features > 0:
            for col in numeric_df.columns:
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = numeric_df[(numeric_df[col] < Q1 - 1.5 * IQR) | 
                                    (numeric_df[col] > Q3 + 1.5 * IQR)]
                outlier_ratio += len(outliers) / n_samples
            outlier_ratio /= n_numeric_features
        
        # Recommended preprocessing
        optimal_preprocessing = []
        if missing_values_ratio > 0.05:
            optimal_preprocessing.append("handle_missing_values")
        if outlier_ratio > 0.1:
            optimal_preprocessing.append("outlier_treatment")
        if n_numeric_features > 0 and numeric_df.std().max() / numeric_df.std().min() > 10:
            optimal_preprocessing.append("scaling")
        if len(feature_types) != n_numeric_features:
            optimal_preprocessing.append("encode_categorical")
        
        # Recommended cluster range
        max_reasonable_k = min(int(np.sqrt(n_samples / 2)), 15)
        recommended_k_range = (2, max(3, max_reasonable_k))
        
        return ClusteringAnalysis(
            n_samples=n_samples,
            n_features=n_features,
            feature_types=feature_types,
            data_density=data_density,
            dimensionality_score=dimensionality_score,
            correlation_matrix_rank=correlation_matrix_rank,
            missing_values_ratio=missing_values_ratio,
            outlier_ratio=outlier_ratio,
            optimal_preprocessing=optimal_preprocessing,
            recommended_k_range=recommended_k_range
        )
    
    def _preprocess_clustering_data(self, df: pd.DataFrame, analysis: ClusteringAnalysis) -> Tuple[pd.DataFrame, List[str]]:
        """Preprocess data for clustering."""
        df_processed = df.copy()
        preprocessing_steps = []
        
        # Handle missing values
        if "handle_missing_values" in analysis.optimal_preprocessing:
            # Simple imputation for numeric columns
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_processed[col].isnull().any():
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
            
            # Mode imputation for categorical columns
            categorical_cols = df_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_processed[col].isnull().any():
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            
            preprocessing_steps.append("imputed_missing_values")
        
        # Handle categorical variables
        if self.handle_categorical and "encode_categorical" in analysis.optimal_preprocessing:
            categorical_cols = df_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col not in ['true_cluster', 'segment', 'topic', 'cell_type']:  # Skip label columns
                    # Simple one-hot encoding
                    dummies = pd.get_dummies(df_processed[col], prefix=col)
                    df_processed = pd.concat([df_processed, dummies], axis=1)
                    df_processed.drop(col, axis=1, inplace=True)
            preprocessing_steps.append("encoded_categorical_variables")
        
        # Remove non-numeric columns for clustering
        label_cols = ['true_cluster', 'segment', 'topic', 'cell_type']
        df_processed = df_processed.select_dtypes(include=[np.number])
        
        # Outlier treatment
        if self.remove_outliers and "outlier_treatment" in analysis.optimal_preprocessing:
            for col in df_processed.columns:
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                df_processed = df_processed[
                    (df_processed[col] >= Q1 - 1.5 * IQR) & 
                    (df_processed[col] <= Q3 + 1.5 * IQR)
                ]
            preprocessing_steps.append("removed_outliers")
        
        # Scaling
        if self.apply_scaling and "scaling" in analysis.optimal_preprocessing:
            if self.scaling_method == "standard":
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            
            df_processed[df_processed.columns] = self.scaler.fit_transform(df_processed)
            preprocessing_steps.append(f"applied_{self.scaling_method}_scaling")
        
        return df_processed, preprocessing_steps
    
    def _apply_dimensionality_reduction(self, df: pd.DataFrame, analysis: ClusteringAnalysis) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply dimensionality reduction techniques."""
        reduction_results = {}
        df_reduced = df.copy()
        
        # PCA for variance-based reduction
        if SKLEARN_AVAILABLE and df.shape[1] > 3:
            pca = PCA(n_components=min(df.shape[1], 20), random_state=self.random_state)
            pca_data = pca.fit_transform(df)
            
            # Find components that explain desired variance
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumvar >= self.pca_variance_threshold) + 1
            n_components = max(2, min(n_components, 10))
            
            # Apply PCA with selected components
            pca_final = PCA(n_components=n_components, random_state=self.random_state)
            df_reduced = pd.DataFrame(
                pca_final.fit_transform(df),
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            
            reduction_results["pca"] = {
                "n_components": n_components,
                "explained_variance_ratio": pca_final.explained_variance_ratio_.tolist(),
                "cumulative_variance": float(cumvar[n_components-1])
            }
        
        return df_reduced, reduction_results
    
    def _determine_optimal_clusters(self, df: pd.DataFrame, analysis: ClusteringAnalysis) -> int:
        """Determine optimal number of clusters using multiple methods."""
        if not self.auto_select_k:
            return self.max_clusters
        
        min_k, max_k = analysis.recommended_k_range
        max_k = min(max_k, self.max_clusters)
        
        if not SKLEARN_AVAILABLE:
            return 3  # Default
        
        # Elbow method using K-means inertia
        inertias = []
        silhouette_scores = []
        k_range = range(min_k, max_k + 1)
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(df)
                
                inertias.append(kmeans.inertia_)
                
                if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
                    silhouette_avg = silhouette_score(df, labels)
                    silhouette_scores.append(silhouette_avg)
                else:
                    silhouette_scores.append(-1)
                    
            except Exception:
                inertias.append(float('inf'))
                silhouette_scores.append(-1)
        
        # Find elbow point (simplified)
        if inertias:
            # Calculate rate of change
            differences = np.diff(inertias)
            elbow_k = min_k + np.argmax(differences) if len(differences) > 0 else min_k
        else:
            elbow_k = min_k
        
        # Find best silhouette score
        if silhouette_scores:
            silhouette_k = min_k + np.argmax(silhouette_scores)
        else:
            silhouette_k = min_k
        
        # Combine both methods (weighted average)
        optimal_k = int(0.6 * silhouette_k + 0.4 * elbow_k)
        optimal_k = max(min_k, min(optimal_k, max_k))
        
        return optimal_k
    
    def _train_and_evaluate_models(self, df: pd.DataFrame, optimal_k: int, task_type: ClusteringTask, analysis: ClusteringAnalysis) -> List[ClusterPerformance]:
        """Train and evaluate multiple clustering models."""
        performances = []
        
        if not SKLEARN_AVAILABLE:
            return performances
        
        # Get available models
        models = self._get_clustering_models(task_type, optimal_k, df.shape)
        
        for model_name, model_info in models.items():
            try:
                self.logger.info(f"Training {model_name}...")
                
                performance = self._train_single_clustering_model(
                    model_info, df, model_name, task_type
                )
                
                if performance:
                    performances.append(performance)
                    self.logger.info(f"{model_name} - Silhouette: {performance.silhouette_score:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        return performances
    
    def _get_clustering_models(self, task_type: ClusteringTask, optimal_k: int, data_shape: Tuple[int, int]) -> Dict[str, Dict[str, Any]]:
        """Get available clustering models."""
        models = {}
        n_samples, n_features = data_shape
        
        # K-Means (always available)
        models["K-Means"] = {
            "type": "kmeans",
            "n_clusters": optimal_k,
            "random_state": self.random_state
        }
        
        # DBSCAN (density-based)
        if not self.quick_mode:
            models["DBSCAN"] = {
                "type": "dbscan",
                "eps": 0.5,  # Will be auto-tuned
                "min_samples": max(2, int(n_samples * 0.01))
            }
        
        # Hierarchical clustering
        if n_samples < 1000:  # Computationally expensive for large datasets
            models["Hierarchical"] = {
                "type": "hierarchical",
                "n_clusters": optimal_k,
                "linkage": "ward"
            }
        
        # Gaussian Mixture Model
        if not self.quick_mode:
            models["Gaussian Mixture"] = {
                "type": "gaussian_mixture",
                "n_components": optimal_k,
                "random_state": self.random_state
            }
        
        # Spectral clustering (for non-convex clusters)
        if n_samples < 500 and not self.quick_mode:
            models["Spectral"] = {
                "type": "spectral",
                "n_clusters": optimal_k,
                "random_state": self.random_state
            }
        
        return models
    
    def _train_single_clustering_model(self, model_info: Dict[str, Any], df: pd.DataFrame, model_name: str, task_type: ClusteringTask) -> Optional[ClusterPerformance]:
        """Train a single clustering model."""
        start_time = time.time()
        
        try:
            model_type = model_info["type"]
            
            if model_type == "kmeans":
                model = KMeans(
                    n_clusters=model_info["n_clusters"],
                    random_state=model_info["random_state"],
                    n_init=10
                )
                labels = model.fit_predict(df)
                centers = model.cluster_centers_
                inertia = model.inertia_
                
            elif model_type == "dbscan":
                # Auto-tune eps using nearest neighbors
                if len(df) > 10:
                    k = min(5, len(df) - 1)
                    neighbors = NearestNeighbors(n_neighbors=k)
                    neighbors.fit(df)
                    distances, _ = neighbors.kneighbors(df)
                    eps = np.percentile(distances[:, -1], 80)
                else:
                    eps = 0.5
                
                model = DBSCAN(eps=eps, min_samples=model_info["min_samples"])
                labels = model.fit_predict(df)
                centers = None
                inertia = None
                
            elif model_type == "hierarchical":
                model = AgglomerativeClustering(
                    n_clusters=model_info["n_clusters"],
                    linkage=model_info["linkage"]
                )
                labels = model.fit_predict(df)
                centers = None
                inertia = None
                
            elif model_type == "gaussian_mixture":
                model = GaussianMixture(
                    n_components=model_info["n_components"],
                    random_state=model_info["random_state"]
                )
                model.fit(df)
                labels = model.predict(df)
                centers = model.means_
                inertia = None
                
            elif model_type == "spectral":
                model = SpectralClustering(
                    n_clusters=model_info["n_clusters"],
                    random_state=model_info["random_state"]
                )
                labels = model.fit_predict(df)
                centers = None
                inertia = None
                
            else:
                return None
            
            training_time = time.time() - start_time
            
            # Calculate performance metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise cluster
            
            if n_clusters < 2:
                return None
            
            # Silhouette score
            silhouette_avg = silhouette_score(df, labels) if n_clusters > 1 else -1
            
            # Calinski-Harabasz score
            ch_score = calinski_harabasz_score(df, labels) if n_clusters > 1 else 0
            
            # Davies-Bouldin score
            db_score = davies_bouldin_score(df, labels) if n_clusters > 1 else float('inf')
            
            # Cluster sizes
            unique, counts = np.unique(labels, return_counts=True)
            cluster_sizes = counts[unique != -1].tolist()  # Exclude noise cluster
            
            # Cluster density (simplified)
            cluster_density = [float(size / len(labels)) for size in cluster_sizes]
            
            # Scalability score (based on training time and data size)
            scalability_score = max(0, 1.0 - (training_time / (len(df) / 1000)))
            
            # Interpretability score (simpler models are more interpretable)
            interpretability_scores = {
                "kmeans": 0.9,
                "hierarchical": 0.8,
                "dbscan": 0.7,
                "gaussian_mixture": 0.6,
                "spectral": 0.5
            }
            interpretability_score = interpretability_scores.get(model_type, 0.5)
            
            return ClusterPerformance(
                algorithm=model_name,
                n_clusters=n_clusters,
                silhouette_score=silhouette_avg,
                calinski_harabasz_score=ch_score,
                davies_bouldin_score=db_score,
                inertia=inertia,
                cluster_sizes=cluster_sizes,
                cluster_density=cluster_density,
                training_time=training_time,
                scalability_score=scalability_score,
                interpretability_score=interpretability_score
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to train {model_name}: {str(e)}")
            return None
    
    def _select_best_model(self, performances: List[ClusterPerformance]) -> Dict[str, Any]:
        """Select best performing clustering model."""
        if not performances:
            raise ValueError("No models were successfully trained")
        
        # Score models based on multiple criteria
        def score_model(perf: ClusterPerformance) -> float:
            silhouette_weight = 0.4
            db_weight = 0.3  # Lower is better
            interpretability_weight = 0.2
            scalability_weight = 0.1
            
            silhouette_score = max(0, perf.silhouette_score + 1) / 2  # Normalize to 0-1
            db_score = max(0, 1.0 / (1 + perf.davies_bouldin_score))  # Lower DB is better
            interpretability_score = perf.interpretability_score
            scalability_score = perf.scalability_score
            
            return (silhouette_weight * silhouette_score +
                    db_weight * db_score +
                    interpretability_weight * interpretability_score +
                    scalability_weight * scalability_score)
        
        best_performance = max(performances, key=score_model)
        
        return {
            "performance": best_performance,
            "algorithm_name": best_performance.algorithm
        }
    
    def _final_clustering_analysis(self, best_model_info: Dict[str, Any], df_processed: pd.DataFrame, df_original: pd.DataFrame, task_type: ClusteringTask, analysis: ClusteringAnalysis, dim_reduction_result: Optional[Dict[str, Any]], preprocessing_steps: List[str]) -> ClusterResult:
        """Perform final clustering analysis."""
        best_performance = best_model_info["performance"]
        
        # Re-train best model to get cluster assignments
        model_type = best_performance.algorithm.lower().replace(" ", "_").replace("-", "_")
        
        if "k_means" in model_type or "kmeans" in model_type:
            model = KMeans(n_clusters=best_performance.n_clusters, random_state=self.random_state)
            model.fit(df_processed)
            cluster_labels = model.labels_.tolist()
            cluster_centers = model.cluster_centers_.tolist()
            
        elif "dbscan" in model_type:
            # Re-estimate DBSCAN parameters
            if len(df_processed) > 10:
                k = min(5, len(df_processed) - 1)
                neighbors = NearestNeighbors(n_neighbors=k)
                neighbors.fit(df_processed)
                distances, _ = neighbors.kneighbors(df_processed)
                eps = np.percentile(distances[:, -1], 80)
            else:
                eps = 0.5
            
            model = DBSCAN(eps=eps, min_samples=max(2, int(len(df_processed) * 0.01)))
            cluster_labels = model.fit_predict(df_processed).tolist()
            cluster_centers = None
            
        elif "hierarchical" in model_type:
            model = AgglomerativeClustering(n_clusters=best_performance.n_clusters, linkage="ward")
            cluster_labels = model.fit_predict(df_processed).tolist()
            cluster_centers = None
            
        else:
            # Default to K-means
            model = KMeans(n_clusters=best_performance.n_clusters, random_state=self.random_state)
            model.fit(df_processed)
            cluster_labels = model.labels_.tolist()
            cluster_centers = model.cluster_centers_.tolist()
        
        return ClusterResult(
            task_type=task_type.value,
            best_algorithm=best_performance.algorithm,
            best_model=model,
            performance_metrics=best_performance,
            all_model_performances=[best_performance],  # Simplified
            clustering_analysis=analysis,
            cluster_labels=cluster_labels,
            cluster_centers=cluster_centers,
            cluster_profiles={},  # Will be filled later
            dimensionality_reduction=dim_reduction_result,
            visualization_data=None,  # Will be filled later
            preprocessing_steps=preprocessing_steps
        )
    
    def _generate_cluster_profiles(self, df: pd.DataFrame, cluster_labels: List[int]) -> Dict[str, Any]:
        """Generate cluster profiles and characteristics."""
        profiles = {}
        
        # Add cluster labels to original data
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # Generate profile for each cluster
        unique_clusters = sorted(set(cluster_labels))
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise cluster in DBSCAN
                continue
                
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            profile = {
                "cluster_id": cluster_id,
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(df_with_clusters) * 100
            }
            
            # Numeric feature statistics
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'cluster']
            
            for col in numeric_cols:
                profile[f"{col}_mean"] = float(cluster_data[col].mean())
                profile[f"{col}_std"] = float(cluster_data[col].std())
                profile[f"{col}_median"] = float(cluster_data[col].median())
            
            # Categorical feature distributions
            categorical_cols = cluster_data.select_dtypes(include=['object']).columns
            categorical_cols = [col for col in categorical_cols if col not in ['cluster', 'true_cluster', 'segment', 'topic', 'cell_type']]
            
            for col in categorical_cols:
                value_counts = cluster_data[col].value_counts()
                profile[f"{col}_distribution"] = value_counts.to_dict()
            
            profiles[f"cluster_{cluster_id}"] = profile
        
        return profiles
    
    def _create_visualizations(self, df: pd.DataFrame, cluster_labels: List[int], dim_reduction_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create visualization data for clusters."""
        visualization_data = {}
        
        # 2D scatter plot data
        if df.shape[1] >= 2:
            # Use first two principal components or first two features
            if dim_reduction_result and "pca" in dim_reduction_result:
                x_data = df.iloc[:, 0].tolist()
                y_data = df.iloc[:, 1].tolist()
                x_label = "PC1"
                y_label = "PC2"
            else:
                x_data = df.iloc[:, 0].tolist()
                y_data = df.iloc[:, 1].tolist()
                x_label = df.columns[0]
                y_label = df.columns[1]
            
            visualization_data["scatter_2d"] = {
                "x_data": x_data,
                "y_data": y_data,
                "cluster_labels": cluster_labels,
                "x_label": x_label,
                "y_label": y_label
            }
        
        # Cluster size distribution
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = {}
        for cluster_id, count in zip(unique_clusters, counts):
            if cluster_id != -1:  # Exclude noise cluster
                cluster_sizes[f"Cluster {cluster_id}"] = int(count)
        
        visualization_data["cluster_sizes"] = cluster_sizes
        
        # Feature importance for clustering (if PCA was used)
        if dim_reduction_result and "pca" in dim_reduction_result:
            visualization_data["pca_variance"] = {
                "components": [f"PC{i+1}" for i in range(len(dim_reduction_result["pca"]["explained_variance_ratio"]))],
                "variance_explained": dim_reduction_result["pca"]["explained_variance_ratio"]
            }
        
        return visualization_data
    
    def _generate_recommendations(self, results: ClusterResult, analysis: ClusteringAnalysis) -> List[str]:
        """Generate recommendations based on clustering results."""
        recommendations = []
        
        # Performance recommendations
        if results.performance_metrics.silhouette_score > 0.5:
            recommendations.append("Excellent clustering quality - clear cluster separation detected")
        elif results.performance_metrics.silhouette_score > 0.3:
            recommendations.append("Good clustering quality - consider validation with domain experts")
        else:
            recommendations.append("Low clustering quality - consider different algorithms or feature engineering")
        
        # Cluster count recommendations
        if results.performance_metrics.n_clusters > 8:
            recommendations.append("Many clusters detected - consider hierarchical interpretation or cluster merging")
        elif results.performance_metrics.n_clusters < 3:
            recommendations.append("Few clusters found - data might have simple structure or need different approach")
        
        # Data characteristics recommendations
        if analysis.dimensionality_score < 0.5:
            recommendations.append("High dimensionality detected - consider feature selection or dimensionality reduction")
        
        if analysis.missing_values_ratio > 0.1:
            recommendations.append("High missing value ratio - improve data collection or imputation methods")
        
        # Algorithm-specific recommendations
        if "K-Means" in results.best_algorithm:
            recommendations.append("K-Means selected - assumes spherical clusters, validate with business context")
        elif "DBSCAN" in results.best_algorithm:
            recommendations.append("DBSCAN selected - good for irregular shapes, check for noise points")
        elif "Hierarchical" in results.best_algorithm:
            recommendations.append("Hierarchical clustering selected - consider dendrogram analysis for cluster relationships")
        
        # Business recommendations
        if "customer" in results.task_type or "segment" in results.task_type:
            recommendations.append("Customer segmentation complete - develop targeted marketing strategies for each segment")
        
        if results.performance_metrics.interpretability_score < 0.6:
            recommendations.append("Complex model selected - create simplified explanations for stakeholders")
        
        return recommendations
    
    def _share_clustering_insights(self, result_data: Dict[str, Any]) -> None:
        """Share clustering insights with other agents."""
        # Share cluster structure insights
        self.share_knowledge(
            knowledge_type="clustering_analysis_results",
            knowledge_data={
                "task_type": result_data["task_type"],
                "n_clusters": result_data["clustering_results"]["performance_metrics"]["n_clusters"],
                "clustering_quality": result_data["clustering_results"]["performance_metrics"]["silhouette_score"],
                "cluster_profiles": result_data["clustering_results"]["cluster_profiles"]
            }
        )
        
        # Share data insights
        self.share_knowledge(
            knowledge_type="unsupervised_data_insights",
            knowledge_data={
                "data_structure": result_data["clustering_analysis"],
                "preprocessing_effectiveness": result_data["preprocessing_steps"],
                "dimensionality_insights": result_data["dimensionality_reduction"]
            }
        )
    
    def _results_to_dict(self, results: ClusterResult) -> Dict[str, Any]:
        """Convert ClusterResult to dictionary."""
        return {
            "task_type": results.task_type,
            "best_algorithm": results.best_algorithm,
            "performance_metrics": self._performance_to_dict(results.performance_metrics),
            "cluster_labels": results.cluster_labels,
            "cluster_centers": results.cluster_centers,
            "cluster_profiles": results.cluster_profiles,
            "dimensionality_reduction": results.dimensionality_reduction,
            "visualization_data": results.visualization_data,
            "preprocessing_steps": results.preprocessing_steps
        }
    
    def _performance_to_dict(self, performance: ClusterPerformance) -> Dict[str, Any]:
        """Convert ClusterPerformance to dictionary."""
        return {
            "algorithm": performance.algorithm,
            "n_clusters": performance.n_clusters,
            "silhouette_score": performance.silhouette_score,
            "calinski_harabasz_score": performance.calinski_harabasz_score,
            "davies_bouldin_score": performance.davies_bouldin_score,
            "inertia": performance.inertia,
            "cluster_sizes": performance.cluster_sizes,
            "cluster_density": performance.cluster_density,
            "training_time": performance.training_time,
            "scalability_score": performance.scalability_score,
            "interpretability_score": performance.interpretability_score
        }
    
    def _clustering_analysis_to_dict(self, analysis: ClusteringAnalysis) -> Dict[str, Any]:
        """Convert ClusteringAnalysis to dictionary."""
        return {
            "n_samples": analysis.n_samples,
            "n_features": analysis.n_features,
            "feature_types": analysis.feature_types,
            "data_density": analysis.data_density,
            "dimensionality_score": analysis.dimensionality_score,
            "correlation_matrix_rank": analysis.correlation_matrix_rank,
            "missing_values_ratio": analysis.missing_values_ratio,
            "outlier_ratio": analysis.outlier_ratio,
            "optimal_preprocessing": analysis.optimal_preprocessing,
            "recommended_k_range": analysis.recommended_k_range
        }
    
    def can_handle_task(self, context: TaskContext) -> bool:
        """Check if this is a clustering task."""
        user_input = context.user_input.lower()
        clustering_keywords = [
            "cluster", "clustering", "segment", "segmentation", "group", "grouping",
            "unsupervised", "pattern", "discover", "customer segment", "market segment",
            "anomaly detection", "outlier detection", "dimensionality reduction", "pca"
        ]
        
        return any(keyword in user_input for keyword in clustering_keywords)
    
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """Estimate clustering task complexity."""
        user_input = context.user_input.lower()
        
        # Expert level tasks
        if any(keyword in user_input for keyword in ["spectral", "density", "hierarchical", "gaussian mixture"]):
            return TaskComplexity.EXPERT
        elif any(keyword in user_input for keyword in ["anomaly", "outlier", "high dimensional"]):
            return TaskComplexity.COMPLEX
        elif any(keyword in user_input for keyword in ["customer", "segment", "market"]):
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _create_refinement_plan(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Create clustering specific refinement plan."""
        return {
            "strategy_name": "advanced_clustering_optimization",
            "steps": [
                "enhanced_feature_engineering",
                "advanced_dimensionality_reduction",
                "ensemble_clustering_methods",
                "cluster_stability_analysis"
            ],
            "estimated_improvement": 0.12,
            "execution_time": 8.0
        }
    
    def _assess_knowledge_relevance(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> float:
        """Assess relevance of shared knowledge to clustering agent."""
        relevance_map = {
            "clustering_analysis_results": 0.9,
            "unsupervised_data_insights": 0.8,
            "data_quality_issues": 0.7,
            "feature_importance": 0.6,
            "preprocessing_effectiveness": 0.7
        }
        return relevance_map.get(knowledge_type, 0.1)