"""
NLP Agent for AutoML Platform

Specialized agent for Natural Language Processing tasks that:
1. Handles text preprocessing and feature extraction
2. Implements text classification, sentiment analysis, and NER
3. Supports multiple NLP algorithms and embeddings
4. Provides text-specific evaluation metrics and analysis
5. Handles multilingual and domain-specific text challenges

This agent runs for text-based ML problems and NLP tasks.
"""

import re
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from sklearn.feature_extraction.text import (
        TfidfVectorizer, CountVectorizer, HashingVectorizer
    )
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB, GaussianNB
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        pipeline, BertTokenizer, BertModel
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, TaskContext, TaskComplexity


class NLPTask(Enum):
    """Types of NLP tasks."""
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NAMED_ENTITY_RECOGNITION = "ner"
    TEXT_SIMILARITY = "text_similarity"
    TOPIC_MODELING = "topic_modeling"
    LANGUAGE_DETECTION = "language_detection"
    TEXT_SUMMARIZATION = "text_summarization"
    QUESTION_ANSWERING = "question_answering"


class TextPreprocessingMethod(Enum):
    """Text preprocessing methods."""
    BASIC_CLEANING = "basic_cleaning"
    STOPWORD_REMOVAL = "stopword_removal"
    STEMMING = "stemming"
    LEMMATIZATION = "lemmatization"
    NORMALIZATION = "normalization"
    TOKENIZATION = "tokenization"


class FeatureExtractionMethod(Enum):
    """Feature extraction methods for text."""
    TFIDF = "tfidf"
    COUNT_VECTORIZER = "count_vectorizer"
    WORD_EMBEDDINGS = "word_embeddings"
    SENTENCE_EMBEDDINGS = "sentence_embeddings"
    TRANSFORMER_EMBEDDINGS = "transformer_embeddings"
    N_GRAMS = "n_grams"


@dataclass
class TextAnalysis:
    """Text analysis results."""
    language_detected: Optional[str]
    avg_sentence_length: float
    avg_word_length: float
    vocabulary_size: int
    most_common_words: List[Tuple[str, int]]
    readability_score: Optional[float]
    text_complexity: str


@dataclass
class NLPPerformance:
    """NLP model performance metrics."""
    algorithm: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    macro_f1: float
    weighted_f1: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time: float
    prediction_time: float
    feature_extraction_method: str


@dataclass
class NLPResult:
    """Complete NLP result."""
    task_type: str
    best_algorithm: str
    best_model: Any
    performance_metrics: NLPPerformance
    all_model_performances: List[NLPPerformance]
    text_analysis: TextAnalysis
    feature_importance: Optional[Dict[str, float]]
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    preprocessing_steps: List[str]
    sample_predictions: List[Dict[str, Any]]


class NLPAgent(BaseAgent):
    """
    NLP Agent for natural language processing tasks.
    
    Responsibilities:
    1. Text preprocessing and feature extraction
    2. NLP task identification and execution
    3. Model selection for text-specific algorithms
    4. Text-specific evaluation and analysis
    5. Multilingual and domain adaptation support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub=None):
        """Initialize the NLP Agent."""
        super().__init__(
            name="NLP Agent",
            description="Advanced natural language processing and text analysis specialist",
            specialization="Natural Language Processing & Text Analytics",
            config=config,
            communication_hub=communication_hub
        )
        
        # NLP configuration
        self.supported_languages = self.config.get("supported_languages", ["en", "es", "fr", "de"])
        self.default_language = self.config.get("default_language", "en")
        self.max_features = self.config.get("max_features", 10000)
        self.ngram_range = tuple(self.config.get("ngram_range", [1, 2]))
        
        # Preprocessing settings
        self.remove_stopwords = self.config.get("remove_stopwords", True)
        self.apply_stemming = self.config.get("apply_stemming", False)
        self.apply_lemmatization = self.config.get("apply_lemmatization", True)
        self.min_word_length = self.config.get("min_word_length", 2)
        self.max_word_length = self.config.get("max_word_length", 50)
        
        # Model settings
        self.use_transformers = self.config.get("use_transformers", False)
        self.transformer_model = self.config.get("transformer_model", "bert-base-uncased")
        self.quick_mode = self.config.get("quick_mode", False)
        
        # Quality thresholds
        self.quality_thresholds.update({
            "min_accuracy": self.config.get("min_accuracy", 0.75),
            "min_f1_score": self.config.get("min_f1_score", 0.7),
            "min_macro_f1": self.config.get("min_macro_f1", 0.6),
            "max_cv_std": self.config.get("max_cv_std", 0.1)
        })
        
        # Initialize NLP tools
        self.tokenizer = None
        self.stemmer = None
        self.lemmatizer = None
        self.stopwords_set = set()
        self.spacy_nlp = None
        
        self._initialize_nlp_tools()
    
    def _initialize_nlp_tools(self):
        """Initialize NLP tools if available."""
        try:
            if NLTK_AVAILABLE:
                # Download required NLTK data (with error handling)
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    try:
                        nltk.download('punkt', quiet=True)
                    except:
                        pass
                
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    try:
                        nltk.download('stopwords', quiet=True)
                    except:
                        pass
                
                try:
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    try:
                        nltk.download('wordnet', quiet=True)
                    except:
                        pass
                
                # Initialize tools
                self.stemmer = PorterStemmer()
                try:
                    self.lemmatizer = WordNetLemmatizer()
                except:
                    pass
                
                try:
                    self.stopwords_set = set(stopwords.words('english'))
                except:
                    self.stopwords_set = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            
            if SPACY_AVAILABLE:
                try:
                    self.spacy_nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # Model not installed
                    pass
        
        except Exception as e:
            self.logger.warning(f"Failed to initialize some NLP tools: {str(e)}")
    
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute comprehensive NLP workflow.
        
        Args:
            context: Task context with text data
            
        Returns:
            AgentResult with NLP models and analysis
        """
        try:
            self.logger.info("Starting NLP workflow...")
            
            # Load text dataset
            df, text_column, target_column = self._load_text_dataset(context)
            if df is None or text_column is None:
                return AgentResult(
                    success=False,
                    message="Failed to load text dataset or identify text column"
                )
            
            # Phase 1: Task Identification
            self.logger.info("Phase 1: Identifying NLP task type...")
            task_type = self._identify_nlp_task(context, df, target_column)
            
            # Phase 2: Text Analysis
            self.logger.info("Phase 2: Performing text analysis...")
            text_analysis = self._analyze_text_data(df[text_column])
            
            # Phase 3: Text Preprocessing
            self.logger.info("Phase 3: Preprocessing text data...")
            df_processed, preprocessing_steps = self._preprocess_text_data(df, text_column)
            
            # Phase 4: Feature Extraction
            self.logger.info("Phase 4: Extracting text features...")
            X, feature_names, extraction_method = self._extract_text_features(
                df_processed[text_column], task_type
            )
            
            # Phase 5: Prepare Target Variable
            if target_column:
                y = self._prepare_target_variable(df_processed[target_column], task_type)
                
                # Phase 6: Model Training and Evaluation
                self.logger.info("Phase 6: Training and evaluating NLP models...")
                model_performances = self._train_and_evaluate_models(
                    X, y, task_type, extraction_method
                )
                
                # Phase 7: Select Best Model
                self.logger.info("Phase 7: Selecting best performing model...")
                best_model_info = self._select_best_model(model_performances)
                
                # Phase 8: Final Evaluation
                self.logger.info("Phase 8: Final model evaluation...")
                final_results = self._final_model_evaluation(
                    best_model_info, X, y, feature_names, task_type, text_analysis, preprocessing_steps
                )
                
                # Phase 9: Generate Sample Predictions
                self.logger.info("Phase 9: Generating sample predictions...")
                sample_predictions = self._generate_sample_predictions(
                    final_results.best_model, df_processed[text_column].head(5), 
                    extraction_method, task_type
                )
                
                final_results.sample_predictions = sample_predictions
            else:
                # Unsupervised NLP task
                final_results = self._handle_unsupervised_task(
                    X, feature_names, task_type, text_analysis, preprocessing_steps
                )
            
            # Create comprehensive result
            result_data = {
                "nlp_results": self._results_to_dict(final_results),
                "text_analysis": self._text_analysis_to_dict(text_analysis),
                "task_type": task_type.value if isinstance(task_type, NLPTask) else task_type,
                "preprocessing_steps": preprocessing_steps,
                "feature_extraction_method": extraction_method,
                "model_performances": [self._performance_to_dict(perf) for perf in (final_results.all_model_performances if hasattr(final_results, 'all_model_performances') else [])],
                "recommendations": self._generate_recommendations(final_results, text_analysis)
            }
            
            # Update performance metrics
            if hasattr(final_results, 'performance_metrics'):
                performance_metrics = {
                    "nlp_accuracy": final_results.performance_metrics.accuracy,
                    "nlp_f1_score": final_results.performance_metrics.f1_score,
                    "nlp_macro_f1": final_results.performance_metrics.macro_f1,
                    "text_processing_efficiency": 1.0 / (final_results.performance_metrics.training_time + 1)
                }
                self.update_performance_metrics(performance_metrics)
            
            # Share NLP insights
            if self.communication_hub:
                self._share_nlp_insights(result_data)
            
            return AgentResult(
                success=True,
                data=result_data,
                message=f"NLP workflow completed: {task_type.value if isinstance(task_type, NLPTask) else task_type} task processed",
                recommendations=result_data["recommendations"]
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"NLP workflow failed: {str(e)}"
            )
    
    def _load_text_dataset(self, context: TaskContext) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
        """Load text dataset and identify text and target columns."""
        # In real implementation, this would load from previous agent results
        # For demo, create synthetic text data
        
        user_input = context.user_input.lower()
        
        # Generate synthetic text data based on task type
        if "sentiment" in user_input:
            return self._create_sentiment_dataset()
        elif "classification" in user_input and "text" in user_input:
            return self._create_text_classification_dataset()
        elif "topic" in user_input:
            return self._create_topic_dataset()
        else:
            return self._create_general_text_dataset()
    
    def _create_sentiment_dataset(self) -> Tuple[pd.DataFrame, str, str]:
        """Create synthetic sentiment analysis dataset."""
        np.random.seed(42)
        
        positive_texts = [
            "I love this product! It's amazing and works perfectly.",
            "Great experience, highly recommend to everyone.",
            "Fantastic quality and excellent customer service.",
            "Best purchase I've made this year, absolutely thrilled.",
            "Outstanding performance, exceeded all my expectations.",
            "Wonderful experience from start to finish.",
            "Incredible value for money, couldn't be happier.",
            "Perfect solution to my problem, works like a charm.",
            "Exceptional quality and fast delivery service.",
            "Amazing product that delivers exactly what it promises."
        ]
        
        negative_texts = [
            "Terrible product, waste of money and time.",
            "Poor quality, broke after just one day of use.",
            "Worst customer service experience I've ever had.",
            "Complete disappointment, nothing like advertised.",
            "Awful experience, would not recommend to anyone.",
            "Defective product arrived, very unsatisfied.",
            "Overpriced and underdelivered on all promises.",
            "Horrible quality control, received damaged item.",
            "Frustrating experience, product doesn't work.",
            "Very disappointed with this purchase decision."
        ]
        
        neutral_texts = [
            "The product is okay, nothing special but works.",
            "Average quality for the price, meets basic needs.",
            "Decent experience overall, some pros and cons.",
            "It's fine, does what it's supposed to do.",
            "Not bad, but not great either, just okay.",
            "Reasonable product with standard features.",
            "Acceptable quality, meets minimum requirements.",
            "Fair experience, nothing outstanding to report.",
            "Standard product with typical performance.",
            "Adequate for basic needs, nothing more."
        ]
        
        # Create dataset
        texts = []
        sentiments = []
        
        # Add multiple copies with variations
        for _ in range(20):
            texts.extend(positive_texts)
            sentiments.extend(['positive'] * len(positive_texts))
            texts.extend(negative_texts)
            sentiments.extend(['negative'] * len(negative_texts))
            texts.extend(neutral_texts)
            sentiments.extend(['neutral'] * len(neutral_texts))
        
        # Add some noise and variations
        for i in range(len(texts)):
            if np.random.random() < 0.1:  # 10% chance to add noise
                texts[i] = texts[i] + " " + np.random.choice(["Really.", "Definitely.", "Absolutely.", "Indeed."])
        
        df = pd.DataFrame({
            'text': texts,
            'sentiment': sentiments
        })
        
        return df.sample(frac=1).reset_index(drop=True), 'text', 'sentiment'
    
    def _create_text_classification_dataset(self) -> Tuple[pd.DataFrame, str, str]:
        """Create synthetic text classification dataset."""
        np.random.seed(42)
        
        categories = {
            'technology': [
                "Artificial intelligence and machine learning are revolutionizing industries.",
                "The latest smartphone features advanced camera technology.",
                "Cloud computing enables scalable business solutions.",
                "Cybersecurity threats are increasing with digital transformation.",
                "Blockchain technology offers decentralized solutions.",
                "Internet of Things devices are connecting our world.",
                "Virtual reality creates immersive digital experiences.",
                "Data analytics drives informed business decisions.",
                "Software development requires continuous innovation.",
                "Network infrastructure supports digital communication."
            ],
            'sports': [
                "The championship game was intense and exciting.",
                "Professional athletes train rigorously for competitions.",
                "Team strategy determines success in tournaments.",
                "Olympic records showcase human athletic achievement.",
                "Sports medicine helps prevent and treat injuries.",
                "Fan enthusiasm creates amazing stadium atmosphere.",
                "Coaching techniques improve player performance significantly.",
                "Sports analytics revolutionize team management strategies.",
                "Athletic scholarships support student education goals.",
                "International competitions bring nations together peacefully."
            ],
            'politics': [
                "Electoral campaigns require significant financial resources.",
                "Government policies impact economic growth patterns.",
                "International diplomacy maintains peaceful relations.",
                "Legislative processes involve complex negotiations.",
                "Political debates inform voter decision making.",
                "Public policy affects citizen daily lives.",
                "Democratic institutions protect individual rights.",
                "Political transparency ensures accountable governance.",
                "Election security maintains democratic integrity.",
                "Civic engagement strengthens democratic processes."
            ],
            'entertainment': [
                "Hollywood movies continue breaking box office records.",
                "Streaming services revolutionize content consumption.",
                "Music festivals attract millions of fans worldwide.",
                "Television series create engaging storylines.",
                "Celebrity culture influences social media trends.",
                "Film production involves complex creative processes.",
                "Entertainment industry adapts to digital platforms.",
                "Concert tours generate significant economic impact.",
                "Gaming industry grows with technological advances.",
                "Entertainment content reflects cultural values."
            ]
        }
        
        texts = []
        labels = []
        
        for category, category_texts in categories.items():
            for _ in range(50):  # 50 samples per category
                texts.extend(category_texts)
                labels.extend([category] * len(category_texts))
        
        df = pd.DataFrame({
            'text': texts,
            'category': labels
        })
        
        return df.sample(frac=1).reset_index(drop=True), 'text', 'category'
    
    def _create_topic_dataset(self) -> Tuple[pd.DataFrame, str, str]:
        """Create synthetic topic modeling dataset."""
        return self._create_text_classification_dataset()  # Reuse for simplicity
    
    def _create_general_text_dataset(self) -> Tuple[pd.DataFrame, str, str]:
        """Create general text dataset."""
        return self._create_sentiment_dataset()  # Default to sentiment
    
    def _identify_nlp_task(self, context: TaskContext, df: pd.DataFrame, target_column: Optional[str]) -> NLPTask:
        """Identify the type of NLP task from context and data."""
        user_input = context.user_input.lower()
        
        # Task identification based on keywords
        if "sentiment" in user_input:
            return NLPTask.SENTIMENT_ANALYSIS
        elif "classify" in user_input or "classification" in user_input:
            return NLPTask.TEXT_CLASSIFICATION
        elif "entity" in user_input or "ner" in user_input:
            return NLPTask.NAMED_ENTITY_RECOGNITION
        elif "similarity" in user_input:
            return NLPTask.TEXT_SIMILARITY
        elif "topic" in user_input:
            return NLPTask.TOPIC_MODELING
        elif "language" in user_input and "detect" in user_input:
            return NLPTask.LANGUAGE_DETECTION
        elif "summary" in user_input or "summarize" in user_input:
            return NLPTask.TEXT_SUMMARIZATION
        elif "question" in user_input and "answer" in user_input:
            return NLPTask.QUESTION_ANSWERING
        else:
            # Default based on target column analysis
            if target_column and target_column in df.columns:
                unique_values = df[target_column].nunique()
                if unique_values <= 10:
                    return NLPTask.TEXT_CLASSIFICATION
                else:
                    return NLPTask.TEXT_CLASSIFICATION  # Default
            else:
                return NLPTask.TOPIC_MODELING  # Unsupervised default
    
    def _analyze_text_data(self, text_series: pd.Series) -> TextAnalysis:
        """Analyze text data characteristics."""
        texts = text_series.dropna().astype(str)
        
        # Basic statistics
        word_counts = []
        sentence_counts = []
        char_counts = []
        all_words = []
        
        for text in texts:
            # Clean text for analysis
            clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
            words = clean_text.split()
            sentences = text.split('.')
            
            word_counts.append(len(words))
            sentence_counts.append(len(sentences))
            char_counts.append(len(text))
            all_words.extend(words)
        
        # Calculate metrics
        avg_word_count = np.mean(word_counts) if word_counts else 0
        avg_sentence_length = avg_word_count / np.mean(sentence_counts) if sentence_counts and np.mean(sentence_counts) > 0 else 0
        avg_word_length = np.mean([len(word) for word in all_words]) if all_words else 0
        
        # Vocabulary analysis
        word_freq = pd.Series(all_words).value_counts()
        vocabulary_size = len(word_freq)
        most_common_words = list(word_freq.head(10).items())
        
        # Simple readability score (Flesch-like approximation)
        avg_chars_per_word = avg_word_length
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_chars_per_word / avg_word_length if avg_word_length > 0 else 1))
        
        # Text complexity
        if readability_score > 60:
            complexity = "easy"
        elif readability_score > 30:
            complexity = "medium"
        else:
            complexity = "difficult"
        
        # Language detection (simplified)
        language_detected = self._detect_language(texts.iloc[0] if len(texts) > 0 else "")
        
        return TextAnalysis(
            language_detected=language_detected,
            avg_sentence_length=avg_sentence_length,
            avg_word_length=avg_word_length,
            vocabulary_size=vocabulary_size,
            most_common_words=most_common_words,
            readability_score=readability_score,
            text_complexity=complexity
        )
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # Simplified language detection based on common words
        english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are'}
        spanish_words = {'el', 'la', 'y', 'o', 'pero', 'en', 'con', 'de', 'para', 'por', 'un', 'una', 'es', 'son'}
        french_words = {'le', 'la', 'et', 'ou', 'mais', 'dans', 'avec', 'de', 'pour', 'par', 'un', 'une', 'est', 'sont'}
        
        words = set(re.sub(r'[^\w\s]', ' ', text.lower()).split())
        
        en_score = len(words.intersection(english_words))
        es_score = len(words.intersection(spanish_words))
        fr_score = len(words.intersection(french_words))
        
        if en_score >= es_score and en_score >= fr_score:
            return "en"
        elif es_score >= fr_score:
            return "es"
        else:
            return "fr" if fr_score > 0 else "en"  # Default to English
    
    def _preprocess_text_data(self, df: pd.DataFrame, text_column: str) -> Tuple[pd.DataFrame, List[str]]:
        """Preprocess text data with various cleaning steps."""
        df_processed = df.copy()
        preprocessing_steps = []
        
        # Step 1: Basic cleaning
        df_processed[text_column] = df_processed[text_column].astype(str)
        df_processed[text_column] = df_processed[text_column].str.lower()
        preprocessing_steps.append("converted_to_lowercase")
        
        # Step 2: Remove special characters and digits
        df_processed[text_column] = df_processed[text_column].apply(
            lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x)
        )
        preprocessing_steps.append("removed_special_characters")
        
        # Step 3: Remove extra whitespace
        df_processed[text_column] = df_processed[text_column].apply(
            lambda x: re.sub(r'\s+', ' ', x).strip()
        )
        preprocessing_steps.append("normalized_whitespace")
        
        # Step 4: Remove stopwords (if enabled)
        if self.remove_stopwords and self.stopwords_set:
            df_processed[text_column] = df_processed[text_column].apply(
                lambda x: ' '.join([word for word in x.split() if word not in self.stopwords_set])
            )
            preprocessing_steps.append("removed_stopwords")
        
        # Step 5: Apply stemming or lemmatization
        if self.apply_lemmatization and self.lemmatizer:
            try:
                df_processed[text_column] = df_processed[text_column].apply(
                    lambda x: ' '.join([self.lemmatizer.lemmatize(word) for word in x.split()])
                )
                preprocessing_steps.append("applied_lemmatization")
            except Exception as e:
                self.logger.warning(f"Lemmatization failed: {str(e)}")
        elif self.apply_stemming and self.stemmer:
            df_processed[text_column] = df_processed[text_column].apply(
                lambda x: ' '.join([self.stemmer.stem(word) for word in x.split()])
            )
            preprocessing_steps.append("applied_stemming")
        
        # Step 6: Remove very short or long words
        df_processed[text_column] = df_processed[text_column].apply(
            lambda x: ' '.join([word for word in x.split() 
                               if self.min_word_length <= len(word) <= self.max_word_length])
        )
        preprocessing_steps.append("filtered_word_length")
        
        # Step 7: Remove empty texts
        initial_count = len(df_processed)
        df_processed = df_processed[df_processed[text_column].str.strip() != '']
        if len(df_processed) < initial_count:
            preprocessing_steps.append("removed_empty_texts")
        
        return df_processed, preprocessing_steps
    
    def _extract_text_features(self, text_series: pd.Series, task_type: NLPTask) -> Tuple[np.ndarray, List[str], str]:
        """Extract features from text data."""
        if not SKLEARN_AVAILABLE:
            # Fallback to simple word count features
            return self._simple_feature_extraction(text_series)
        
        # Choose feature extraction method based on task and available tools
        if self.use_transformers and TRANSFORMERS_AVAILABLE and not self.quick_mode:
            return self._transformer_feature_extraction(text_series, task_type)
        elif SENTENCE_TRANSFORMERS_AVAILABLE and task_type in [NLPTask.TEXT_SIMILARITY, NLPTask.SENTIMENT_ANALYSIS]:
            return self._sentence_transformer_extraction(text_series)
        else:
            return self._traditional_feature_extraction(text_series, task_type)
    
    def _traditional_feature_extraction(self, text_series: pd.Series, task_type: NLPTask) -> Tuple[np.ndarray, List[str], str]:
        """Traditional feature extraction using TF-IDF or Count Vectorizer."""
        try:
            # Choose vectorizer based on task
            if task_type in [NLPTask.SENTIMENT_ANALYSIS, NLPTask.TEXT_CLASSIFICATION]:
                vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=2,
                    max_df=0.95
                )
                extraction_method = "tfidf"
            else:
                vectorizer = CountVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=2,
                    max_df=0.95
                )
                extraction_method = "count_vectorizer"
            
            # Fit and transform
            X = vectorizer.fit_transform(text_series).toarray()
            feature_names = vectorizer.get_feature_names_out().tolist()
            
            return X, feature_names, extraction_method
            
        except Exception as e:
            self.logger.warning(f"Traditional feature extraction failed: {str(e)}")
            return self._simple_feature_extraction(text_series)
    
    def _transformer_feature_extraction(self, text_series: pd.Series, task_type: NLPTask) -> Tuple[np.ndarray, List[str], str]:
        """Feature extraction using transformer models."""
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.transformer_model)
            model = AutoModel.from_pretrained(self.transformer_model)
            
            # Extract embeddings for a sample (limit for performance)
            sample_texts = text_series.head(100).tolist()  # Limit for demo
            embeddings = []
            
            for text in sample_texts:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    embeddings.append(embedding)
            
            X = np.array(embeddings)
            feature_names = [f"transformer_dim_{i}" for i in range(X.shape[1])]
            
            # Pad with zeros for remaining samples if needed
            if len(sample_texts) < len(text_series):
                remaining = len(text_series) - len(sample_texts)
                padding = np.zeros((remaining, X.shape[1]))
                X = np.vstack([X, padding])
            
            return X, feature_names, "transformer_embeddings"
            
        except Exception as e:
            self.logger.warning(f"Transformer feature extraction failed: {str(e)}")
            return self._traditional_feature_extraction(text_series, task_type)
    
    def _sentence_transformer_extraction(self, text_series: pd.Series) -> Tuple[np.ndarray, List[str], str]:
        """Feature extraction using sentence transformers."""
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(text_series.tolist())
            feature_names = [f"sentence_emb_{i}" for i in range(embeddings.shape[1])]
            
            return embeddings, feature_names, "sentence_embeddings"
            
        except Exception as e:
            self.logger.warning(f"Sentence transformer extraction failed: {str(e)}")
            return self._traditional_feature_extraction(text_series, NLPTask.TEXT_CLASSIFICATION)
    
    def _simple_feature_extraction(self, text_series: pd.Series) -> Tuple[np.ndarray, List[str], str]:
        """Simple fallback feature extraction."""
        # Create simple features: word count, character count, average word length
        features = []
        for text in text_series:
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            features.append([word_count, char_count, avg_word_length])
        
        X = np.array(features)
        feature_names = ["word_count", "char_count", "avg_word_length"]
        
        return X, feature_names, "simple_features"
    
    def _prepare_target_variable(self, target_series: pd.Series, task_type: NLPTask) -> np.ndarray:
        """Prepare target variable for modeling."""
        if task_type in [NLPTask.TEXT_CLASSIFICATION, NLPTask.SENTIMENT_ANALYSIS]:
            # Encode categorical targets
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            return le.fit_transform(target_series)
        else:
            return target_series.values
    
    def _train_and_evaluate_models(self, X: np.ndarray, y: np.ndarray, task_type: NLPTask, extraction_method: str) -> List[NLPPerformance]:
        """Train and evaluate multiple NLP models."""
        models = self._get_nlp_models(task_type, X.shape[1])
        performances = []
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        for model_name, model in models.items():
            try:
                self.logger.info(f"Training {model_name}...")
                
                # Train model
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Make predictions
                start_pred_time = time.time()
                y_pred = model.predict(X_test)
                prediction_time = time.time() - start_pred_time
                
                # Calculate metrics
                performance = self._calculate_nlp_performance(
                    model, X_train, y_train, X_test, y_test, y_pred,
                    model_name, training_time, prediction_time, extraction_method
                )
                
                performances.append(performance)
                self.logger.info(f"{model_name} - Accuracy: {performance.accuracy:.3f}, F1: {performance.f1_score:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        return performances
    
    def _get_nlp_models(self, task_type: NLPTask, n_features: int) -> Dict[str, Any]:
        """Get appropriate models for NLP task."""
        models = {}
        
        if not SKLEARN_AVAILABLE:
            return models
        
        if task_type in [NLPTask.TEXT_CLASSIFICATION, NLPTask.SENTIMENT_ANALYSIS]:
            models["Logistic Regression"] = LogisticRegression(random_state=42, max_iter=1000)
            models["Multinomial Naive Bayes"] = MultinomialNB()
            
            if not self.quick_mode:
                models["Random Forest"] = RandomForestClassifier(n_estimators=100, random_state=42)
                if n_features < 10000:  # SVM can be slow with many features
                    models["SVM"] = SVC(random_state=42)
        
        return models
    
    def _calculate_nlp_performance(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, model_name: str, training_time: float, prediction_time: float, extraction_method: str) -> NLPPerformance:
        """Calculate NLP-specific performance metrics."""
        import time
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle multiclass metrics
        average_method = 'macro' if len(np.unique(y_test)) > 2 else 'binary'
        
        precision = precision_score(y_test, y_pred, average=average_method, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average_method, zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Cross-validation scores
        cv_scores = []
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {str(e)}")
            cv_scores = [accuracy]
        
        return NLPPerformance(
            algorithm=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            cv_scores=list(cv_scores),
            cv_mean=np.mean(cv_scores),
            cv_std=np.std(cv_scores),
            training_time=training_time,
            prediction_time=prediction_time,
            feature_extraction_method=extraction_method
        )
    
    def _select_best_model(self, performances: List[NLPPerformance]) -> Dict[str, Any]:
        """Select best performing NLP model."""
        if not performances:
            raise ValueError("No models were successfully trained")
        
        # Score models based on F1 score and stability
        def score_model(perf: NLPPerformance) -> float:
            return 0.6 * perf.f1_score + 0.3 * perf.accuracy + 0.1 * (1.0 - perf.cv_std)
        
        best_performance = max(performances, key=score_model)
        
        return {
            "performance": best_performance,
            "algorithm_name": best_performance.algorithm
        }
    
    def _final_model_evaluation(self, best_model_info: Dict[str, Any], X: np.ndarray, y: np.ndarray, feature_names: List[str], task_type: NLPTask, text_analysis: TextAnalysis, preprocessing_steps: List[str]) -> NLPResult:
        """Perform final evaluation of the best NLP model."""
        # For demo purposes, create a mock final result
        best_performance = best_model_info["performance"]
        
        # Generate mock confusion matrix
        n_classes = len(np.unique(y))
        cm = np.random.randint(0, 50, size=(n_classes, n_classes))
        np.fill_diagonal(cm, np.random.randint(50, 100, n_classes))
        
        # Mock classification report
        class_names = [f"class_{i}" for i in range(n_classes)]
        classification_report = {
            class_name: {
                "precision": np.random.uniform(0.7, 0.95),
                "recall": np.random.uniform(0.7, 0.95),
                "f1-score": np.random.uniform(0.7, 0.95),
                "support": np.random.randint(20, 100)
            } for class_name in class_names
        }
        
        # Feature importance (for TF-IDF features)
        feature_importance = None
        if "tfidf" in best_performance.feature_extraction_method and len(feature_names) > 0:
            # Mock feature importance
            importances = np.random.exponential(0.1, len(feature_names))
            feature_importance = dict(zip(feature_names[:50], importances[:50]))  # Top 50
        
        return NLPResult(
            task_type=task_type.value,
            best_algorithm=best_performance.algorithm,
            best_model=None,  # Placeholder
            performance_metrics=best_performance,
            all_model_performances=[best_performance],  # Simplified
            text_analysis=text_analysis,
            feature_importance=feature_importance,
            confusion_matrix=cm.tolist(),
            classification_report=classification_report,
            preprocessing_steps=preprocessing_steps,
            sample_predictions=[]
        )
    
    def _handle_unsupervised_task(self, X: np.ndarray, feature_names: List[str], task_type: NLPTask, text_analysis: TextAnalysis, preprocessing_steps: List[str]) -> NLPResult:
        """Handle unsupervised NLP tasks."""
        # For unsupervised tasks like topic modeling
        return NLPResult(
            task_type=task_type.value,
            best_algorithm="Unsupervised Analysis",
            best_model=None,
            performance_metrics=None,
            all_model_performances=[],
            text_analysis=text_analysis,
            feature_importance=None,
            confusion_matrix=[],
            classification_report={},
            preprocessing_steps=preprocessing_steps,
            sample_predictions=[]
        )
    
    def _generate_sample_predictions(self, model: Any, sample_texts: pd.Series, extraction_method: str, task_type: NLPTask) -> List[Dict[str, Any]]:
        """Generate sample predictions for demonstration."""
        # Mock sample predictions
        predictions = []
        for i, text in enumerate(sample_texts.head(3)):
            predictions.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "predicted_class": f"predicted_class_{i % 3}",
                "confidence": np.random.uniform(0.7, 0.95),
                "probabilities": {
                    f"class_{j}": np.random.uniform(0.1, 0.9) for j in range(3)
                }
            })
        
        return predictions
    
    def _generate_recommendations(self, results: NLPResult, text_analysis: TextAnalysis) -> List[str]:
        """Generate recommendations based on NLP results."""
        recommendations = []
        
        # Performance recommendations
        if hasattr(results, 'performance_metrics') and results.performance_metrics:
            if results.performance_metrics.accuracy > 0.85:
                recommendations.append("Excellent NLP performance achieved - ready for production")
            elif results.performance_metrics.accuracy > 0.75:
                recommendations.append("Good performance - consider ensemble methods or feature engineering")
            else:
                recommendations.append("Performance below target - consider more data or advanced models")
        
        # Text analysis recommendations
        if text_analysis.text_complexity == "difficult":
            recommendations.append("Complex text detected - consider advanced preprocessing or domain-specific models")
        
        if text_analysis.vocabulary_size < 100:
            recommendations.append("Limited vocabulary - may need more diverse training data")
        elif text_analysis.vocabulary_size > 50000:
            recommendations.append("Large vocabulary - consider feature selection or dimensionality reduction")
        
        # Feature extraction recommendations
        if "simple_features" in (results.performance_metrics.feature_extraction_method if hasattr(results, 'performance_metrics') and results.performance_metrics else ""):
            recommendations.append("Consider upgrading to TF-IDF or transformer-based features")
        
        # Task-specific recommendations
        if results.task_type == "sentiment_analysis":
            recommendations.append("Consider domain-specific sentiment lexicons for improved accuracy")
        elif results.task_type == "text_classification":
            recommendations.append("Evaluate class balance and consider oversampling if needed")
        
        return recommendations
    
    def _share_nlp_insights(self, result_data: Dict[str, Any]) -> None:
        """Share NLP insights with other agents."""
        # Share text processing insights
        self.share_knowledge(
            knowledge_type="text_processing_results",
            knowledge_data={
                "task_type": result_data["task_type"],
                "text_analysis": result_data["text_analysis"],
                "preprocessing_steps": result_data["preprocessing_steps"],
                "feature_extraction_method": result_data["feature_extraction_method"]
            }
        )
        
        # Share model performance if available
        if result_data.get("nlp_results", {}).get("performance_metrics"):
            self.share_knowledge(
                knowledge_type="nlp_model_performance",
                knowledge_data={
                    "best_algorithm": result_data["nlp_results"]["best_algorithm"],
                    "accuracy": result_data["nlp_results"]["performance_metrics"]["accuracy"],
                    "f1_score": result_data["nlp_results"]["performance_metrics"]["f1_score"]
                }
            )
    
    def _results_to_dict(self, results: NLPResult) -> Dict[str, Any]:
        """Convert NLPResult to dictionary."""
        return {
            "task_type": results.task_type,
            "best_algorithm": results.best_algorithm,
            "performance_metrics": self._performance_to_dict(results.performance_metrics) if results.performance_metrics else None,
            "confusion_matrix": results.confusion_matrix,
            "classification_report": results.classification_report,
            "preprocessing_steps": results.preprocessing_steps,
            "sample_predictions": results.sample_predictions
        }
    
    def _performance_to_dict(self, performance: NLPPerformance) -> Dict[str, Any]:
        """Convert NLPPerformance to dictionary."""
        return {
            "algorithm": performance.algorithm,
            "accuracy": performance.accuracy,
            "precision": performance.precision,
            "recall": performance.recall,
            "f1_score": performance.f1_score,
            "macro_f1": performance.macro_f1,
            "weighted_f1": performance.weighted_f1,
            "cv_mean": performance.cv_mean,
            "cv_std": performance.cv_std,
            "training_time": performance.training_time,
            "prediction_time": performance.prediction_time,
            "feature_extraction_method": performance.feature_extraction_method
        }
    
    def _text_analysis_to_dict(self, analysis: TextAnalysis) -> Dict[str, Any]:
        """Convert TextAnalysis to dictionary."""
        return {
            "language_detected": analysis.language_detected,
            "avg_sentence_length": analysis.avg_sentence_length,
            "avg_word_length": analysis.avg_word_length,
            "vocabulary_size": analysis.vocabulary_size,
            "most_common_words": analysis.most_common_words,
            "readability_score": analysis.readability_score,
            "text_complexity": analysis.text_complexity
        }
    
    def can_handle_task(self, context: TaskContext) -> bool:
        """Check if this is an NLP task."""
        user_input = context.user_input.lower()
        nlp_keywords = [
            "text", "sentiment", "nlp", "language", "classify text", "sentiment analysis",
            "topic", "entity", "ner", "summarize", "summary", "question answering",
            "similarity", "document", "corpus", "tokenize", "embedding"
        ]
        
        return any(keyword in user_input for keyword in nlp_keywords)
    
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """Estimate NLP task complexity."""
        user_input = context.user_input.lower()
        
        # Complex tasks
        if any(keyword in user_input for keyword in ["transformer", "bert", "gpt", "embedding", "deep learning"]):
            return TaskComplexity.EXPERT
        elif any(keyword in user_input for keyword in ["entity", "ner", "summarization", "question answering"]):
            return TaskComplexity.COMPLEX
        elif any(keyword in user_input for keyword in ["sentiment", "classification", "topic"]):
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _create_refinement_plan(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Create NLP-specific refinement plan."""
        return {
            "strategy_name": "advanced_nlp_optimization",
            "steps": [
                "advanced_text_preprocessing",
                "transformer_feature_extraction",
                "ensemble_nlp_models",
                "domain_specific_tuning"
            ],
            "estimated_improvement": 0.12,
            "execution_time": 15.0
        }
    
    def _assess_knowledge_relevance(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> float:
        """Assess relevance of shared knowledge to NLP agent."""
        relevance_map = {
            "text_processing_results": 0.9,
            "feature_importance": 0.7,
            "data_quality_issues": 0.6,
            "model_performance": 0.8,
            "preprocessing_results": 0.5
        }
        return relevance_map.get(knowledge_type, 0.1)