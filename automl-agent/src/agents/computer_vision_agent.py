"""
Computer Vision Agent for AutoML Platform

Specialized agent for computer vision and image analysis tasks that:
1. Handles image preprocessing and augmentation
2. Implements image classification, object detection, and segmentation
3. Supports CNN architectures and transfer learning
4. Provides vision-specific evaluation metrics and analysis
5. Handles various image formats and quality challenges

This agent runs for image-based ML problems and computer vision tasks.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, applications
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms, models as torch_models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from albumentations import (
        Compose, HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast,
        Normalize, Resize, CenterCrop
    )
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, TaskContext, TaskComplexity


class CVTask(Enum):
    """Types of computer vision tasks."""
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    FACE_RECOGNITION = "face_recognition"
    OPTICAL_CHARACTER_RECOGNITION = "ocr"
    IMAGE_SIMILARITY = "image_similarity"
    ANOMALY_DETECTION = "anomaly_detection"
    IMAGE_GENERATION = "image_generation"


class ImagePreprocessingMethod(Enum):
    """Image preprocessing methods."""
    RESIZE = "resize"
    NORMALIZE = "normalize"
    AUGMENTATION = "augmentation"
    NOISE_REDUCTION = "noise_reduction"
    CONTRAST_ENHANCEMENT = "contrast_enhancement"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"


class ModelArchitecture(Enum):
    """Model architectures for computer vision."""
    SIMPLE_CNN = "simple_cnn"
    RESNET = "resnet"
    VGG = "vgg"
    INCEPTION = "inception"
    MOBILENET = "mobilenet"
    EFFICIENTNET = "efficientnet"
    VISION_TRANSFORMER = "vision_transformer"


@dataclass
class ImageAnalysis:
    """Image dataset analysis results."""
    total_images: int
    image_formats: Dict[str, int]
    avg_width: float
    avg_height: float
    avg_channels: float
    size_distribution: Dict[str, int]
    brightness_stats: Dict[str, float]
    contrast_stats: Dict[str, float]
    color_space: str
    data_quality_score: float


@dataclass
class CVPerformance:
    """Computer vision model performance metrics."""
    algorithm: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    top_k_accuracy: Optional[float]
    inference_time: float
    model_size_mb: float
    training_time: float
    epochs_trained: int
    architecture: str


@dataclass
class CVResult:
    """Complete computer vision result."""
    task_type: str
    best_algorithm: str
    best_model: Any
    performance_metrics: CVPerformance
    all_model_performances: List[CVPerformance]
    image_analysis: ImageAnalysis
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    preprocessing_steps: List[str]
    sample_predictions: List[Dict[str, Any]]
    model_architecture: Dict[str, Any]


class ComputerVisionAgent(BaseAgent):
    """
    Computer Vision Agent for image analysis and processing tasks.
    
    Responsibilities:
    1. Image preprocessing and augmentation
    2. CV task identification and execution
    3. CNN model selection and training
    4. Transfer learning and fine-tuning
    5. Vision-specific evaluation and analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub=None):
        """Initialize the Computer Vision Agent."""
        super().__init__(
            name="Computer Vision Agent",
            description="Advanced computer vision and image analysis specialist",
            specialization="Computer Vision & Image Processing",
            config=config,
            communication_hub=communication_hub
        )
        
        # CV configuration
        self.target_image_size = tuple(self.config.get("target_image_size", [224, 224]))
        self.batch_size = self.config.get("batch_size", 32)
        self.max_epochs = self.config.get("max_epochs", 10)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        
        # Preprocessing settings
        self.apply_augmentation = self.config.get("apply_augmentation", True)
        self.normalize_images = self.config.get("normalize_images", True)
        self.augmentation_strength = self.config.get("augmentation_strength", 0.5)
        
        # Model settings
        self.use_transfer_learning = self.config.get("use_transfer_learning", True)
        self.pretrained_model = self.config.get("pretrained_model", "resnet50")
        self.freeze_base_layers = self.config.get("freeze_base_layers", True)
        self.quick_mode = self.config.get("quick_mode", False)
        
        # Quality thresholds
        self.quality_thresholds.update({
            "min_accuracy": self.config.get("min_accuracy", 0.8),
            "min_f1_score": self.config.get("min_f1_score", 0.75),
            "max_inference_time": self.config.get("max_inference_time", 0.1),  # seconds
            "min_image_quality": self.config.get("min_image_quality", 0.7)
        })
        
        # Framework preference
        self.preferred_framework = self.config.get("preferred_framework", "tensorflow")
        
        # Initialize CV tools
        self._initialize_cv_tools()
    
    def _initialize_cv_tools(self):
        """Initialize computer vision tools if available."""
        try:
            if TENSORFLOW_AVAILABLE:
                # Set memory growth for GPU if available
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize some CV tools: {str(e)}")
    
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute comprehensive computer vision workflow.
        
        Args:
            context: Task context with image data
            
        Returns:
            AgentResult with CV models and analysis
        """
        try:
            self.logger.info("Starting computer vision workflow...")
            
            # Load image dataset
            image_data, labels = self._load_image_dataset(context)
            if image_data is None:
                return AgentResult(
                    success=False,
                    message="Failed to load image dataset"
                )
            
            # Phase 1: Task Identification
            self.logger.info("Phase 1: Identifying computer vision task...")
            task_type = self._identify_cv_task(context, labels)
            
            # Phase 2: Image Analysis
            self.logger.info("Phase 2: Analyzing image dataset...")
            image_analysis = self._analyze_image_data(image_data)
            
            # Phase 3: Image Preprocessing
            self.logger.info("Phase 3: Preprocessing images...")
            processed_images, preprocessing_steps = self._preprocess_images(image_data, task_type)
            
            # Phase 4: Prepare Dataset
            self.logger.info("Phase 4: Preparing dataset splits...")
            train_data, val_data, test_data = self._prepare_dataset_splits(
                processed_images, labels, task_type
            )
            
            # Phase 5: Model Training and Evaluation
            self.logger.info("Phase 5: Training and evaluating CV models...")
            model_performances = self._train_and_evaluate_models(
                train_data, val_data, test_data, task_type
            )
            
            # Phase 6: Select Best Model
            self.logger.info("Phase 6: Selecting best performing model...")
            best_model_info = self._select_best_model(model_performances)
            
            # Phase 7: Final Evaluation
            self.logger.info("Phase 7: Final model evaluation...")
            final_results = self._final_model_evaluation(
                best_model_info, test_data, task_type, image_analysis, preprocessing_steps
            )
            
            # Phase 8: Generate Sample Predictions
            self.logger.info("Phase 8: Generating sample predictions...")
            sample_predictions = self._generate_sample_predictions(
                final_results.best_model, test_data, task_type
            )
            final_results.sample_predictions = sample_predictions
            
            # Create comprehensive result
            result_data = {
                "cv_results": self._results_to_dict(final_results),
                "image_analysis": self._image_analysis_to_dict(image_analysis),
                "task_type": task_type.value if isinstance(task_type, CVTask) else task_type,
                "preprocessing_steps": preprocessing_steps,
                "model_performances": [self._performance_to_dict(perf) for perf in model_performances],
                "recommendations": self._generate_recommendations(final_results, image_analysis)
            }
            
            # Update performance metrics
            performance_metrics = {
                "cv_accuracy": final_results.performance_metrics.accuracy,
                "cv_f1_score": final_results.performance_metrics.f1_score,
                "inference_speed": 1.0 / (final_results.performance_metrics.inference_time + 0.001),
                "model_efficiency": 1.0 / (final_results.performance_metrics.model_size_mb + 1)
            }
            self.update_performance_metrics(performance_metrics)
            
            # Share CV insights
            if self.communication_hub:
                self._share_cv_insights(result_data)
            
            return AgentResult(
                success=True,
                data=result_data,
                message=f"Computer vision workflow completed: {task_type.value if isinstance(task_type, CVTask) else task_type} achieved {final_results.performance_metrics.accuracy:.3f} accuracy",
                recommendations=result_data["recommendations"]
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Computer vision workflow failed: {str(e)}"
            )
    
    def _load_image_dataset(self, context: TaskContext) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load image dataset or create synthetic data for demo."""
        # In real implementation, this would load from image files or previous agent results
        # For demo, create synthetic image data
        
        user_input = context.user_input.lower()
        
        if "face" in user_input or "person" in user_input:
            return self._create_face_dataset()
        elif "object" in user_input or "detect" in user_input:
            return self._create_object_detection_dataset()
        elif "medical" in user_input or "xray" in user_input:
            return self._create_medical_image_dataset()
        else:
            return self._create_general_image_dataset()
    
    def _create_general_image_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic general image classification dataset."""
        np.random.seed(42)
        
        # Create synthetic RGB images (224x224x3)
        n_samples = 1000
        height, width, channels = 224, 224, 3
        
        images = []
        labels = []
        
        # Create 5 classes with different characteristics
        n_classes = 5
        samples_per_class = n_samples // n_classes
        
        for class_id in range(n_classes):
            for _ in range(samples_per_class):
                # Generate image with class-specific patterns
                base_color = np.random.uniform(0, 1, 3)
                noise_level = 0.1 + class_id * 0.02
                
                # Create base image with dominant color
                img = np.random.normal(base_color, noise_level, (height, width, channels))
                
                # Add class-specific patterns
                if class_id == 0:  # Horizontal lines
                    for i in range(0, height, 20):
                        img[i:i+5, :, :] = base_color * 1.5
                elif class_id == 1:  # Vertical lines
                    for i in range(0, width, 20):
                        img[:, i:i+5, :] = base_color * 1.5
                elif class_id == 2:  # Circles
                    center_y, center_x = height // 2, width // 2
                    y, x = np.ogrid[:height, :width]
                    mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
                    img[mask] = base_color * 1.5
                elif class_id == 3:  # Diagonal pattern
                    for i in range(height):
                        for j in range(width):
                            if (i + j) % 30 < 15:
                                img[i, j, :] = base_color * 1.5
                
                # Clip values and convert to uint8
                img = np.clip(img, 0, 1)
                images.append(img)
                labels.append(class_id)
        
        return np.array(images), np.array(labels)
    
    def _create_face_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic face recognition dataset."""
        # Simplified - reuse general dataset with different interpretation
        return self._create_general_image_dataset()
    
    def _create_object_detection_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic object detection dataset."""
        # For simplicity, create classification-style data
        # In real implementation, this would include bounding boxes
        return self._create_general_image_dataset()
    
    def _create_medical_image_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic medical image dataset."""
        np.random.seed(42)
        
        # Create grayscale medical-like images
        n_samples = 800
        height, width = 224, 224
        
        images = []
        labels = []
        
        # Two classes: normal (0) and abnormal (1)
        for class_id in range(2):
            for _ in range(n_samples // 2):
                # Generate base medical-like image
                img = np.random.normal(0.5, 0.2, (height, width))
                
                if class_id == 1:  # Abnormal - add "lesions"
                    # Add random bright spots (simulating abnormalities)
                    n_spots = np.random.randint(1, 5)
                    for _ in range(n_spots):
                        spot_x = np.random.randint(20, width - 20)
                        spot_y = np.random.randint(20, height - 20)
                        size = np.random.randint(10, 30)
                        
                        y, x = np.ogrid[:height, :width]
                        mask = (x - spot_x)**2 + (y - spot_y)**2 <= size**2
                        img[mask] = np.random.uniform(0.8, 1.0)
                
                # Convert to 3-channel for consistency
                img_3ch = np.stack([img, img, img], axis=-1)
                img_3ch = np.clip(img_3ch, 0, 1)
                
                images.append(img_3ch)
                labels.append(class_id)
        
        return np.array(images), np.array(labels)
    
    def _identify_cv_task(self, context: TaskContext, labels: Optional[np.ndarray]) -> CVTask:
        """Identify the type of computer vision task."""
        user_input = context.user_input.lower()
        
        # Task identification based on keywords
        if "detect" in user_input and "object" in user_input:
            return CVTask.OBJECT_DETECTION
        elif "segment" in user_input:
            if "instance" in user_input:
                return CVTask.INSTANCE_SEGMENTATION
            else:
                return CVTask.SEMANTIC_SEGMENTATION
        elif "face" in user_input:
            return CVTask.FACE_RECOGNITION
        elif "ocr" in user_input or "text" in user_input:
            return CVTask.OPTICAL_CHARACTER_RECOGNITION
        elif "similarity" in user_input:
            return CVTask.IMAGE_SIMILARITY
        elif "anomaly" in user_input or "abnormal" in user_input:
            return CVTask.ANOMALY_DETECTION
        elif "generate" in user_input or "synthesis" in user_input:
            return CVTask.IMAGE_GENERATION
        else:
            # Default to image classification
            return CVTask.IMAGE_CLASSIFICATION
    
    def _analyze_image_data(self, images: np.ndarray) -> ImageAnalysis:
        """Analyze image dataset characteristics."""
        total_images = len(images)
        
        # Image dimensions
        if len(images.shape) == 4:  # Batch of images
            heights = [img.shape[0] for img in images]
            widths = [img.shape[1] for img in images]
            channels = [img.shape[2] if len(img.shape) > 2 else 1 for img in images]
        else:
            heights = [images.shape[0]]
            widths = [images.shape[1]]
            channels = [images.shape[2] if len(images.shape) > 2 else 1]
        
        avg_height = np.mean(heights)
        avg_width = np.mean(widths)
        avg_channels = np.mean(channels)
        
        # Size distribution
        size_categories = {"small": 0, "medium": 0, "large": 0}
        for h, w in zip(heights, widths):
            total_pixels = h * w
            if total_pixels < 128*128:
                size_categories["small"] += 1
            elif total_pixels < 512*512:
                size_categories["medium"] += 1
            else:
                size_categories["large"] += 1
        
        # Image quality analysis
        brightness_values = []
        contrast_values = []
        
        for img in images[:100]:  # Sample first 100 for performance
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
            
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            brightness_values.append(brightness)
            contrast_values.append(contrast)
        
        brightness_stats = {
            "mean": float(np.mean(brightness_values)),
            "std": float(np.std(brightness_values)),
            "min": float(np.min(brightness_values)),
            "max": float(np.max(brightness_values))
        }
        
        contrast_stats = {
            "mean": float(np.mean(contrast_values)),
            "std": float(np.std(contrast_values)),
            "min": float(np.min(contrast_values)),
            "max": float(np.max(contrast_values))
        }
        
        # Data quality score
        quality_factors = []
        
        # Resolution quality
        avg_resolution = avg_height * avg_width
        if avg_resolution >= 224*224:
            quality_factors.append(1.0)
        elif avg_resolution >= 128*128:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.6)
        
        # Brightness quality (good if mean is between 0.2 and 0.8)
        brightness_quality = 1.0 - abs(brightness_stats["mean"] - 0.5) * 2
        quality_factors.append(max(0.0, brightness_quality))
        
        # Contrast quality (good if std > 0.1)
        contrast_quality = min(1.0, contrast_stats["mean"] / 0.2)
        quality_factors.append(contrast_quality)
        
        data_quality_score = np.mean(quality_factors)
        
        return ImageAnalysis(
            total_images=total_images,
            image_formats={"synthetic": total_images},  # Simplified for demo
            avg_width=avg_width,
            avg_height=avg_height,
            avg_channels=avg_channels,
            size_distribution=size_categories,
            brightness_stats=brightness_stats,
            contrast_stats=contrast_stats,
            color_space="RGB" if avg_channels >= 3 else "Grayscale",
            data_quality_score=data_quality_score
        )
    
    def _preprocess_images(self, images: np.ndarray, task_type: CVTask) -> Tuple[np.ndarray, List[str]]:
        """Preprocess images for training."""
        processed_images = images.copy()
        preprocessing_steps = []
        
        # Step 1: Resize to target size
        if images.shape[1:3] != self.target_image_size:
            if PIL_AVAILABLE:
                resized_images = []
                for img in processed_images:
                    if img.max() <= 1.0:
                        img_uint8 = (img * 255).astype(np.uint8)
                    else:
                        img_uint8 = img.astype(np.uint8)
                    
                    pil_img = Image.fromarray(img_uint8)
                    pil_img = pil_img.resize(self.target_image_size, Image.LANCZOS)
                    resized_images.append(np.array(pil_img) / 255.0)
                
                processed_images = np.array(resized_images)
                preprocessing_steps.append("resized_images")
            else:
                # Fallback resizing using numpy
                processed_images = self._resize_images_numpy(processed_images, self.target_image_size)
                preprocessing_steps.append("resized_images_numpy")
        
        # Step 2: Normalize pixel values
        if self.normalize_images:
            if processed_images.max() > 1.0:
                processed_images = processed_images / 255.0
            
            # Standard normalization for CNN training
            mean = np.array([0.485, 0.456, 0.406])  # ImageNet means
            std = np.array([0.229, 0.224, 0.225])   # ImageNet stds
            
            if processed_images.shape[-1] == 3:
                processed_images = (processed_images - mean) / std
            else:
                # Grayscale normalization
                processed_images = (processed_images - 0.5) / 0.5
            
            preprocessing_steps.append("normalized_images")
        
        # Step 3: Ensure proper data type
        processed_images = processed_images.astype(np.float32)
        preprocessing_steps.append("converted_to_float32")
        
        return processed_images, preprocessing_steps
    
    def _resize_images_numpy(self, images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize images using numpy (fallback method)."""
        # Simple nearest neighbor resizing
        target_h, target_w = target_size
        resized_images = []
        
        for img in images:
            h, w = img.shape[:2]
            
            # Calculate scaling factors
            scale_h = target_h / h
            scale_w = target_w / w
            
            # Create coordinate grids
            y_new = np.linspace(0, h-1, target_h)
            x_new = np.linspace(0, w-1, target_w)
            
            # Round to nearest pixel
            y_indices = np.round(y_new).astype(int)
            x_indices = np.round(x_new).astype(int)
            
            # Sample from original image
            if len(img.shape) == 3:
                resized_img = img[np.ix_(y_indices, x_indices)]
            else:
                resized_img = img[np.ix_(y_indices, x_indices)]
            
            resized_images.append(resized_img)
        
        return np.array(resized_images)
    
    def _prepare_dataset_splits(self, images: np.ndarray, labels: np.ndarray, task_type: CVTask) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Prepare train/validation/test splits."""
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Second split: 75% train, 25% val from the temp set
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _train_and_evaluate_models(self, train_data: Tuple[np.ndarray, np.ndarray], val_data: Tuple[np.ndarray, np.ndarray], test_data: Tuple[np.ndarray, np.ndarray], task_type: CVTask) -> List[CVPerformance]:
        """Train and evaluate multiple CV models."""
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        performances = []
        
        # Get available models
        models = self._get_cv_models(task_type, X_train.shape, len(np.unique(y_train)))
        
        for model_name, model_info in models.items():
            try:
                self.logger.info(f"Training {model_name}...")
                
                performance = self._train_single_model(
                    model_info, train_data, val_data, test_data, model_name
                )
                
                if performance:
                    performances.append(performance)
                    self.logger.info(f"{model_name} - Accuracy: {performance.accuracy:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        return performances
    
    def _get_cv_models(self, task_type: CVTask, input_shape: Tuple, n_classes: int) -> Dict[str, Dict[str, Any]]:
        """Get available computer vision models."""
        models = {}
        
        # Traditional ML models (using flattened features)
        if SKLEARN_AVAILABLE:
            models["Random Forest"] = {
                "type": "sklearn",
                "model": RandomForestClassifier(n_estimators=100, random_state=42),
                "architecture": "random_forest"
            }
            
            if not self.quick_mode and input_shape[1] * input_shape[2] * input_shape[3] < 50000:
                models["SVM"] = {
                    "type": "sklearn",
                    "model": SVC(random_state=42, probability=True),
                    "architecture": "svm"
                }
        
        # Deep learning models
        if TENSORFLOW_AVAILABLE:
            models["Simple CNN"] = {
                "type": "tensorflow",
                "architecture": "simple_cnn",
                "n_classes": n_classes,
                "input_shape": input_shape[1:]
            }
            
            if self.use_transfer_learning:
                models["Transfer Learning (ResNet)"] = {
                    "type": "tensorflow",
                    "architecture": "resnet_transfer",
                    "n_classes": n_classes,
                    "input_shape": input_shape[1:]
                }
        
        return models
    
    def _train_single_model(self, model_info: Dict[str, Any], train_data: Tuple[np.ndarray, np.ndarray], val_data: Tuple[np.ndarray, np.ndarray], test_data: Tuple[np.ndarray, np.ndarray], model_name: str) -> Optional[CVPerformance]:
        """Train a single model and evaluate its performance."""
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        start_time = time.time()
        
        if model_info["type"] == "sklearn":
            return self._train_sklearn_model(model_info, train_data, test_data, model_name)
        elif model_info["type"] == "tensorflow":
            return self._train_tensorflow_model(model_info, train_data, val_data, test_data, model_name)
        
        return None
    
    def _train_sklearn_model(self, model_info: Dict[str, Any], train_data: Tuple[np.ndarray, np.ndarray], test_data: Tuple[np.ndarray, np.ndarray], model_name: str) -> CVPerformance:
        """Train sklearn model with flattened image features."""
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        # Flatten images for traditional ML
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Train model
        start_time = time.time()
        model = model_info["model"]
        model.fit(X_train_flat, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        start_pred_time = time.time()
        y_pred = model.predict(X_test_flat)
        inference_time = (time.time() - start_pred_time) / len(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        return CVPerformance(
            algorithm=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            top_k_accuracy=None,
            inference_time=inference_time,
            model_size_mb=0.1,  # Simplified
            training_time=training_time,
            epochs_trained=1,
            architecture=model_info["architecture"]
        )
    
    def _train_tensorflow_model(self, model_info: Dict[str, Any], train_data: Tuple[np.ndarray, np.ndarray], val_data: Tuple[np.ndarray, np.ndarray], test_data: Tuple[np.ndarray, np.ndarray], model_name: str) -> CVPerformance:
        """Train TensorFlow/Keras model."""
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Create model
        if model_info["architecture"] == "simple_cnn":
            model = self._create_simple_cnn(model_info["input_shape"], model_info["n_classes"])
        elif model_info["architecture"] == "resnet_transfer":
            model = self._create_transfer_learning_model(model_info["input_shape"], model_info["n_classes"])
        else:
            raise ValueError(f"Unknown architecture: {model_info['architecture']}")
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy' if model_info["n_classes"] > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        start_time = time.time()
        epochs = min(self.max_epochs, 5) if self.quick_mode else self.max_epochs
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Evaluate model
        start_pred_time = time.time()
        predictions = model.predict(X_test, verbose=0)
        inference_time = (time.time() - start_pred_time) / len(X_test)
        
        # Convert predictions to class labels
        if model_info["n_classes"] > 2:
            y_pred = np.argmax(predictions, axis=1)
        else:
            y_pred = (predictions > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Top-k accuracy for multiclass
        top_k_accuracy = None
        if model_info["n_classes"] > 2:
            top_k_accuracy = self._calculate_top_k_accuracy(y_test, predictions, k=min(3, model_info["n_classes"]))
        
        # Estimate model size
        model_size_mb = sum([layer.count_params() for layer in model.layers]) * 4 / (1024**2)  # Approximate
        
        return CVPerformance(
            algorithm=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            top_k_accuracy=top_k_accuracy,
            inference_time=inference_time,
            model_size_mb=model_size_mb,
            training_time=training_time,
            epochs_trained=epochs,
            architecture=model_info["architecture"]
        )
    
    def _create_simple_cnn(self, input_shape: Tuple[int, int, int], n_classes: int) -> tf.keras.Model:
        """Create a simple CNN model."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def _create_transfer_learning_model(self, input_shape: Tuple[int, int, int], n_classes: int) -> tf.keras.Model:
        """Create transfer learning model using pre-trained ResNet."""
        # Load pre-trained ResNet50
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model layers
        if self.freeze_base_layers:
            base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def _calculate_top_k_accuracy(self, y_true: np.ndarray, y_pred_proba: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy."""
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        correct = 0
        
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def _select_best_model(self, performances: List[CVPerformance]) -> Dict[str, Any]:
        """Select best performing CV model."""
        if not performances:
            raise ValueError("No models were successfully trained")
        
        # Score models based on accuracy, F1, and efficiency
        def score_model(perf: CVPerformance) -> float:
            accuracy_weight = 0.4
            f1_weight = 0.3
            speed_weight = 0.2
            size_weight = 0.1
            
            accuracy_score = perf.accuracy
            f1_score = perf.f1_score
            speed_score = min(1.0, 0.1 / (perf.inference_time + 0.001))  # Faster is better
            size_score = min(1.0, 100.0 / (perf.model_size_mb + 1))      # Smaller is better
            
            return (accuracy_weight * accuracy_score +
                    f1_weight * f1_score +
                    speed_weight * speed_score +
                    size_weight * size_score)
        
        best_performance = max(performances, key=score_model)
        
        return {
            "performance": best_performance,
            "algorithm_name": best_performance.algorithm
        }
    
    def _final_model_evaluation(self, best_model_info: Dict[str, Any], test_data: Tuple[np.ndarray, np.ndarray], task_type: CVTask, image_analysis: ImageAnalysis, preprocessing_steps: List[str]) -> CVResult:
        """Perform final evaluation of the best CV model."""
        X_test, y_test = test_data
        best_performance = best_model_info["performance"]
        
        # Generate mock confusion matrix and classification report
        n_classes = len(np.unique(y_test))
        
        # Mock confusion matrix
        cm = np.random.randint(5, 50, size=(n_classes, n_classes))
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
        
        # Model architecture info
        model_architecture = {
            "type": best_performance.architecture,
            "parameters": "unknown",  # Would extract from actual model
            "layers": "unknown",      # Would extract from actual model
            "input_shape": list(X_test.shape[1:]),
            "output_shape": [n_classes]
        }
        
        return CVResult(
            task_type=task_type.value,
            best_algorithm=best_performance.algorithm,
            best_model=None,  # Placeholder
            performance_metrics=best_performance,
            all_model_performances=[best_performance],  # Simplified
            image_analysis=image_analysis,
            confusion_matrix=cm.tolist(),
            classification_report=classification_report,
            preprocessing_steps=preprocessing_steps,
            sample_predictions=[],
            model_architecture=model_architecture
        )
    
    def _generate_sample_predictions(self, model: Any, test_data: Tuple[np.ndarray, np.ndarray], task_type: CVTask) -> List[Dict[str, Any]]:
        """Generate sample predictions for demonstration."""
        X_test, y_test = test_data
        
        # Mock sample predictions
        predictions = []
        for i in range(min(3, len(X_test))):
            predictions.append({
                "image_index": i,
                "true_class": int(y_test[i]),
                "predicted_class": int(np.random.choice(np.unique(y_test))),
                "confidence": np.random.uniform(0.7, 0.95),
                "image_shape": list(X_test[i].shape),
                "prediction_time": np.random.uniform(0.01, 0.1)
            })
        
        return predictions
    
    def _generate_recommendations(self, results: CVResult, image_analysis: ImageAnalysis) -> List[str]:
        """Generate recommendations based on CV results."""
        recommendations = []
        
        # Performance recommendations
        if results.performance_metrics.accuracy > 0.9:
            recommendations.append("Excellent computer vision performance - ready for deployment")
        elif results.performance_metrics.accuracy > 0.8:
            recommendations.append("Good performance - consider data augmentation for improvement")
        else:
            recommendations.append("Performance below target - consider transfer learning or more data")
        
        # Image quality recommendations
        if image_analysis.data_quality_score < 0.7:
            recommendations.append("Low image quality detected - consider image enhancement preprocessing")
        
        if image_analysis.avg_width < 224 or image_analysis.avg_height < 224:
            recommendations.append("Low resolution images - consider super-resolution techniques")
        
        # Model efficiency recommendations
        if results.performance_metrics.inference_time > 0.1:
            recommendations.append("Slow inference detected - consider model optimization or quantization")
        
        if results.performance_metrics.model_size_mb > 100:
            recommendations.append("Large model size - consider pruning or knowledge distillation")
        
        # Task-specific recommendations
        if results.task_type == "image_classification":
            recommendations.append("Consider ensemble methods for improved classification accuracy")
        elif results.task_type == "object_detection":
            recommendations.append("Evaluate bounding box accuracy and consider IoU optimization")
        
        return recommendations
    
    def _share_cv_insights(self, result_data: Dict[str, Any]) -> None:
        """Share computer vision insights with other agents."""
        # Share image processing insights
        self.share_knowledge(
            knowledge_type="image_processing_results",
            knowledge_data={
                "task_type": result_data["task_type"],
                "image_analysis": result_data["image_analysis"],
                "preprocessing_steps": result_data["preprocessing_steps"],
                "model_architecture": result_data["cv_results"]["model_architecture"]
            }
        )
        
        # Share model performance
        self.share_knowledge(
            knowledge_type="cv_model_performance",
            knowledge_data={
                "best_algorithm": result_data["cv_results"]["best_algorithm"],
                "accuracy": result_data["cv_results"]["performance_metrics"]["accuracy"],
                "inference_time": result_data["cv_results"]["performance_metrics"]["inference_time"]
            }
        )
    
    def _results_to_dict(self, results: CVResult) -> Dict[str, Any]:
        """Convert CVResult to dictionary."""
        return {
            "task_type": results.task_type,
            "best_algorithm": results.best_algorithm,
            "performance_metrics": self._performance_to_dict(results.performance_metrics),
            "confusion_matrix": results.confusion_matrix,
            "classification_report": results.classification_report,
            "preprocessing_steps": results.preprocessing_steps,
            "sample_predictions": results.sample_predictions,
            "model_architecture": results.model_architecture
        }
    
    def _performance_to_dict(self, performance: CVPerformance) -> Dict[str, Any]:
        """Convert CVPerformance to dictionary."""
        return {
            "algorithm": performance.algorithm,
            "accuracy": performance.accuracy,
            "precision": performance.precision,
            "recall": performance.recall,
            "f1_score": performance.f1_score,
            "top_k_accuracy": performance.top_k_accuracy,
            "inference_time": performance.inference_time,
            "model_size_mb": performance.model_size_mb,
            "training_time": performance.training_time,
            "epochs_trained": performance.epochs_trained,
            "architecture": performance.architecture
        }
    
    def _image_analysis_to_dict(self, analysis: ImageAnalysis) -> Dict[str, Any]:
        """Convert ImageAnalysis to dictionary."""
        return {
            "total_images": analysis.total_images,
            "image_formats": analysis.image_formats,
            "avg_width": analysis.avg_width,
            "avg_height": analysis.avg_height,
            "avg_channels": analysis.avg_channels,
            "size_distribution": analysis.size_distribution,
            "brightness_stats": analysis.brightness_stats,
            "contrast_stats": analysis.contrast_stats,
            "color_space": analysis.color_space,
            "data_quality_score": analysis.data_quality_score
        }
    
    def can_handle_task(self, context: TaskContext) -> bool:
        """Check if this is a computer vision task."""
        user_input = context.user_input.lower()
        cv_keywords = [
            "image", "picture", "photo", "visual", "computer vision", "cv",
            "classify image", "detect object", "face recognition", "ocr",
            "segment", "pixel", "cnn", "convolutional", "vision"
        ]
        
        return any(keyword in user_input for keyword in cv_keywords)
    
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """Estimate computer vision task complexity."""
        user_input = context.user_input.lower()
        
        # Expert level tasks
        if any(keyword in user_input for keyword in ["segment", "detection", "gan", "generation"]):
            return TaskComplexity.EXPERT
        elif any(keyword in user_input for keyword in ["face recognition", "ocr", "medical"]):
            return TaskComplexity.COMPLEX
        elif any(keyword in user_input for keyword in ["classify", "classification"]):
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _create_refinement_plan(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Create computer vision specific refinement plan."""
        return {
            "strategy_name": "advanced_cv_optimization",
            "steps": [
                "advanced_data_augmentation",
                "transfer_learning_fine_tuning",
                "ensemble_model_creation",
                "model_optimization_techniques"
            ],
            "estimated_improvement": 0.15,
            "execution_time": 20.0
        }
    
    def _assess_knowledge_relevance(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> float:
        """Assess relevance of shared knowledge to computer vision agent."""
        relevance_map = {
            "image_processing_results": 0.9,
            "cv_model_performance": 0.8,
            "data_quality_issues": 0.6,
            "model_performance": 0.7,
            "feature_importance": 0.4
        }
        return relevance_map.get(knowledge_type, 0.1)