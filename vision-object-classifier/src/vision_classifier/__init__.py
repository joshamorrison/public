"""
Vision Object Classifier Package
A deep learning solution for classifying dish cleanliness using computer vision.
"""

from .model import create_model, save_model, load_model, FocalLoss, ModelEvaluator
from .data_utils import create_data_loaders, prepare_dataset_structure, DishDataset
from .predict import DishClassifier, predict_single_image
from .synthetic_data import SyntheticDirtyDishGenerator

__version__ = "1.0.0"
__author__ = "Vision Classifier Team"

__all__ = [
    'create_model',
    'save_model', 
    'load_model',
    'FocalLoss',
    'ModelEvaluator',
    'create_data_loaders',
    'prepare_dataset_structure', 
    'DishDataset',
    'DishClassifier',
    'predict_single_image',
    'SyntheticDirtyDishGenerator'
]