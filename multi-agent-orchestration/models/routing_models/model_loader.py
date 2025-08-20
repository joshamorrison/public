"""
Routing Model Loader

Utilities for loading, saving, and managing trained routing models.
"""

import pickle
import json
import joblib
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Get the directory containing this file
MODELS_DIR = Path(__file__).parent

# Available model types and their file extensions
MODEL_TYPES = {
    "task_router": ".pkl",
    "performance_predictor": ".pkl", 
    "capability_matcher": ".pkl",
    "load_balancer": ".pkl",
    "cost_predictor": ".pkl"
}

def load_routing_model(model_name: str, model_type: str = "task_router") -> Any:
    """
    Load a trained routing model.
    
    Args:
        model_name: Name of the model to load
        model_type: Type of model (task_router, performance_predictor, etc.)
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model_type is unknown
    """
    if model_type not in MODEL_TYPES:
        available = ", ".join(MODEL_TYPES.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")
    
    # Construct file path
    extension = MODEL_TYPES[model_type]
    model_file = MODELS_DIR / f"{model_name}_{model_type}{extension}"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    try:
        # Load model based on extension
        if extension == ".pkl":
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        elif extension == ".joblib":
            model = joblib.load(model_file)
        else:
            raise ValueError(f"Unsupported model file extension: {extension}")
        
        logger.info(f"Loaded {model_type} model '{model_name}' from {model_file}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model {model_file}: {e}")
        raise

def save_routing_model(model: Any, model_name: str, model_type: str = "task_router", 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save a trained routing model.
    
    Args:
        model: Model object to save
        model_name: Name for the saved model
        model_type: Type of model
        metadata: Optional metadata about the model
        
    Returns:
        True if saved successfully
    """
    if model_type not in MODEL_TYPES:
        available = ", ".join(MODEL_TYPES.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")
    
    # Construct file path
    extension = MODEL_TYPES[model_type]
    model_file = MODELS_DIR / f"{model_name}_{model_type}{extension}"
    
    try:
        # Save model based on extension
        if extension == ".pkl":
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        elif extension == ".joblib":
            joblib.dump(model, model_file)
        else:
            raise ValueError(f"Unsupported model file extension: {extension}")
        
        # Save metadata if provided
        if metadata:
            metadata_file = MODELS_DIR / f"{model_name}_{model_type}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {model_type} model '{model_name}' to {model_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save model {model_file}: {e}")
        return False

def get_available_routing_models() -> Dict[str, List[str]]:
    """
    Get list of available routing models by type.
    
    Returns:
        Dictionary mapping model type to list of available model names
    """
    result = {model_type: [] for model_type in MODEL_TYPES}
    
    if not MODELS_DIR.exists():
        return result
    
    # Scan directory for model files
    for model_file in MODELS_DIR.glob("*"):
        if model_file.is_file():
            filename = model_file.stem
            
            # Parse filename to extract model name and type
            for model_type, extension in MODEL_TYPES.items():
                suffix = f"_{model_type}"
                if filename.endswith(suffix):
                    model_name = filename[:-len(suffix)]
                    result[model_type].append(model_name)
                    break
    
    return result

def load_model_metadata(model_name: str, model_type: str) -> Dict[str, Any]:
    """
    Load metadata for a saved model.
    
    Args:
        model_name: Name of the model
        model_type: Type of model
        
    Returns:
        Model metadata dictionary
    """
    metadata_file = MODELS_DIR / f"{model_name}_{model_type}_metadata.json"
    
    if not metadata_file.exists():
        return {}
    
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load metadata for {model_name}: {e}")
        return {}

def create_default_models():
    """
    Create default routing models with basic functionality.
    Used for bootstrapping the system before training custom models.
    """
    from .task_router import TaskRouter
    from .capability_matcher import CapabilityMatcher
    
    # Create default task router
    default_router = TaskRouter()
    save_routing_model(
        default_router, 
        "default", 
        "task_router",
        metadata={
            "created_at": str(Path(__file__).stat().st_mtime),
            "description": "Default rule-based task router",
            "version": "1.0.0",
            "training_data": "none - rule-based"
        }
    )
    
    # Create default capability matcher
    default_matcher = CapabilityMatcher()
    save_routing_model(
        default_matcher,
        "default",
        "capability_matcher", 
        metadata={
            "created_at": str(Path(__file__).stat().st_mtime),
            "description": "Default capability matching algorithm",
            "version": "1.0.0",
            "training_data": "none - rule-based"
        }
    )
    
    logger.info("Created default routing models")

def list_model_info() -> Dict[str, Any]:
    """
    Get detailed information about all available models.
    
    Returns:
        Comprehensive model information
    """
    available_models = get_available_routing_models()
    model_info = {}
    
    for model_type, model_names in available_models.items():
        model_info[model_type] = []
        
        for model_name in model_names:
            metadata = load_model_metadata(model_name, model_type)
            
            # Get file info
            extension = MODEL_TYPES[model_type]
            model_file = MODELS_DIR / f"{model_name}_{model_type}{extension}"
            
            info = {
                "name": model_name,
                "type": model_type,
                "file_path": str(model_file),
                "file_size": model_file.stat().st_size if model_file.exists() else 0,
                "metadata": metadata
            }
            
            model_info[model_type].append(info)
    
    return model_info