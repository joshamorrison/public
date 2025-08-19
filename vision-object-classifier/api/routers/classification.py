"""
Classification endpoints for the Vision Object Classifier API
"""

import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional
from PIL import Image
import sys

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.vision_classifier.predict import DishCleanlinessPredictor
from api.models.request_models import ClassificationRequest, ModelType
from api.models.response_models import (
    ClassificationResponse, 
    ClassificationResult,
    ModelInfoResponse,
    ErrorResponse
)

router = APIRouter()

# Global predictor instances (lazy loaded)
predictors = {}

def get_predictor(model_type: ModelType) -> DishCleanlinessPredictor:
    """Get or create predictor instance for specified model type"""
    if model_type.value not in predictors:
        models_dir = project_root / "models"
        model_path = models_dir / f"{model_type.value}_model.pth"
        config_path = models_dir / f"{model_type.value}_config.json"
        
        if not model_path.exists():
            raise HTTPException(
                status_code=503, 
                detail=f"Model '{model_type.value}' not available. File not found: {model_path}"
            )
        
        try:
            predictor = DishCleanlinessPredictor(
                model_path=str(model_path),
                config_path=str(config_path) if config_path.exists() else None
            )
            predictors[model_type.value] = predictor
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load model '{model_type.value}': {str(e)}"
            )
    
    return predictors[model_type.value]

@router.post("/single", response_model=ClassificationResponse)
async def classify_single_image(
    image: UploadFile = File(..., description="Image file to classify"),
    model_type: Optional[str] = Form(default="balanced", description="Model type to use"),
    return_confidence: Optional[bool] = Form(default=True, description="Return confidence scores"),
    min_confidence: Optional[float] = Form(default=0.0, description="Minimum confidence threshold")
):
    """
    Classify a single uploaded image as clean or dirty
    """
    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate model type
    try:
        model_enum = ModelType(model_type)
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model type. Must be one of: {[e.value for e in ModelType]}"
        )
    
    # Validate confidence threshold
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise HTTPException(status_code=400, detail="min_confidence must be between 0.0 and 1.0")
    
    try:
        # Read and validate image
        contents = await image.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Check file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(contents) > max_size:
            raise HTTPException(status_code=413, detail="Image file too large. Maximum size is 10MB")
        
        # Load image
        pil_image = Image.open(BytesIO(contents))
        
        # Get predictor and make prediction
        predictor = get_predictor(model_enum)
        
        start_time = time.time()
        result = predictor.predict_single(pil_image)
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Map prediction to class name
        predicted_class = "clean" if result['prediction'] == 0 else "dirty"
        confidence = result['confidence']
        
        # Check confidence threshold
        if confidence < min_confidence:
            raise HTTPException(
                status_code=422,
                detail=f"Prediction confidence {confidence:.3f} below threshold {min_confidence}"
            )
        
        # Prepare response
        classification_result = ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence if return_confidence else None,
            probabilities={
                "clean": result['clean_prob'],
                "dirty": result['dirty_prob']
            } if return_confidence else None,
            processing_time_ms=processing_time
        )
        
        return ClassificationResponse(
            success=True,
            result=classification_result,
            model_type=model_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return ClassificationResponse(
            success=False,
            error=f"Classification failed: {str(e)}",
            model_type=model_type
        )

@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about available models
    """
    try:
        models_dir = project_root / "models"
        available_models = []
        model_details = {}
        
        # Check for model files
        for model_type in ModelType:
            model_file = models_dir / f"{model_type.value}_model.pth"
            config_file = models_dir / f"{model_type.value}_config.json"
            
            if model_file.exists():
                available_models.append(model_type.value)
                
                details = {
                    "file_size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                    "config_available": config_file.exists(),
                    "status": "available"
                }
                
                # Add model characteristics
                if model_type == ModelType.FAST:
                    details.update({
                        "speed": "high",
                        "accuracy": "medium", 
                        "memory_usage": "low",
                        "description": "Optimized for speed with acceptable accuracy"
                    })
                elif model_type == ModelType.BALANCED:
                    details.update({
                        "speed": "medium",
                        "accuracy": "high",
                        "memory_usage": "medium",
                        "description": "Good balance of speed and accuracy"
                    })
                elif model_type == ModelType.ACCURATE:
                    details.update({
                        "speed": "low",
                        "accuracy": "very_high",
                        "memory_usage": "high", 
                        "description": "Maximum accuracy, slower inference"
                    })
                
                model_details[model_type.value] = details
        
        if not available_models:
            raise HTTPException(status_code=503, detail="No models available")
        
        return ModelInfoResponse(
            available_models=available_models,
            default_model="balanced",
            model_details=model_details,
            supported_formats=["jpg", "jpeg", "png"],
            max_file_size_mb=10
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.post("/validate")
async def validate_image(image: UploadFile = File(...)):
    """
    Validate an image file without classification
    """
    try:
        # Check content type
        if not image.content_type or not image.content_type.startswith('image/'):
            return {"valid": False, "reason": "Not an image file"}
        
        # Check size
        contents = await image.read()
        size_mb = len(contents) / (1024 * 1024)
        
        if size_mb > 10:
            return {"valid": False, "reason": f"File too large: {size_mb:.1f}MB (max 10MB)"}
        
        # Try to open as image
        try:
            pil_image = Image.open(BytesIO(contents))
            width, height = pil_image.size
            
            return {
                "valid": True,
                "info": {
                    "format": pil_image.format,
                    "mode": pil_image.mode,
                    "size": [width, height],
                    "file_size_mb": round(size_mb, 2)
                }
            }
        except Exception:
            return {"valid": False, "reason": "Invalid or corrupted image"}
            
    except Exception as e:
        return {"valid": False, "reason": f"Validation error: {str(e)}"}