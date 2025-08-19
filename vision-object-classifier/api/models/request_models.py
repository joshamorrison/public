"""
Request models for the Vision Object Classifier API
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum

class ModelType(str, Enum):
    """Available model types"""
    FAST = "fast"
    BALANCED = "balanced" 
    ACCURATE = "accurate"

class ClassificationRequest(BaseModel):
    """Request model for single image classification"""
    model_type: Optional[ModelType] = Field(
        default=ModelType.BALANCED,
        description="Model variant to use for classification"
    )
    return_confidence: Optional[bool] = Field(
        default=True,
        description="Whether to return confidence scores"
    )
    min_confidence: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold (0.0-1.0)"
    )

class BatchClassificationRequest(BaseModel):
    """Request model for batch image classification"""
    model_type: Optional[ModelType] = Field(
        default=ModelType.BALANCED,
        description="Model variant to use for classification"
    )
    return_confidence: Optional[bool] = Field(
        default=True,
        description="Whether to return confidence scores"
    )
    min_confidence: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    max_batch_size: Optional[int] = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of images to process (1-50)"
    )

class ModelConfigRequest(BaseModel):
    """Request for model configuration updates"""
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Global confidence threshold"
    )
    preprocessing_options: Optional[dict] = Field(
        default=None,
        description="Image preprocessing options"
    )

    @validator('preprocessing_options')
    def validate_preprocessing(cls, v):
        if v is not None:
            allowed_keys = {'resize', 'normalize', 'augment'}
            if not all(key in allowed_keys for key in v.keys()):
                raise ValueError(f"Preprocessing options must be subset of {allowed_keys}")
        return v