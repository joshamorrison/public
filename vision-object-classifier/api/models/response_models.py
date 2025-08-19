"""
Response models for the Vision Object Classifier API
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class ClassificationResult(BaseModel):
    """Single classification result"""
    predicted_class: str = Field(description="Predicted class (clean/dirty)")
    confidence: Optional[float] = Field(description="Confidence score (0.0-1.0)")
    probabilities: Optional[Dict[str, float]] = Field(description="Class probabilities")
    processing_time_ms: Optional[float] = Field(description="Processing time in milliseconds")

class ClassificationResponse(BaseModel):
    """Response for single image classification"""
    success: bool = Field(description="Whether classification succeeded")
    result: Optional[ClassificationResult] = Field(description="Classification result")
    error: Optional[str] = Field(description="Error message if failed")
    model_type: str = Field(description="Model type used")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

class BatchClassificationResult(BaseModel):
    """Result for a single image in batch processing"""
    image_id: str = Field(description="Image identifier")
    filename: Optional[str] = Field(description="Original filename")
    predicted_class: Optional[str] = Field(description="Predicted class (clean/dirty)")
    confidence: Optional[float] = Field(description="Confidence score")
    probabilities: Optional[Dict[str, float]] = Field(description="Class probabilities")
    processing_time_ms: Optional[float] = Field(description="Processing time in milliseconds")
    error: Optional[str] = Field(description="Error message if processing failed")

class BatchClassificationResponse(BaseModel):
    """Response for batch image classification"""
    success: bool = Field(description="Whether batch processing succeeded")
    results: List[BatchClassificationResult] = Field(description="Classification results")
    total_images: int = Field(description="Total number of images processed")
    successful_classifications: int = Field(description="Number of successful classifications")
    failed_classifications: int = Field(description="Number of failed classifications")
    total_processing_time_ms: Optional[float] = Field(description="Total processing time")
    model_type: str = Field(description="Model type used")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="Service status (healthy/unhealthy)")
    version: str = Field(description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    uptime_seconds: Optional[float] = Field(description="Service uptime in seconds")
    model_status: Optional[Dict[str, str]] = Field(description="Status of each model type")

class ModelInfoResponse(BaseModel):
    """Model information response"""
    available_models: List[str] = Field(description="Available model types")
    default_model: str = Field(description="Default model type")
    model_details: Dict[str, Dict[str, Any]] = Field(description="Details for each model")
    supported_formats: List[str] = Field(description="Supported image formats")
    max_file_size_mb: int = Field(description="Maximum file size in MB")

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(description="Error message")
    error_code: Optional[str] = Field(description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(description="Request identifier for tracking")