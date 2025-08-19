"""
Batch processing endpoints for the Vision Object Classifier API
"""

import time
import uuid
import asyncio
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from PIL import Image
import sys

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.vision_classifier.predict import DishCleanlinessPredictor
from api.models.request_models import BatchClassificationRequest, ModelType
from api.models.response_models import (
    BatchClassificationResponse,
    BatchClassificationResult,
    ErrorResponse
)

router = APIRouter()

# Import predictor getter from classification router
from .classification import get_predictor

@router.post("/classify", response_model=BatchClassificationResponse)
async def classify_batch_images(
    images: List[UploadFile] = File(..., description="List of image files to classify"),
    model_type: Optional[str] = Form(default="balanced", description="Model type to use"),
    return_confidence: Optional[bool] = Form(default=True, description="Return confidence scores"),
    min_confidence: Optional[float] = Form(default=0.0, description="Minimum confidence threshold"),
    max_batch_size: Optional[int] = Form(default=10, description="Maximum batch size")
):
    """
    Classify multiple uploaded images as clean or dirty
    """
    # Validate batch size
    if len(images) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(images)} exceeds maximum {max_batch_size}"
        )
    
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")
    
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
        # Get predictor
        predictor = get_predictor(model_enum)
        
        # Process batch
        batch_start_time = time.time()
        results = []
        successful_count = 0
        failed_count = 0
        
        for i, image in enumerate(images):
            image_id = str(uuid.uuid4())
            filename = image.filename or f"image_{i}"
            
            try:
                # Validate file type
                if not image.content_type or not image.content_type.startswith('image/'):
                    results.append(BatchClassificationResult(
                        image_id=image_id,
                        filename=filename,
                        error="File must be an image"
                    ))
                    failed_count += 1
                    continue
                
                # Read and validate image
                contents = await image.read()
                if len(contents) == 0:
                    results.append(BatchClassificationResult(
                        image_id=image_id,
                        filename=filename,
                        error="Empty image file"
                    ))
                    failed_count += 1
                    continue
                
                # Check file size (10MB limit per image)
                max_size = 10 * 1024 * 1024  # 10MB
                if len(contents) > max_size:
                    results.append(BatchClassificationResult(
                        image_id=image_id,
                        filename=filename,
                        error="Image file too large (max 10MB)"
                    ))
                    failed_count += 1
                    continue
                
                # Load and process image
                pil_image = Image.open(BytesIO(contents))
                
                # Make prediction
                start_time = time.time()
                result = predictor.predict_single(pil_image)
                processing_time = (time.time() - start_time) * 1000
                
                # Map prediction to class name
                predicted_class = "clean" if result['prediction'] == 0 else "dirty"
                confidence = result['confidence']
                
                # Check confidence threshold
                if confidence < min_confidence:
                    results.append(BatchClassificationResult(
                        image_id=image_id,
                        filename=filename,
                        error=f"Confidence {confidence:.3f} below threshold {min_confidence}"
                    ))
                    failed_count += 1
                    continue
                
                # Successful classification
                classification_result = BatchClassificationResult(
                    image_id=image_id,
                    filename=filename,
                    predicted_class=predicted_class,
                    confidence=confidence if return_confidence else None,
                    probabilities={
                        "clean": result['clean_prob'],
                        "dirty": result['dirty_prob']
                    } if return_confidence else None,
                    processing_time_ms=processing_time
                )
                
                results.append(classification_result)
                successful_count += 1
                
            except Exception as e:
                results.append(BatchClassificationResult(
                    image_id=image_id,
                    filename=filename,
                    error=f"Processing failed: {str(e)}"
                ))
                failed_count += 1
        
        total_processing_time = (time.time() - batch_start_time) * 1000
        
        return BatchClassificationResponse(
            success=True,
            results=results,
            total_images=len(images),
            successful_classifications=successful_count,
            failed_classifications=failed_count,
            total_processing_time_ms=total_processing_time,
            model_type=model_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return BatchClassificationResponse(
            success=False,
            results=[],
            total_images=len(images),
            successful_classifications=0,
            failed_classifications=len(images),
            model_type=model_type
        )

@router.get("/status/{batch_id}")
async def get_batch_status(batch_id: str):
    """
    Get status of a batch processing job (placeholder for async processing)
    """
    # This would be used for async batch processing with job queues
    return {
        "batch_id": batch_id,
        "status": "not_implemented",
        "message": "Async batch processing not yet implemented. Use synchronous /classify endpoint."
    }

@router.get("/limits")
async def get_batch_limits():
    """
    Get current batch processing limits and configuration
    """
    return {
        "max_batch_size": 50,
        "max_file_size_mb": 10,
        "supported_formats": ["jpg", "jpeg", "png"],
        "concurrent_batches": 1,  # Current limitation
        "processing_timeout_seconds": 300,
        "features": {
            "async_processing": False,
            "progress_tracking": False,
            "result_caching": False
        }
    }

@router.post("/validate")
async def validate_batch_images(
    images: List[UploadFile] = File(...),
    max_batch_size: Optional[int] = Form(default=10)
):
    """
    Validate a batch of images without classification
    """
    if len(images) > max_batch_size:
        return {
            "valid": False,
            "reason": f"Batch size {len(images)} exceeds maximum {max_batch_size}",
            "total_images": len(images)
        }
    
    validation_results = []
    total_size_mb = 0
    
    for i, image in enumerate(images):
        filename = image.filename or f"image_{i}"
        
        try:
            # Check content type
            if not image.content_type or not image.content_type.startswith('image/'):
                validation_results.append({
                    "filename": filename,
                    "valid": False,
                    "reason": "Not an image file"
                })
                continue
            
            # Check size
            contents = await image.read()
            size_mb = len(contents) / (1024 * 1024)
            total_size_mb += size_mb
            
            if size_mb > 10:
                validation_results.append({
                    "filename": filename,
                    "valid": False,
                    "reason": f"File too large: {size_mb:.1f}MB (max 10MB)"
                })
                continue
            
            # Try to open as image
            try:
                pil_image = Image.open(BytesIO(contents))
                width, height = pil_image.size
                
                validation_results.append({
                    "filename": filename,
                    "valid": True,
                    "info": {
                        "format": pil_image.format,
                        "mode": pil_image.mode,
                        "size": [width, height],
                        "file_size_mb": round(size_mb, 2)
                    }
                })
            except Exception:
                validation_results.append({
                    "filename": filename,
                    "valid": False,
                    "reason": "Invalid or corrupted image"
                })
                
        except Exception as e:
            validation_results.append({
                "filename": filename,
                "valid": False,
                "reason": f"Validation error: {str(e)}"
            })
    
    valid_count = sum(1 for r in validation_results if r.get("valid", False))
    
    return {
        "batch_valid": valid_count == len(images),
        "total_images": len(images),
        "valid_images": valid_count,
        "invalid_images": len(images) - valid_count,
        "total_size_mb": round(total_size_mb, 2),
        "results": validation_results
    }