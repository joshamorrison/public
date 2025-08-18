#!/usr/bin/env python3
"""
Production API for Media Mix Model
FastAPI endpoint for MMM predictions and budget optimization
"""

import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path

# Load production model
MODEL_PATH = "production_mmm_model.joblib"
model = joblib.load(Path(__file__).parent / "models" / MODEL_PATH)

app = FastAPI(title="Media Mix Model API", version="1.0.0")

class PredictionRequest(BaseModel):
    spend_data: Dict[str, float]
    
class BudgetOptimizationRequest(BaseModel):
    total_budget: float
    constraints: Dict[str, Any] = {}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict_revenue(request: PredictionRequest):
    """Predict revenue from media spend"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([request.spend_data])
        
        # Make prediction
        prediction = model.predict(df)
        
        return {
            "prediction": float(prediction[0]),
            "input_spend": request.spend_data,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/optimize")
async def optimize_budget(request: BudgetOptimizationRequest):
    """Optimize budget allocation across channels"""
    try:
        # This would integrate with the budget optimizer
        # For now, return a simple allocation
        channels = ["tv_spend", "digital_spend", "radio_spend", "print_spend", "social_spend"]
        allocation = {channel: request.total_budget / len(channels) for channel in channels}
        
        return {
            "optimal_allocation": allocation,
            "total_budget": request.total_budget,
            "expected_revenue": request.total_budget * 1.5,  # Simplified
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
