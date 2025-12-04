#!/usr/bin/env python3
"""
FastAPI Backend for EV Battery Dashboard
Serves predictions, explanations, and what-if analysis.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
import asyncio

# Import our models
from train_baseline import BaselineModel
import sys
sys.path.append('../explainability')
from shap_explainer import BatteryExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EV Battery Prediction API",
    description="API for battery health predictions and explanations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
baseline_model = None
explainer = None

# Pydantic models
class PredictionRequest(BaseModel):
    vehicle_id: str
    features: Dict[str, float]

class WhatIfRequest(BaseModel):
    scenarios: Dict[str, float]

class PredictionResponse(BaseModel):
    vehicle_id: str
    current_soh: float
    predicted_rul: int
    uncertainty: Optional[float] = None
    confidence_interval: Optional[List[float]] = None
    risk_level: str
    timestamp: datetime

class ExplanationResponse(BaseModel):
    vehicle_id: str
    top_features: List[Dict[str, Any]]
    prediction: float
    base_value: float

class FleetSummary(BaseModel):
    total_vehicles: int
    avg_soh: float
    high_risk_count: int
    avg_rul: float

class VehicleData(BaseModel):
    vehicle_id: str
    current_soh: float
    predicted_rul: int
    risk_level: str
    last_update: datetime
    mileage: int
    location: str

class FleetResponse(BaseModel):
    vehicles: List[VehicleData]
    summary: FleetSummary

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global baseline_model, explainer
    
    try:
        logger.info("Loading baseline model...")
        baseline_model = BaselineModel()
        baseline_model.load_model("artifacts")
        logger.info("Baseline model loaded successfully")
        
        logger.info("Loading explainer...")
        explainer = BatteryExplainer("artifacts")
        explainer.load_model()
        logger.info("Explainer loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Continue without models for demo purposes

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": {
            "baseline": baseline_model is not None,
            "explainer": explainer is not None
        }
    }

# Fleet overview endpoint
@app.get("/fleet/overview", response_model=FleetResponse)
async def get_fleet_overview():
    """Get fleet overview with summary statistics"""
    try:
        # Generate mock fleet data
        vehicles = []
        for i in range(1, 51):  # 50 vehicles
            vehicle_id = f"EV_{i:04d}"
            soh = 85 + np.random.random() * 10
            rul = int(200 + np.random.random() * 800)
            
            risk_level = "low"
            if soh < 85 or rul < 300:
                risk_level = "high"
            elif soh < 90 or rul < 500:
                risk_level = "medium"
            
            vehicles.append(VehicleData(
                vehicle_id=vehicle_id,
                current_soh=soh,
                predicted_rul=rul,
                risk_level=risk_level,
                last_update=datetime.now() - timedelta(hours=np.random.randint(0, 24)),
                mileage=int(20000 + np.random.random() * 80000),
                location=np.random.choice([
                    "New York, NY", "Los Angeles, CA", "Chicago, IL", 
                    "Houston, TX", "Phoenix, AZ", "Philadelphia, PA"
                ])
            ))
        
        # Calculate summary
        avg_soh = np.mean([v.current_soh for v in vehicles])
        high_risk_count = len([v for v in vehicles if v.risk_level == "high"])
        avg_rul = np.mean([v.predicted_rul for v in vehicles])
        
        summary = FleetSummary(
            total_vehicles=len(vehicles),
            avg_soh=avg_soh,
            high_risk_count=high_risk_count,
            avg_rul=avg_rul
        )
        
        return FleetResponse(vehicles=vehicles, summary=summary)
        
    except Exception as e:
        logger.error(f"Error in fleet overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Vehicle data endpoint
@app.get("/vehicle/{vehicle_id}/data")
async def get_vehicle_data(vehicle_id: str):
    """Get detailed data for a specific vehicle"""
    try:
        # Generate mock timeline data
        now = datetime.now()
        timeline = []
        
        for i in range(90, -1, -1):  # 90 days of data
            timestamp = now - timedelta(days=i)
            soh = 90 - (i * 0.05) + np.random.normal(0, 1)
            
            timeline.append({
                "timestamp": timestamp.isoformat(),
                "soh": max(75, min(100, soh)),
                "pack_voltage": 380 + np.random.normal(0, 10),
                "pack_current": -50 + np.random.normal(0, 30),
                "pack_temp": 25 + np.random.normal(0, 8),
                "soc": 20 + np.random.random() * 60
            })
        
        current_status = {
            "soh": timeline[-1]["soh"],
            "soc": timeline[-1]["soc"],
            "pack_voltage": timeline[-1]["pack_voltage"],
            "pack_temp": timeline[-1]["pack_temp"],
            "mileage": 45000 + np.random.randint(0, 20000),
            "location": "San Francisco, CA",
            "last_charge": (now - timedelta(hours=6)).isoformat()
        }
        
        alerts = [
            {"type": "warning", "message": "Temperature exceeded 40°C during last charge session"},
            {"type": "info", "message": "Recommended maintenance check in 30 days"}
        ]
        
        return {
            "vehicle_id": vehicle_id,
            "timeline": timeline,
            "current_status": current_status,
            "alerts": alerts
        }
        
    except Exception as e:
        logger.error(f"Error getting vehicle data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Vehicle prediction endpoint
@app.get("/vehicle/{vehicle_id}/prediction", response_model=PredictionResponse)
async def get_vehicle_prediction(vehicle_id: str):
    """Get prediction for a specific vehicle"""
    try:
        # Mock prediction data
        soh = 88.5 + np.random.normal(0, 2)
        rul = int(400 + np.random.normal(0, 100))
        uncertainty = 45
        
        risk_level = "low"
        if soh < 85 or rul < 300:
            risk_level = "high"
        elif soh < 90 or rul < 500:
            risk_level = "medium"
        
        return PredictionResponse(
            vehicle_id=vehicle_id,
            current_soh=soh,
            predicted_rul=rul,
            uncertainty=uncertainty,
            confidence_interval=[rul - uncertainty, rul + uncertainty],
            risk_level=risk_level,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prediction explanation endpoint
@app.get("/vehicle/{vehicle_id}/explanation", response_model=ExplanationResponse)
async def get_prediction_explanation(vehicle_id: str):
    """Get SHAP explanation for a vehicle's prediction"""
    try:
        # Mock explanation data
        top_features = [
            {"name": "pack_temp_max_30d", "value": 42.3, "contribution": 0.15},
            {"name": "fast_charge_sessions_30d", "value": 8, "contribution": 0.12},
            {"name": "cumulative_cycles", "value": 1250, "contribution": 0.10},
            {"name": "vehicle_age_years", "value": 2.3, "contribution": 0.08},
            {"name": "thermal_stress_hours_30d", "value": 15, "contribution": 0.07}
        ]
        
        return ExplanationResponse(
            vehicle_id=vehicle_id,
            top_features=top_features,
            prediction=88.5,
            base_value=85.0
        )
        
    except Exception as e:
        logger.error(f"Error getting explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# What-if analysis endpoint
@app.post("/vehicle/{vehicle_id}/what-if")
async def run_what_if_analysis(vehicle_id: str, request: WhatIfRequest):
    """Run what-if analysis for different scenarios"""
    try:
        scenarios = request.scenarios
        
        # Mock what-if calculation
        baseline_rul = 456
        rul_change = 0
        
        # Calculate impact based on scenario changes
        if "fast_charging_frequency" in scenarios:
            fc_change = (scenarios["fast_charging_frequency"] - 100) / 100
            rul_change += fc_change * -50
        
        if "max_temperature" in scenarios:
            temp_change = (scenarios["max_temperature"] - 100) / 100
            rul_change += temp_change * -30
        
        if "charging_depth" in scenarios:
            depth_change = (scenarios["charging_depth"] - 100) / 100
            rul_change += depth_change * -20
        
        if "driving_aggressiveness" in scenarios:
            drive_change = (scenarios["driving_aggressiveness"] - 100) / 100
            rul_change += drive_change * -15
        
        new_rul = max(50, baseline_rul + rul_change)
        soh_change = rul_change * 0.01
        new_soh = min(100, 88.5 + soh_change)
        
        # Generate timeline projection
        timeline_projection = []
        days = min(int(new_rul), 365)
        daily_degradation = (new_soh - 80) / days
        
        for i in range(0, days + 1, 7):  # Weekly points
            timestamp = datetime.now() + timedelta(days=i)
            soh = max(80, new_soh - (i * daily_degradation))
            timeline_projection.append({
                "timestamp": timestamp.isoformat(),
                "soh": soh,
                "scenario": "modified"
            })
        
        return {
            "vehicle_id": vehicle_id,
            "new_rul": int(new_rul),
            "new_soh": new_soh,
            "rul_change": int(rul_change),
            "soh_change": soh_change,
            "factor_impacts": {
                "fast_charging": int(fc_change * -50) if "fast_charging_frequency" in scenarios else 0,
                "temperature": int(temp_change * -30) if "max_temperature" in scenarios else 0,
                "charging_depth": int(depth_change * -20) if "charging_depth" in scenarios else 0,
                "driving": int(drive_change * -15) if "driving_aggressiveness" in scenarios else 0
            },
            "timeline_projection": timeline_projection
        }
        
    except Exception as e:
        logger.error(f"Error in what-if analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict/batch")
async def batch_predict(requests: List[PredictionRequest]):
    """Run batch predictions for multiple vehicles"""
    try:
        results = []
        
        for req in requests:
            # Mock prediction
            soh = 85 + np.random.random() * 10
            rul = int(200 + np.random.random() * 800)
            
            risk_level = "low"
            if soh < 85 or rul < 300:
                risk_level = "high"
            elif soh < 90 or rul < 500:
                risk_level = "medium"
            
            results.append(PredictionResponse(
                vehicle_id=req.vehicle_id,
                current_soh=soh,
                predicted_rul=rul,
                uncertainty=45,
                confidence_interval=[rul - 45, rul + 45],
                risk_level=risk_level,
                timestamp=datetime.now()
            ))
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model retraining endpoint
@app.post("/model/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Trigger model retraining (background task)"""
    try:
        background_tasks.add_task(retrain_models_background)
        return {"message": "Model retraining started", "status": "accepted"}
        
    except Exception as e:
        logger.error(f"Error starting retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def retrain_models_background():
    """Background task for model retraining"""
    try:
        logger.info("Starting model retraining...")
        # Simulate retraining process
        await asyncio.sleep(10)  # Simulate training time
        logger.info("Model retraining completed")
        
    except Exception as e:
        logger.error(f"Error in background retraining: {e}")

# Model metrics endpoint
@app.get("/model/metrics")
async def get_model_metrics():
    """Get current model performance metrics"""
    try:
        return {
            "baseline_model": {
                "mae": 1.85,
                "rmse": 2.34,
                "r2": 0.92,
                "coverage": 0.89,
                "last_trained": "2024-01-15T10:30:00Z"
            },
            "sequence_model": {
                "mae": 1.62,
                "rmse": 2.01,
                "r2": 0.94,
                "coverage": 0.91,
                "last_trained": "2024-01-15T11:45:00Z"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)