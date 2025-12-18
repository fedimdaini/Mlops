from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, Optional
import joblib
import numpy as np
import pandas as pd

# -------------------------------
# Load models (adjust paths if needed)
# -------------------------------
kmeans = joblib.load("kmeans_model.joblib")      # Your KMeans model
scaler = joblib.load("scaler.joblib")            # Your RobustScaler

# Load cluster info and create cluster_id -> rush_category mapping
cluster_info = joblib.load("cluster_info.joblib")
agg_data = cluster_info['agg_data']
cluster_to_rush = agg_data.groupby('cluster')['rush_category'] \
                           .agg(lambda x: x.mode()[0]) \
                           .to_dict()
# Example: {0: 'Moderate', 1: 'Non-rush', 2: 'Rush'}

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Rush Period Prediction API")

# Input model
class PredictionInput(BaseModel):
    orders_count: float
    avg_onshift: float
    utilization_rate: float
    demand_supply_ratio: float
    driver_efficiency: Optional[float] = None  # Optional

# Output model
class PredictionOutput(BaseModel):
    cluster_id: int
    rush_category: Literal["Rush", "Moderate", "Non-rush"]
    recommendations: list[str]

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    X_input = np.array([[data.orders_count,
                         data.avg_onshift,
                         data.utilization_rate,
                         data.demand_supply_ratio,
                         data.driver_efficiency if data.driver_efficiency is not None else 0]])

    # Scale input
    X_scaled = scaler.transform(X_input)

    # Predict cluster
    cluster_id = int(kmeans.predict(X_scaled)[0])

    # Get rush category from mapping
    rush_category = cluster_to_rush.get(cluster_id)
    if rush_category is None:
        raise ValueError(f"Unexpected cluster_id: {cluster_id}")

    # Generate recommendations
    if rush_category == "Rush":
        recommendations = ["Schedule 30-50% extra drivers"]
    elif rush_category == "Moderate":
        recommendations = ["Maintain standard staffing"]
    else:  # "Non-rush"
        recommendations = ["Reduce staffing by 20-30%"]

    return {
        "cluster_id": cluster_id,
        "rush_category": rush_category,
        "recommendations": recommendations
    }

# ===============================
# Optional root endpoint
# ===============================
@app.get("/")
def read_root():
    return {"message": "Rush Period Prediction API is running!"}
