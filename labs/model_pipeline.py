"""
Machine Learning Pipeline for Delivery Time Analysis
Fully modular implementation aligned with academic requirements
"""

import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

warnings.filterwarnings("ignore")
np.random.seed(42)


# =========================================================
# 1. DATA PREPARATION
# =========================================================
def prepare_data(file_path):
    """
    Load and preprocess delivery data

    Args:
        file_path (str): Path to CSV file

    Returns:
        tuple: cleaned dataframe, aggregated dataframe
    """
    print("=== STEP 1: DATA PREPARATION ===")

    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")

    # Convert timestamps
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"], utc=True)

    # Shift timezone (business logic preserved)
    df["created_at_shifted"] = df["created_at"] - pd.Timedelta(hours=7)
    df["actual_delivery_time_shifted"] = df["actual_delivery_time"] - pd.Timedelta(hours=7)

    # Delivery duration
    df["delivery_time"] = (
        df["actual_delivery_time_shifted"] - df["created_at_shifted"]
    ).dt.total_seconds() / 60

    # Cleaning
    df = df.dropna(subset=["store_primary_category", "order_protocol", "market_id"])
    df = df[(df["delivery_time"] > 0) & (df["delivery_time"] < 500)]

    for col in [
        "total_onshift_partners",
        "total_busy_partners",
        "total_outstanding_orders",
    ]:
        df = df[df[col] >= 0]

    # Feature engineering
    df["hour"] = df["created_at_shifted"].dt.hour
    df["weekday"] = df["created_at_shifted"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    df["total_onshift_partners"] = df["total_onshift_partners"].replace(0, np.nan)
    df["load_ratio"] = (
        df["total_busy_partners"] / df["total_onshift_partners"]
    ).fillna(0).clip(0, 2)

    df["time_bin"] = (
        (df["created_at_shifted"].dt.hour * 60 + df["created_at_shifted"].dt.minute) // 15
    )
    df["date"] = df["created_at_shifted"].dt.date

    # Aggregation for clustering
    agg = (
        df.groupby(["date", "time_bin"])
        .agg(
            orders_count=("load_ratio", "count"),
            avg_onshift=("total_onshift_partners", "mean"),
            avg_busy=("total_busy_partners", "mean"),
            mean_load_ratio=("load_ratio", "mean"),
            avg_outstanding=("total_outstanding_orders", "mean"),
        )
        .reset_index()
    )

    agg["utilization_rate"] = (
        agg["avg_busy"] / agg["avg_onshift"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    agg["demand_supply_ratio"] = (
        agg["orders_count"] / agg["avg_onshift"].replace(0, 1)
    ).clip(0, 10)

    agg["driver_efficiency"] = (
        agg["orders_count"] / agg["avg_busy"].replace(0, 1)
    )

    agg = agg[agg["orders_count"] >= 1]

    print(f"Prepared data shape: {agg.shape}")
    return df, agg


# =========================================================
# 2. MODEL TRAINING
# =========================================================
def train_model(agg_data, k_range=range(2, 6)):
    """
    Train clustering model

    Args:
        agg_data (DataFrame): aggregated data
        k_range (range): candidate cluster numbers

    Returns:
        tuple: trained model, scaler, updated dataframe
    """
    print("=== STEP 2: MODEL TRAINING ===")

    features = [
        "orders_count",
        "avg_onshift",
        "utilization_rate",
        "demand_supply_ratio",
        
    ]

    X = agg_data[features].replace([np.inf, -np.inf], 0).fillna(0)

    # Outlier clipping
    for col in features:
        X[col] = X[col].clip(X[col].quantile(0.01), X[col].quantile(0.99))

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = model.fit_predict(X_scaled)
        results.append(
            {
                "k": k,
                "silhouette": silhouette_score(X_scaled, labels),
                "calinski": calinski_harabasz_score(X_scaled, labels),
                "davies": davies_bouldin_score(X_scaled, labels),
            }
        )

    results_df = pd.DataFrame(results)
    best_k = results_df.loc[results_df["silhouette"].idxmax(), "k"]

    print(results_df)
    print(f"Selected k = {best_k}")

    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    agg_data["cluster"] = final_model.fit_predict(X_scaled)

    return final_model, scaler, agg_data, X_scaled


# =========================================================
# 3. MODEL EVALUATION
# =========================================================
def evaluate_model(model, X_scaled, agg_data):
    """
    Evaluate clustering model

    Returns:
        dict: evaluation metrics
    """
    print("=== STEP 3: MODEL EVALUATION ===")

    metrics = {
        "silhouette": silhouette_score(X_scaled, agg_data["cluster"]),
        "calinski": calinski_harabasz_score(X_scaled, agg_data["cluster"]),
        "davies": davies_bouldin_score(X_scaled, agg_data["cluster"]),
    }

    print(metrics)
    return metrics


# =========================================================
# 4. SAVE / LOAD MODEL
# =========================================================
def save_model(model, scaler, agg_data, path="model_pipeline.joblib"):
    """
    Save trained model
    """
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "agg_data": agg_data,
        },
        path,
    )
    print(f"Model saved to {path}")


def load_model(path="model_pipeline.joblib"):
    """
    Load trained model
    """
    data = joblib.load(path)
    print(f"Model loaded from {path}")
    return data["model"], data["scaler"], data["agg_data"]
