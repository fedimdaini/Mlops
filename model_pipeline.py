"""
Machine Learning Pipeline for Delivery Time Analysis
Fully modular implementation aligned with academic requirements
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import os
import requests
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

warnings.filterwarnings("ignore")
np.random.seed(42)


def log_to_elasticsearch(step_name, data, step_type="pipeline"):
    """
    Log pipeline step to Elasticsearch - FLATTENED version for Kibana
    """
    try:
        import numpy as np

        def flatten_dict(d, parent_key='', sep='_'):
            """Flatten nested dictionary for Elasticsearch"""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k

                # Convert numpy types to Python native types
                if isinstance(v, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    v = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32, np.float16)):
                    v = float(v)
                elif isinstance(v, np.ndarray):
                    v = v.tolist()
                elif isinstance(v, np.bool_):
                    v = bool(v)

                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    # Handle lists by converting to string or leaving as is
                    if all(isinstance(item, (int, float, str, bool)) for item in v):
                        items.append((new_key, v))
                    else:
                        items.append((new_key, str(v)))
                else:
                    items.append((new_key, v))
            return dict(items)

        # Create base document
        doc = {
            "@timestamp": datetime.now().isoformat(),
            "pipeline_step": step_name,
            "step_type": step_type,
            "environment": "production" if __name__ == "__main__" else "development"
        }

        # Flatten and add the data fields at top level
        if isinstance(data, dict):
            flat_data = flatten_dict(data)
            doc.update(flat_data)
        else:
            doc[f"{step_name}_value"] = str(data)

        # Send to Elasticsearch
        es_url = "http://localhost:9200/ml-pipeline/_doc"
        response = requests.post(es_url, json=doc, timeout=2)

        if response.status_code in [200, 201]:
            print(f"✅ Sent to Elasticsearch: {step_name}")
        else:
            print(f"⚠️ Failed to log {step_name}: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"⚠️ Elasticsearch logging failed for {step_name}: {e}")



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

    # Log start of data preparation
    log_to_elasticsearch("data_preparation_start", {
        "file_path": file_path,
        "timestamp": datetime.now().isoformat()
    })

    df = pd.read_csv(file_path)

    # Log initial data stats
    log_to_elasticsearch("data_loaded", {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
    })

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
    initial_rows = df.shape[0]
    df = df.dropna(subset=["store_primary_category", "order_protocol", "market_id"])
    df = df[(df["delivery_time"] > 0) & (df["delivery_time"] < 500)]

    # Log cleaning results
    log_to_elasticsearch("data_cleaning", {
        "initial_rows": initial_rows,
        "final_rows": df.shape[0],
        "rows_removed": initial_rows - df.shape[0],
        "removal_rate": ((initial_rows - df.shape[0]) / initial_rows * 100) if initial_rows > 0 else 0
    })

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

    # Log final aggregation
    log_to_elasticsearch("data_aggregation", {
        "aggregated_rows": agg.shape[0],
        "features_created": list(agg.columns),
        "memory_usage_mb": agg.memory_usage(deep=True).sum() / 1024 ** 2
    })

    print(f"Prepared data shape: {agg.shape}")
    return df, agg


def train_model(agg_data, k_range=range(2, 6)):
    """
    Train clustering model
    """
    print("=== STEP 2: MODEL TRAINING ===")

    features = [
        "orders_count",
        "avg_onshift",
        "utilization_rate",
        "demand_supply_ratio",
    ]

    # Log training start (now features is defined)
    log_to_elasticsearch("model_training_start", {
        "k_range": list(k_range),
        "data_shape": agg_data.shape,
        "features": features
    })

    X = agg_data[features].replace([np.inf, -np.inf], 0).fillna(0)

    # Log feature statistics
    log_to_elasticsearch("feature_statistics", {
        "feature_means": X.mean().to_dict(),
        "feature_stds": X.std().to_dict(),
        "feature_ranges": {
            col: {"min": X[col].min(), "max": X[col].max()}
            for col in features
        }
    })

    # Outlier clipping
    for col in features:
        X[col] = X[col].clip(X[col].quantile(0.01), X[col].quantile(0.99))

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Log scaling results
    log_to_elasticsearch("feature_scaling", {
        "scaler_type": "RobustScaler",
        "scaled_shape": X_scaled.shape,
        "scale_params": {
            "center": scaler.center_.tolist() if hasattr(scaler, 'center_') else None,
            "scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
        }
    })

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

    # Log final model
    log_to_elasticsearch("model_training_complete", {
        "final_k": best_k,
        "final_inertia": final_model.inertia_,
        "cluster_sizes": pd.Series(agg_data["cluster"]).value_counts().to_dict(),
        "training_time": datetime.now().isoformat()
    })

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

    # Log evaluation metrics
    log_to_elasticsearch("model_evaluation", {
        "metrics": metrics,
        "cluster_distribution": pd.Series(agg_data["cluster"]).value_counts().to_dict(),
        "cluster_quality": "good" if metrics["silhouette"] > 0.5 else "fair" if metrics["silhouette"] > 0.3 else "poor"
    })

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


