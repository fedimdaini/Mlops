# Delivery Rush-Period Detection: an MLOps pipeline

Detects rush periods in food-delivery operations by clustering 15-minute time windows on demand and
driver supply, then serves the result through an API with experiment tracking and monitoring.

Built over the seven labs (TP1 to TP7) of the MLOps course at ESPRIT: each lab added one layer, from a
notebook to a modular pipeline, Make targets, an API, MLflow, Docker and finally Elasticsearch/Kibana
monitoring. This repository is the final state; the per-lab files are in [`labs/`](labs/).

## The model

1. **Data preparation** (`model_pipeline.prepare_data`): delivery timestamps are converted and shifted to
   local time, then orders are grouped into 15-minute windows per day. Each window gets four features:
   orders count, average drivers on shift, driver utilization rate, and demand/supply ratio.
2. **Training** (`train_model`): features are scaled with a `RobustScaler`, KMeans is fitted for k = 2 to 5,
   and k is chosen by silhouette score (Calinski-Harabasz and Davies-Bouldin are logged alongside).
3. **Labelling**: each cluster is mapped to **Rush**, **Moderate** or **Non-rush**, and the API returns
   operational recommendations for that category.

## Stack

| Layer | Tool |
|---|---|
| Pipeline | Python modules (`model_pipeline.py`, `main.py`) with train / evaluate / full / monitored modes |
| Automation | `Makefile` (setup, train, evaluate, API, Docker, MLflow, ELK targets) |
| Tracking | MLflow: parameters, the three clustering metrics, and the model |
| Serving | FastAPI `POST /predict`, Pydantic input and output schemas |
| Packaging | Docker image on `python:3.12-slim`, with a Make target to push it to Docker Hub |
| Monitoring | Elasticsearch + Kibana (`docker-compose.yml`): every pipeline step, its duration and metrics are logged as events |

## Run it

```bash
make setup        # virtual environment + dependencies
make train        # train and log the run to MLflow
make evaluate     # evaluate the saved model
make mlflow_ui    # MLflow UI on http://localhost:5001
make run_api      # API on http://localhost:8000 (docs at /docs)
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"orders_count": 42, "avg_onshift": 18, "utilization_rate": 0.85, "demand_supply_ratio": 2.3}'
```

The response holds the cluster id, its category (`Rush`, `Moderate` or `Non-rush`) and a list of
recommendations.

With Docker:

```bash
make docker_build
make docker_run
```

With monitoring:

```bash
make start_elk                 # Elasticsearch on :9200, Kibana on :5601
python main.py --data_path cleaned_datadelevry.csv --mode monitored
```

## Files

| Path | What it is |
|---|---|
| `model_pipeline.py` | Data preparation, training, evaluation, saving and loading |
| `main.py` | Command-line entry point with MLflow tracking |
| `pipeline_monitor.py`, `elasticsearch_logger.py`, `monitor_system.py` | Pipeline and system monitoring to Elasticsearch |
| `app.py` | FastAPI service |
| `*.joblib` | Trained KMeans model, scaler and cluster mapping used by the API |
| `cleaned_datadelevry.csv` | Cleaned delivery dataset |
| `ml_mlops_miniprojet.ipynb` | Original exploration notebook |
| `labs/TP1` … `labs/TP7` | Code, Makefiles and screenshots from each lab |
