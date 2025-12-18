# ==========================
# Delivery Time ML Pipeline
# ==========================

# Variables
PYTHON = python3
PIP = pip3
DATA_FILE = cleaned_datadelevry.csv
MODEL_FILE = model_pipeline.joblib
ENV_NAME = venv

# ==========================
# 1. Setup environment
# ==========================
setup:
	@echo "🚀 Setting up virtual environment..."
	$(PYTHON) -m venv venv
	. venv/bin/activate && $(PIP) install --upgrade pip
	. venv/bin/activate && $(PIP) install -r requirements.txt
	@echo "✅ Environment setup complete"

# ==========================
# 2. Prepare data (just to check)
# ==========================
data:
	@echo "📥 Preparing data..."
	$(PYTHON) main.py --data_path $(DATA_FILE) --mode full
	@echo "✅ Data preparation completed"

# ==========================
# 3. Train model
# ==========================
train:
	@echo "🤖 Training model..."
	$(PYTHON) main.py --data_path $(DATA_FILE) --mode train --model_path $(MODEL_FILE)
	@echo "✅ Model training completed"

# ==========================
# 4. Evaluate existing model
# ==========================
evaluate:
	@echo "📊 Evaluating model..."
	$(PYTHON) main.py --data_path $(DATA_FILE) --mode evaluate --model_path $(MODEL_FILE)
	@echo "✅ Model evaluation completed"

# ==========================
# 5. Launch Jupyter Notebook
# ==========================
notebook:
	@echo "📓 Launching Jupyter Notebook..."
	jupyter notebook

# ==========================
# 6. Clean temporary files
# ==========================
clean:
	@echo "🧹 Cleaning project..."
	rm -rf __pycache__ *.pyc *.pyo *.log
	rm -rf venv
	rm -f $(MODEL_FILE)
	@echo "✅ Clean completed"

# ==========================
# Default target
# ==========================
all: setup train evaluate

# ==========================
# 7. FastAPI API
# ==========================
run_api:
	@echo "🚀 Starting FastAPI server..."
	@. $(ENV_NAME)/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000

test_api:
	@echo "🔍 Testing FastAPI /predict endpoint..."
	@. $(ENV_NAME)/bin/activate && python -c "import requests; r = requests.get('http://localhost:8000/predict'); print(r.status_code, r.text)" || echo "Server not running"

requirements_api:
	@echo "📄 Generating FastAPI requirements..."
	@. $(ENV_NAME)/bin/activate && pip freeze | grep -E '(fastapi|uvicorn|pydantic)' > api_requirements.txt


# ==========================
# Docker
# ==========================

DOCKER_IMAGE =fedi_mdaini_mlops
DOCKER_TAG =latest
FULL_IMAGE =$(DOCKER_IMAGE):$(DOCKER_TAG)
DOCKERHUB_USERNAME =fedimdaini
# Build Docker image
docker_build:
	@echo "🐳 Building Docker image..."
	docker build -t $(FULL_IMAGE) .

# Run Docker container locally
docker_run:
	@echo "🚀 Running Docker container..."
	docker run -p 8000:8000 --name $(DOCKER_IMAGE)_container $(FULL_IMAGE)

# Stop and remove container
docker_stop:
	@echo "🛑 Stopping Docker container..."
	docker stop $(DOCKER_IMAGE)_container || true
	docker rm $(DOCKER_IMAGE)_container || true

# Login to Docker Hub
docker_login:
	@echo "🔑 Logging in to Docker Hub..."
	docker login

# Tag and push to Docker Hub
docker_push: docker_login
	@echo "📤 Tagging and pushing image to Docker Hub..."
	docker tag $(FULL_IMAGE) $(DOCKERHUB_USERNAME)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKERHUB_USERNAME)/$(DOCKER_IMAGE):$(DOCKER_TAG)


# ==========================
# MLflow UI
# ==========================
mlflow_ui:
	@echo "📊 Starting MLflow UI..."
	mlflow ui --host 0.0.0.0 --port 5001


# ==========================
# ELK Stack
# ==========================
start_elk:
	@echo "🚀 Starting Elasticsearch and Kibana..."
	@docker-compose down 2>/dev/null || true
	@docker-compose up -d
	@sleep 10
	@echo "✅ ELK stack started"
	@echo "   Elasticsearch: http://localhost:9200"
	@echo "   Kibana: http://localhost:5601"

stop_elk:
	@echo "🛑 Stopping ELK stack..."
	docker-compose down
	@echo "✅ ELK stack stopped"

check_elk:
	@echo "🔍 Checking ELK status..."
	@curl -s http://localhost:9200 | python -m json.tool || echo "Elasticsearch not running"
	@curl -s http://localhost:5601/api/status | python -m json.tool || echo "Kibana not running"

# ==========================
# MLflow UI with Elasticsearch for artifacts
# ==========================
mlflow_ui_es:
	@echo "📊 Starting MLflow UI with Elasticsearch artifact storage..."
	mlflow ui \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root elasticsearch+http://localhost:9200/mlflow_artifacts \
		--host 0.0.0.0 \
		--port 5000

# Alternative: Use file system for artifacts, Elasticsearch for logging only
mlflow_ui_local:
	@echo "📊 Starting MLflow UI with local storage..."
	mlflow ui \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 0.0.0.0 \
		--port 5000

# ==========================
# Run training with monitoring
# ==========================
#train_with_monitoring: start_elk
#	@echo "🤖 Training with full monitoring..."
#	$(PYTHON) main.py --data_path $(DATA_FILE) --mode train
#	@echo "✅ Training completed. Metrics sent to Elasticsearch"

# ==========================
# View logs in Kibana
# ==========================
setup_kibana:
	@echo "🔧 Setting up Kibana index pattern..."
	@echo "1. Go to http://localhost:5601"
	@echo "2. Navigate to Stack Management > Index Patterns"
	@echo "3. Create index pattern: mlflow-metrics*"
	@echo "4. Use @timestamp as time field"
	@echo "5. Explore in Discover tab"

# ==========================
# Clean ELK data
# ==========================
clean_elk:
	@echo "🧹 Cleaning ELK data..."
	docker-compose down -v
	@echo "✅ ELK data cleaned"


# ==========================
# Monitored Pipeline (Atelier 7)
# ==========================

# Run the complete monitored pipeline
run_monitored_pipeline: start_elk
	@echo "🚀 Running monitored pipeline with Elasticsearch logging..."
	@. $(ENV_NAME)/bin/activate && python pipeline_monitor.py
	@echo "✅ Monitored pipeline completed"

# Run pipeline with full monitoring
run_full_monitoring: start_elk create_pipeline_indices
	@echo "📊 Starting complete monitoring pipeline..."
	@. $(ENV_NAME)/bin/activate && python pipeline_monitor.py
	@echo "✅ Check Kibana for pipeline metrics"

# Create Elasticsearch indices for pipeline monitoring
create_pipeline_indices:
	@echo "🔧 Creating pipeline monitoring indices..."
	@curl -X PUT "http://localhost:9200/ml-pipeline" \
		-H "Content-Type: application/json" \
		-d '{"mappings": {"properties": {"@timestamp": {"type": "date"},"pipeline_step": {"type": "keyword"},"step_type": {"type": "keyword"},"data": {"type": "object"}}}}' || true
	@curl -X PUT "http://localhost:9200/pipeline-events" \
		-H "Content-Type: application/json" \
		-d '{"mappings": {"properties": {"@timestamp": {"type": "date"},"pipeline_id": {"type": "keyword"},"event_type": {"type": "keyword"},"stage": {"type": "keyword"},"details": {"type": "object"}}}}' || true
	@echo "✅ Indices created"

# View pipeline logs
view_pipeline_logs:
	@echo "📋 Pipeline logs in Elasticsearch:"
	@curl -s "http://localhost:9200/ml-pipeline/_search?q=*&pretty" | head -100
	@echo "\n📋 Pipeline events:"
	@curl -s "http://localhost:9200/pipeline-events/_search?q=*&pretty" | head -100


# ==========================
# MLflow with Elasticsearch (Atelier 7)
# ==========================

# Start MLflow server with Elasticsearch backend
mlflow_server:
	@echo "🚀 Starting MLflow server with Elasticsearch backend..."
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root elasticsearch+http://localhost:9200/mlflow_artifacts \
		--host 0.0.0.0 \
		--port 5000

# Train with full monitoring
train_with_monitoring: start_elk
	@echo "🤖 Training with complete monitoring (MLflow + Elasticsearch)..."
	MLFLOW_TRACKING_URI=http://localhost:5000 \
	$(PYTHON) main.py --data_path $(DATA_FILE) --mode train
	@echo "✅ Training completed. Check Elasticsearch for logs."

# Create Elasticsearch index for MLflow
create_es_index:
	@echo "🔧 Creating MLflow index in Elasticsearch..."
	@curl -X PUT "http://localhost:9200/mlflow-metrics" \
		-H "Content-Type: application/json" \
		-d '{
			"mappings": {
				"properties": {
					"@timestamp": {"type": "date"},
					"run_id": {"type": "keyword"},
					"experiment": {"type": "keyword"},
					"hyperparameters": {"type": "object"},
					"metrics": {"type": "object"},
					"source": {"type": "keyword"}
				}
			}
		}' || true

# Create system monitoring index
create_system_index:
	@echo "🔧 Creating system metrics index..."
	@curl -X PUT "http://localhost:9200/system-metrics" \
		-H "Content-Type: application/json" \
		-d '{
			"mappings": {
				"properties": {
					"@timestamp": {"type": "date"},
					"cpu_percent": {"type": "float"},
					"memory_percent": {"type": "float"},
					"disk_usage": {"type": "float"}
				}
			}
		}' || true

# Start system monitoring in background
start_monitoring:
	@echo "📊 Starting system monitoring..."
	@. $(ENV_NAME)/bin/activate && python system_monitor.py &
	@echo "✅ System monitoring started (PID: $$!)"

# Complete monitoring setup
setup_monitoring: start_elk create_es_index create_system_index mlflow_server
	@echo "✅ Complete monitoring stack is ready!"
	@echo "   MLflow UI: http://localhost:5000"
	@echo "   Kibana: http://localhost:5601"
	@echo "   Elasticsearch: http://localhost:9200"