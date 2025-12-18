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
	mlflow ui --host 0.0.0.0 --port 5000

