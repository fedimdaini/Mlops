from elasticsearch import Elasticsearch, exceptions
from datetime import datetime
import json
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElasticsearchLogger:
    def __init__(self, host='localhost', port=9200, index_prefix="mlflow"):
        self.host = host
        self.port = port
        self.index_prefix = index_prefix
        self.metrics_index = f"{index_prefix}-metrics"
        self.system_index = f"{index_prefix}-system"

        # Initialize Elasticsearch client
        self._init_client()

    def _init_client(self):
        """Initialize Elasticsearch client with retry logic"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.es = Elasticsearch(
                    [{'host': self.host, 'port': self.port}],
                    request_timeout=30,
                    max_retries=3,
                    retry_on_timeout=True
                )

                # Test connection
                if self.es.ping():
                    logger.info(f"✅ Connected to Elasticsearch at {self.host}:{self.port}")
                    return
                else:
                    logger.warning(f"⚠️ Elasticsearch ping failed, attempt {attempt + 1}/{max_retries}")

            except exceptions.ConnectionError as e:
                logger.warning(f"⚠️ Could not connect to Elasticsearch, attempt {attempt + 1}/{max_retries}: {e}")

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"❌ Failed to connect to Elasticsearch after {max_retries} attempts")
                self.es = None

    def create_index(self, index_name=None):
        """Create index with mapping for MLflow metrics"""
        if self.es is None:
            logger.warning("⚠️ Elasticsearch client not initialized, skipping index creation")
            return False

        index_to_create = index_name or self.metrics_index

        try:
            # Check if index exists
            if self.es.indices.exists(index=index_to_create):
                logger.info(f"📁 Index {index_to_create} already exists")
                return True

            # Define mappings for metrics
            mapping = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "1s"
                },
                "mappings": {
                    "properties": {
                        "run_id": {"type": "keyword"},
                        "experiment_id": {"type": "keyword"},
                        "metric_name": {"type": "keyword"},
                        "metric_value": {"type": "float"},
                        "metric_value_json": {"type": "text"},
                        "timestamp": {
                            "type": "date",
                            "format": "strict_date_optional_time||epoch_millis"
                        },
                        "hyperparameters": {"type": "object"},
                        "model_type": {"type": "keyword"},
                        "step": {"type": "keyword"},
                        "data_source": {"type": "keyword"}
                    }
                }
            }

            # Create index
            self.es.indices.create(index=index_to_create, body=mapping, ignore=400)
            logger.info(f"✅ Created index: {index_to_create}")

            # Create system metrics index too
            if index_to_create == self.metrics_index:
                system_mapping = mapping.copy()
                system_mapping["mappings"]["properties"]["metric_name"]["type"] = "keyword"
                self.es.indices.create(index=self.system_index, body=system_mapping, ignore=400)

            return True

        except Exception as e:
            logger.error(f"❌ Failed to create index {index_to_create}: {e}")
            return False

    def log_metric(self, run_id, experiment_id, metric_name, metric_value, hyperparams=None, step="training"):
        """Log a metric to Elasticsearch"""
        if self.es is None:
            logger.warning("⚠️ Elasticsearch client not available, skipping metric logging")
            return False

        try:
            # Determine if metric_value is numeric or JSON string
            is_json = False
            numeric_value = 0.0

            if isinstance(metric_value, (int, float)):
                numeric_value = float(metric_value)
            elif isinstance(metric_value, str):
                try:
                    # Try to parse as JSON
                    parsed = json.loads(metric_value)
                    if isinstance(parsed, dict):
                        is_json = True
                        numeric_value = 0.0
                    else:
                        numeric_value = float(parsed)
                except (ValueError, TypeError):
                    numeric_value = 0.0
            else:
                numeric_value = float(metric_value)

            # Prepare document
            doc = {
                "run_id": str(run_id),
                "experiment_id": str(experiment_id),
                "metric_name": metric_name,
                "metric_value": numeric_value,
                "timestamp": datetime.utcnow(),
                "hyperparameters": hyperparams or {},
                "model_type": "KMeans",
                "step": step,
                "data_source": "mlflow_pipeline"
            }

            # Add JSON value if needed
            if is_json:
                doc["metric_value_json"] = metric_value

            # Determine index based on metric type
            if metric_name == "system_performance":
                index_to_use = self.system_index
                doc["data_source"] = "system_monitoring"
            else:
                index_to_use = self.metrics_index

            # Index the document
            response = self.es.index(index=index_to_use, body=doc)

            # Force refresh for immediate visibility
            self.es.indices.refresh(index=index_to_use)

            logger.debug(f"📨 Logged metric {metric_name}={metric_value} to {index_to_use}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to log metric {metric_name}: {e}")
            return False

    def log_run(self, run_id, experiment_id, metrics_dict, hyperparams):
        """Log all metrics from a run"""
        if self.es is None:
            return False

        success_count = 0
        for metric_name, metric_value in metrics_dict.items():
            if self.log_metric(run_id, experiment_id, metric_name, metric_value, hyperparams):
                success_count += 1

        logger.info(f"✅ Logged {success_count}/{len(metrics_dict)} metrics to Elasticsearch")
        return success_count > 0

    def search_metrics(self, query=None, size=100):
        """Search metrics in Elasticsearch"""
        if self.es is None:
            return []

        try:
            if query:
                search_body = {
                    "query": {
                        "query_string": {
                            "query": query
                        }
                    },
                    "size": size,
                    "sort": [{"timestamp": {"order": "desc"}}]
                }
            else:
                search_body = {
                    "query": {"match_all": {}},
                    "size": size,
                    "sort": [{"timestamp": {"order": "desc"}}]
                }

            response = self.es.search(index=f"{self.index_prefix}-*", body=search_body)
            return [hit["_source"] for hit in response["hits"]["hits"]]

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []

    def get_run_metrics(self, run_id):
        """Get all metrics for a specific run"""
        return self.search_metrics(query=f"run_id:{run_id}")

    def get_latest_run(self):
        """Get metrics from the latest run"""
        metrics = self.search_metrics(size=1)
        return metrics[0] if metrics else None

    def health_check(self):
        """Check Elasticsearch health"""
        if self.es is None:
            return {"status": "disconnected", "message": "Client not initialized"}

        try:
            health = self.es.cluster.health()
            indices = self.es.cat.indices(format="json")

            return {
                "status": "connected",
                "cluster_health": health,
                "indices": indices,
                "metrics_index_exists": self.es.indices.exists(index=self.metrics_index),
                "system_index_exists": self.es.indices.exists(index=self.system_index)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Test the Elasticsearch logger
    logger = ElasticsearchLogger()
    if logger.es:
        # Create index
        logger.create_index()

        # Log test metric
        logger.log_metric(
            run_id="test_run_123",
            experiment_id="test_experiment",
            metric_name="test_metric",
            metric_value=0.95,
            hyperparams={"test": True}
        )

        # Check health
        print("Health check:", logger.health_check())

        # Search metrics
        metrics = logger.search_metrics()
        print(f"Found {len(metrics)} metrics")