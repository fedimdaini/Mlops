"""
Complete pipeline with monitoring for Atelier 7
"""
import time
import os
from datetime import datetime
from model_pipeline import prepare_data, train_model, evaluate_model, save_model
import requests

class MonitoredPipeline:
    def __init__(self, data_path, model_path="model_pipeline.joblib"):
        """Initialize pipeline with data and model paths"""
        self.data_path = data_path
        self.model_path = model_path
        self.pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def log_pipeline_event(self, event_type, details):
        """Log pipeline events to Elasticsearch - flattened version"""
        try:
            import numpy as np
            
            def flatten_dict(d, parent_key='', sep='_'):
                """Flatten nested dictionary"""
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    
                    # Convert numpy types
                    if isinstance(v, (np.integer, np.int64, np.int32)):
                        v = int(v)
                    elif isinstance(v, (np.floating, np.float64, np.float32)):
                        v = float(v)
                    elif isinstance(v, np.ndarray):
                        v = v.tolist()
                    
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)
            
            # Flatten details
            if isinstance(details, dict):
                flat_details = flatten_dict(details)
            else:
                flat_details = {"value": str(details)}
            
            doc = {
                "@timestamp": datetime.now().isoformat(),
                "pipeline_id": self.pipeline_id,
                "event_type": event_type,
                "stage": "monitored_pipeline",
                **flat_details  # Spread flattened details at top level
            }
            
            response = requests.post(
                "http://localhost:9200/pipeline-events/_doc",
                json=doc,
                timeout=2
            )
            
            if response.status_code in [200, 201]:
                print(f"📊 Pipeline event logged: {event_type}")
            else:
                print(f"⚠️ Failed to log pipeline event: {response.text}")
                
        except Exception as e:
            print(f"⚠️ Failed to log pipeline event: {e}")
    
    def run(self):
        """Run the complete monitored pipeline"""
        self.log_pipeline_event("pipeline_started", {
            "data_path": self.data_path,
            "model_path": self.model_path
        })
        
        start_time = time.time()
        
        try:
            # Step 1: Prepare data
            self.log_pipeline_event("data_preparation_start", {})
            df, agg = prepare_data(self.data_path)
            self.log_pipeline_event("data_preparation_complete", {
                "data_shape": agg.shape,
                "duration_seconds": time.time() - start_time
            })
            
            # Step 2: Train model
            self.log_pipeline_event("model_training_start", {})
            model_start = time.time()
            model, scaler, agg_data, X_scaled = train_model(agg)
            self.log_pipeline_event("model_training_complete", {
                "duration_seconds": time.time() - model_start,
                "n_clusters": int(model.n_clusters),
                "inertia": float(model.inertia_)
            })
            
            # Step 3: Evaluate model
            self.log_pipeline_event("model_evaluation_start", {})
            metrics = evaluate_model(model, X_scaled, agg_data)
            self.log_pipeline_event("model_evaluation_complete", {
                "metrics": metrics
            })
            
            # Step 4: Save model
            self.log_pipeline_event("model_saving_start", {})
            save_model(model, scaler, agg_data, self.model_path)
            model_size = os.path.getsize(self.model_path) if os.path.exists(self.model_path) else 0
            self.log_pipeline_event("model_saving_complete", {
                "model_path": self.model_path,
                "model_size_bytes": int(model_size)
            })
            
            # Pipeline success
            total_duration = time.time() - start_time
            self.log_pipeline_event("pipeline_completed", {
                "success": True,
                "total_duration_seconds": float(total_duration),
                "final_metrics": metrics
            })
            
            print(f"✅ Pipeline completed in {total_duration:.2f} seconds")
            
        except Exception as e:
            # Pipeline failure
            self.log_pipeline_event("pipeline_failed", {
                "error": str(e),
                "duration_seconds": time.time() - start_time
            })
            raise e

if __name__ == "__main__":
    pipeline = MonitoredPipeline("cleaned_datadelevry.csv")
    pipeline.run()