import psutil
import time
import requests
from datetime import datetime


def monitor_system():
    """Monitor system resources and send to Elasticsearch"""
    while True:
        try:
            # Collect system metrics
            metrics = {
                "@timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "network_io": {
                    "bytes_sent": psutil.net_io_counters().bytes_sent,
                    "bytes_recv": psutil.net_io_counters().bytes_recv
                },
                "process_count": len(psutil.pids())
            }

            # Send to Elasticsearch
            response = requests.post(
                "http://localhost:9200/system-metrics/_doc",
                json=metrics,
                timeout=5
            )

            if response.status_code in [200, 201]:
                print(f"📊 System metrics sent: CPU={metrics['cpu_percent']}%, MEM={metrics['memory_percent']}%")

        except Exception as e:
            print(f"System monitoring error: {e}")

        time.sleep(60)  # Collect every minute


if __name__ == "__main__":
    monitor_system()