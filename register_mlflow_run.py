import argparse
import json
from pathlib import Path

import mlflow

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric_path', type=str, required=True)
    parameters = parser.parse_args()

    mlflow.start_run()

    mlflow.log_param("experiment_id", parameters.metric_path)
    with open(Path(parameters.metric_path) / 'model_size.json') as f:
        model_size = json.load(f)
    with open(Path(parameters.metric_path) / 'metrics.json') as f:
        test_metrics = json.load(f)
    for k, v in model_size.items():
        mlflow.log_metric(k, v)
    for k, v in test_metrics.items():
        mlflow.log_metric(k, v)\

    mlflow.end_run()
