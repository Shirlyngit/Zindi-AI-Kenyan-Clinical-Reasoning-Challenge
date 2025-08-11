
import os
import mlflow

class MLflowTracker:
    def __init__(self, experiment_name="Clinical_Summarization"):
        """
        Initialize MLflow tracking for summarization project.

        Args:
            experiment_name (str): Name of the MLflow experiment
        """
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(experiment_name)
        self.run = None

    def start_run(self, run_name=None, tags=None):
        """
        Start a new MLflow run.

        Args:
            run_name (str): Optional name for the run
            tags (dict): Optional dictionary of tags
        """
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        print(f"🚀 Started MLflow run: {self.run.info.run_id}")

    def log_params(self, params: dict):
        """
        Log parameters to MLflow.
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step=None):
        """
        Log metrics to MLflow.
        """
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, file_path: str):
        """
        Log a file as an artifact in MLflow.
        """
        mlflow.log_artifact(file_path)

    def end_run(self):
        """
        End the MLflow run.
        """
        mlflow.end_run()
        print("✅ MLflow run ended.")



# Example usage
if __name__ == "__main__":
    tracker = MLflowTracker()

    tracker.start_run(run_name="test_run", tags={"mode": "gemini"})

    tracker.log_params({"model": "gemini-1.5-pro", "samples": 20})
    tracker.log_metrics({"rougeL": 0.78, "rouge1": 0.82})

    tracker.log_artifact("evaluation_results.csv")

    tracker.end_run()
