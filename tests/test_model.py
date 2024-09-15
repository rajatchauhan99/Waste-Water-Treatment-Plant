import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "rajatchauhan99"
        repo_name = "Waste-Water-Treatment-Plant"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the latest model from MLflow model registry
        cls.new_model_name = "random_forest_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.sklearn.load_model(cls.new_model_uri)

        # Load test data
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        features_folder = os.path.join(base_dir, "data", "features")
        cls.X_test_scaled = pd.read_csv(os.path.join(features_folder, "X_test_scaled.csv"))
        cls.y_test_scaled = pd.read_csv(os.path.join(features_folder, "y_test_scaled.csv")).values.ravel()

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        """Test that the model is loaded properly from MLflow."""
        self.assertIsNotNone(self.new_model, "Model should be loaded")

    def test_model_signature(self):
        """Test the model's input-output signature."""
        # Predict on test data to verify input-output signature
        y_test_pred = self.new_model.predict(self.X_test_scaled)
        self.assertEqual(len(self.y_test_scaled), len(y_test_pred), "Output size should match test labels")

    def test_model_performance(self):
        """Test model performance on test data."""
        # Predict using the loaded model
        y_test_pred = self.new_model.predict(self.X_test_scaled)

        # Evaluate model performance
        mse_test = mean_squared_error(self.y_test_scaled, y_test_pred)
        r2_test = r2_score(self.y_test_scaled, y_test_pred)

        # Define expected thresholds
        expected_mse = 0.1  # Example thresholds
        expected_r2 = 0.6

        # Check that the performance meets the expected thresholds
        self.assertLessEqual(mse_test, expected_mse, f'MSE should be less than {expected_mse}')
        self.assertGreaterEqual(r2_test, expected_r2, f'R2 should be greater than {expected_r2}')

if __name__ == "__main__":
    unittest.main()
