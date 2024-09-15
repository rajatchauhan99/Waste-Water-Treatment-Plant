import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import yaml
import dagshub
import pickle  # for saving the model locally
from mlflow.models.signature import infer_signature

# Load parameters from params.yaml
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

# Define the folder paths in an OS-independent way
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Go up one level from src
features_folder = os.path.join(base_dir, "data", "features")
models_folder = os.path.join(base_dir, "artifacts", "models_RF")

# Ensure the models folder exists
os.makedirs(models_folder, exist_ok=True)

# Load the scaled data
X_train_scaled = pd.read_csv(os.path.join(features_folder, "X_train_scaled.csv"))
X_test_scaled = pd.read_csv(os.path.join(features_folder, "X_test_scaled.csv"))
y_train_scaled = pd.read_csv(os.path.join(features_folder, "y_train_scaled.csv"))
y_test_scaled = pd.read_csv(os.path.join(features_folder, "y_test_scaled.csv"))

# Extract the target column (reshape to 1D array)
y_train_scaled = y_train_scaled.values.ravel()
y_test_scaled = y_test_scaled.values.ravel()

# Define the RandomForest model
rf = RandomForestRegressor(random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV with both scoring metrics
grid_search = GridSearchCV(
    estimator=rf, 
    param_grid=param_grid, 
    cv=5, 
    n_jobs=-1, 
    scoring=['neg_mean_squared_error', 'r2'], 
    refit='neg_mean_squared_error'
)

# Set up DAGSHUB credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = 'rajatchauhan99'
repo_name = 'Waste-Water-Treatment-Plant'

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# Set the experiment name
mlflow.set_experiment("Random Forest")

# Start an MLflow run to log metrics and parameters
with mlflow.start_run():
    # Train GridSearchCV
    grid_search.fit(X_train_scaled, y_train_scaled)
    
    # Fetch best model and parameters
    best_rf_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Predict on train and test data
    y_train_pred = best_rf_model.predict(X_train_scaled)
    y_test_pred = best_rf_model.predict(X_test_scaled)
    
    # Evaluate performance on train data
    mse_train = mean_squared_error(y_train_scaled, y_train_pred)
    r2_train = r2_score(y_train_scaled, y_train_pred)
    
    # Evaluate performance on test data
    mse_test = mean_squared_error(y_test_scaled, y_test_pred)
    r2_test = r2_score(y_test_scaled, y_test_pred)
    
    # Log model parameters and performance metrics
    mlflow.log_params(best_params)
    mlflow.log_metric('mse_train', mse_train)
    mlflow.log_metric('r2_train', r2_train)
    mlflow.log_metric('mse_test', mse_test)
    mlflow.log_metric('r2_test', r2_test)

    # Infer signature (input-output schema) for the model
    signature = infer_signature(X_train_scaled, y_train_pred)

    # Save the best model locally
    model_path = os.path.join(models_folder, "best_rf_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_rf_model, f)
    
    # Log the model to MLflow
    mlflow.sklearn.log_model(
        sk_model=best_rf_model,
        artifact_path="random_forest_model",
        signature=signature
    )
    
    # Register the model with the model registry in MLflow
    mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/random_forest_model",
        name="Random Forest Model"
    )
    
    # Output the results
    print(f"Best Model Parameters: {best_params}")
    print(f"Train MSE: {mse_train}, Train R2: {r2_train}")
    print(f"Test MSE: {mse_test}, Test R2: {r2_test}")
