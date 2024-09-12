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

# Load parameters from params.yaml
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

# Define the folder paths
features_folder = r"C:\Users\rajat.chauhan\Downloads\Data science\Waste Water Treatment Plant\data\features"
models_folder = r"C:\Users\rajat.chauhan\Downloads\Data science\Waste Water Treatment Plant\artifacts\models_RF"

# Ensure the models folder exists
os.makedirs(models_folder, exist_ok=True)

# Load the scaled data
X_train_scaled = pd.read_csv(os.path.join(features_folder, "X_train_scaled.csv"))
X_test_scaled = pd.read_csv(os.path.join(features_folder, "X_test_scaled.csv"))
y_train_scaled = pd.read_csv(os.path.join(features_folder, "y_train_scaled.csv"))
y_test_scaled = pd.read_csv(os.path.join(features_folder, "y_test_scaled.csv"))

# Extract the target column
y_train_scaled = y_train_scaled.values.ravel()
y_test_scaled = y_test_scaled.values.ravel()

# Define the RandomForest model
rf = RandomForestRegressor(random_state=42)

# Define the parameter grid for Grid Search with more balanced complexity
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

# Initialize DagsHub and MLflow
mlflow.set_tracking_uri("https://dagshub.com/rajatchauhan99/Waste-Water-Treatment-Plant.mlflow")
dagshub.init(repo_owner='rajatchauhan99', repo_name='Waste-Water-Treatment-Plant', mlflow=True)

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
    
    # Log model parameters, and performance metrics on both train and test sets
    mlflow.log_params(best_params)
    
    # Logging training metrics
    mlflow.log_metric('mse_train', mse_train)
    mlflow.log_metric('r2_train', r2_train)
    
    # Logging test metrics
    mlflow.log_metric('mse_test', mse_test)
    mlflow.log_metric('r2_test', r2_test)
    
    # Save the best model locally
    model_path = os.path.join(models_folder, "best_rf_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_rf_model, f)
    
    # Log the model to MLflow
    mlflow.sklearn.log_model(best_rf_model, "random_forest_model")
    
    # Output the results
    print(f"Best Model Parameters: {best_params}")
    print(f"Train MSE: {mse_train}, Train R2: {r2_train}")
    print(f"Test MSE: {mse_test}, Test R2: {r2_test}")
