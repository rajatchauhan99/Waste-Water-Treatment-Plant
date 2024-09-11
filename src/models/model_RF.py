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

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

# Initialize DagsHub and MLflow
mlflow.set_tracking_uri("https://dagshub.com/rajatchauhan99/Waste-Water-Treatment-Plant.mlflow")
dagshub.init(repo_owner='rajatchauhan99', repo_name='Waste-Water-Treatment-Plant', mlflow=True)

# Set the experiment name
mlflow.set_experiment("Random Forest")

# Train GridSearchCV
grid_search.fit(X_train_scaled, y_train_scaled)

# Iterate through each parameter combination and log results
for i, (params, mean_score, scores) in enumerate(zip(
    grid_search.cv_results_['params'],
    grid_search.cv_results_['mean_test_score'],
    grid_search.cv_results_['std_test_score']
)):
    with mlflow.start_run(run_name=f"Run_{i+1}"):
        # Log Grid Search parameters
        mlflow.log_params(params)
        
        # Train the model with current parameters
        best_rf = RandomForestRegressor(random_state=42, **params)
        best_rf.fit(X_train_scaled, y_train_scaled)

        # Predict and calculate metrics
        y_train_pred = best_rf.predict(X_train_scaled)
        y_test_pred = best_rf.predict(X_test_scaled)

        train_mse = mean_squared_error(y_train_scaled, y_train_pred)
        test_mse = mean_squared_error(y_test_scaled, y_test_pred)
        train_r2 = r2_score(y_train_scaled, y_train_pred)
        test_r2 = r2_score(y_test_scaled, y_test_pred)

        # Log metrics
        mlflow.log_metric("Train Mean Squared Error", train_mse)
        mlflow.log_metric("Test Mean Squared Error", test_mse)
        mlflow.log_metric("Train R² Score", train_r2)
        mlflow.log_metric("Test R² Score", test_r2)
        mlflow.log_metric("Mean Test Score", mean_score)
        mlflow.log_metric("Std Test Score", scores)

        # Log the model to DagsHub
        mlflow.sklearn.log_model(best_rf, "best_rf_model")

        # Save the model locally
        local_model_path = os.path.join(models_folder, f"best_rf_model_run_{i+1}.pkl")
        with open(local_model_path, 'wb') as model_file:
            pickle.dump(best_rf, model_file)
        mlflow.log_artifact(local_model_path)

        # Optionally, print metrics
        print(f"Run {i+1}:")
        print(f"Train Mean Squared Error: {train_mse}")
        print(f"Test Mean Squared Error: {test_mse}")
        print(f"Train R² Score: {train_r2}")
        print(f"Test R² Score: {test_r2}")
