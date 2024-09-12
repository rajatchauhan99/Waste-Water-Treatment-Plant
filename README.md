
# Project: Waste Water Treatment Plant Analysis

This project aims to build a predictive model for waste water treatment plants. The project uses various stages of data processing, including data ingestion, feature engineering, and model training using Random Forest (RF). The goal is to track the various runs and experiment results using MLflow, ensuring reproducibility and effective performance tracking.

## Project Structure

The project is organized into multiple stages. Each stage runs a script responsible for specific tasks in the machine learning pipeline, using DVC (Data Version Control) to manage data dependencies and outputs.

### Stages

1. **Data Ingestion**
   - **Script:** `src/data/data_ingestion.py`
   - **Description:** Ingests raw data from the `data/processed` folder, processes it, and splits the data into training and testing sets.
   - **Inputs:** 
     - `data/processed` (processed data folder)
     - `src/data/data_ingestion.py` (data ingestion script)
   - **Outputs:** 
     - `data/train_test_split` (folder containing the split datasets)

2. **Feature Engineering**
   - **Script:** `src/data/feature_engineering.py`
   - **Description:** Performs feature engineering on the training and testing datasets, and outputs engineered features along with scalers.
   - **Inputs:** 
     - `data/train_test_split` (train/test data)
     - `src/data/feature_engineering.py` (feature engineering script)
   - **Outputs:** 
     - `data/features` (folder containing engineered features)
     - `artifacts/scalers` (scalers used for feature scaling)

3. **Model Training (Random Forest)**
   - **Script:** `src/models/model_RF.py`
   - **Description:** Trains a Random Forest model using the engineered features and saves the trained model.
   - **Inputs:** 
     - `data/features` (engineered features)
     - `src/models/model_RF.py` (Random Forest model script)
   - **Outputs:** 
     - `artifacts/models_RF` (folder containing the trained Random Forest models)

## How to Run the Project

To execute the project stages, use the following commands:

1. **Data Ingestion:**
   ```bash
   dvc repro data_ingestion
   ```

2. **Feature Engineering:**
   ```bash
   dvc repro feature_engineering
   ```

3. **Model Training (Random Forest):**
   ```bash
   dvc repro model_RF
   ```

## Dependencies

Make sure to install the required dependencies before running the scripts. You can find the list of dependencies in the `requirements.txt` file.

To install the dependencies, run:
```bash
pip install -r requirements.txt
```

## Tracking with MLflow

This project uses MLflow for tracking experiments. The `mlruns/` folder is used to store run logs, parameters, and results for each model training session.

You can start the MLflow server locally by running:
```bash
mlflow ui
```

This will allow you to track and visualize experiment runs.

## Project Workflow

1. Data is ingested, processed, and split into train and test sets.
2. Feature engineering is performed, and the output is stored in the `data/features` folder.
3. The Random Forest model is trained using the engineered features, and the model artifacts are stored in the `artifacts/models_RF` folder.

## Contact

For any issues or questions about this project, feel free to reach out to the maintainer.
```

This `README.md` file outlines the steps in your project and provides instructions for running each stage.
