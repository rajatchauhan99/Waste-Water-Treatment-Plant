import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import yaml

# Load parameters from params.yaml
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

# Define the folder paths
train_test_split_folder = r"C:\Users\rajat.chauhan\Downloads\Data science\Waste Water Treatment Plant\data\train_test_split"
features_folder = r"C:\Users\rajat.chauhan\Downloads\Data science\Waste Water Treatment Plant\data\features"
models_folder = r"C:\Users\rajat.chauhan\Downloads\Data science\Waste Water Treatment Plant\models"

# Ensure the directories exist
os.makedirs(features_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

# Load the train-test split data
X_train = pd.read_csv(os.path.join(train_test_split_folder, "X_train.csv"))
X_test = pd.read_csv(os.path.join(train_test_split_folder, "X_test.csv"))
y_train = pd.read_csv(os.path.join(train_test_split_folder, "y_train.csv"))
y_test = pd.read_csv(os.path.join(train_test_split_folder, "y_test.csv"))

# Scaling
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Save the scaled data to CSV files
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(os.path.join(features_folder, "X_train_scaled.csv"), index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(os.path.join(features_folder, "X_test_scaled.csv"), index=False)
pd.DataFrame(y_train_scaled, columns=["TSS_out_mg_l"]).to_csv(os.path.join(features_folder, "y_train_scaled.csv"), index=False)
pd.DataFrame(y_test_scaled, columns=["TSS_out_mg_l"]).to_csv(os.path.join(features_folder, "y_test_scaled.csv"), index=False)

# Save the scalers using pickle
with open(os.path.join(models_folder, "scaler_X.pkl"), 'wb') as file:
    pickle.dump(scaler_X, file)

with open(os.path.join(models_folder, "scaler_y.pkl"), 'wb') as file:
    pickle.dump(scaler_y, file)
