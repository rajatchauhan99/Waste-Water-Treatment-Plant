import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Load parameters from params.yaml
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

# Extract test_size from the parameters
test_size = params['data_ingestion']['test_size']

# Load the dataset
df = pd.read_csv("data/processed/processed.csv")

# Split the data
X = df.drop(columns="TSS_out_mg_l")
y = df["TSS_out_mg_l"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Define the folder path as a relative path
folder_path = "data/train_test_split"

# Ensure the directory exists
os.makedirs(folder_path, exist_ok=True)

# Save the split data to CSV files in the relative path
X_train.to_csv(os.path.join(folder_path, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(folder_path, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(folder_path, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(folder_path, "y_test.csv"), index=False)
