from flask import Flask, request, render_template
import pandas as pd
import mlflow
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Set tracking URI to the correct path (local or remote)
mlflow.set_tracking_uri("https://dagshub.com/rajatchauhan99/Waste-Water-Treatment-Plant.mlflow")

# Load the MLflow model from the specific run path
logged_model = 'runs:/bc280f64216142d6afa39ca8c9f6339f/random_forest_model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Define feature names for the form
feature_names = [
    "Temp_C", "pH", "TSS_in_mg_l", "BOD_in_mg_l", "COD_in_mg_l", "BOD_out_mg_l", "COD_out_mg_l", "DO_out_mg_l"
]

# Paths to scalers
scalers_folder = r"C:\Users\rajat.chauhan\Downloads\Data science\Waste Water Treatment Plant\artifacts\scalers"
scaler_X_path = os.path.join(scalers_folder, "scaler_X.pkl")
scaler_y_path = os.path.join(scalers_folder, "scaler_y.pkl")

# Load the pre-saved scalers
with open(scaler_X_path, 'rb') as file:
    scaler_X = pickle.load(file)

with open(scaler_y_path, 'rb') as file:
    scaler_y = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        input_data = [float(request.form[feature]) for feature in feature_names]
        
        # Create a DataFrame for the model input (single row)
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Scale the input features using the pre-loaded scaler
        scaled_input = scaler_X.transform(input_df)
        
        # Convert the scaled input back to a DataFrame for MLflow model prediction
        scaled_input_df = pd.DataFrame(scaled_input, columns=feature_names)
        
        # Make a prediction using the loaded model
        prediction_scaled = loaded_model.predict(scaled_input_df)
        
        # Inverse transform the scaled prediction to get the actual target value
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
        
        # Display the prediction result
        return render_template('index.html', prediction_text=f'Predicted TSS_out_mg_l: {prediction[0][0]:.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
