import joblib
import os
import pandas as pd
import numpy as np

def save_model(model, model_name, output_dir='models'):
    """Save the trained model to a file"""
    print(f"\n----- Saving Model: {model_name} -----")
    
    # Create directory if doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Save the model
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model_path

def create_sample_prediction_function(model, feature_names):
    """Create a function for making predictions on new data"""
    def predict_churn(customer_data):
        """
        Predict churn for a customer
        
        Parameters:
        customer_data (dict): Dictionary containing customer information
        
        Returns:
        dict: Prediction results with probability
        """
        # Convert input to DataFrame with correct columns
        input_df = pd.DataFrame([customer_data])
        
        # Handle missing columns
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Select only the features used by the model
        input_df = input_df[feature_names]
        
        # Make prediction
        churn_prob = model.predict_proba(input_df)[0, 1]
        churn_prediction = 1 if churn_prob >= 0.5 else 0
        
        return {
            'churn_prediction': churn_prediction,
            'churn_probability': churn_prob,
            'churn_risk': 'High' if churn_prob >= 0.7 else 'Medium' if churn_prob >= 0.3 else 'Low'
        }
    
    return predict_churn

def create_fastapi_app(model_path, feature_names):
    """Generate FastAPI code for deploying the model"""
    print("\n----- Creating FastAPI Deployment Code -----")
    
    api_code = f"""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

# Load the saved model
model = joblib.load("{model_path}")

# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn for Safaricom",
    version="1.0.0"
)

class CustomerData(BaseModel):
    \"\"\"Pydantic model for customer data input\"\"\"
    # Add required fields here based on your model features
    # Below are examples - adjust as needed
    gender: int = Field(..., description="Customer gender (0=Female, 1=Male)")
    tenure: float = Field(..., description="Number of months the customer has been with the company")
    MonthlyCharges: float = Field(..., description="Monthly charges")
    # Add more fields as needed

    class Config:
        schema_extra = {{
            "example": {{
                "gender": 1,
                "tenure": 0.5,  # Scaled value
                "MonthlyCharges": 0.65,  # Scaled value
                # Add more example values
            }}
        }}

@app.get("/")
def read_root():
    \"\"\"Root endpoint\"\"\"
    return {{"message": "Welcome to the Customer Churn Prediction API"}}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    \"\"\"Predict customer churn\"\"\"
    try:
        # Convert input to DataFrame
        customer_dict = customer.dict()
        input_df = pd.DataFrame([customer_dict])
        
        # Make prediction
        churn_prob = model.predict_proba(input_df)[0, 1]
        churn_prediction = 1 if churn_prob >= 0.5 else 0
        
        return {{
            "churn_prediction": bool(churn_prediction),
            "churn_probability": float(churn_prob),
            "churn_risk": "High" if churn_prob >= 0.7 else "Medium" if churn_prob >= 0.3 else "Low"
        }}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn app:app --reload
"""
    
    # Save the API code to a file
    with open('app.py', 'w') as f:
        f.write(api_code)
    
    print("FastAPI code saved to 'app.py'")
    print("Run the API with: uvicorn app:app --reload")
    
    return api_code

def test_saved_model(model_path, X_test):
    """Test loading and using the saved model"""
    print(f"\n----- Testing Saved Model -----")
    
    # Load the model
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully")
    
    # Make a test prediction
    sample = X_test.iloc[0:1]
    prediction = loaded_model.predict(sample)
    probability = loaded_model.predict_proba(sample)[0, 1]
    
    print(f"Test prediction successful:")
    print(f"  Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    print(f"  Probability of churn: {probability:.4f}")
    
    return loaded_model

if __name__ == "__main__":
    # This is just for testing the module
    from data_preparation import load_data, preprocess_data, split_dataset
    from model_training import train_models
    
    print("Loading and preprocessing data...")
    df = load_data("Telco-Customer-Churn.csv")
    processed_df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(processed_df)
    
    print("Training a simple model for testing...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    print("Testing model deployment...")
    model_path = save_model(model, "test_model")
    test_saved_model(model_path, X_test)
    create_fastapi_app(model_path, X_train.columns.tolist())
    
    print("Model deployment testing completed successfully!")