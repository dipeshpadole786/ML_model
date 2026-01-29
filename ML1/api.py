from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load model and scaler
model = None
scaler = None
feature_names = None

def load_model():
    """Load the trained model and scaler"""
    global model, scaler, feature_names
    
    try:
        # Load model
        model_path = 'models/best_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("✓ ML Model loaded successfully")
        else:
            print(f"✗ Model file not found: {model_path}")
            # Try to load any model file
            for file in os.listdir('models'):
                if file.endswith('.pkl') and 'model' in file.lower():
                    model = joblib.load(f'models/{file}')
                    print(f"✓ Loaded model from: models/{file}")
                    break
        
        # Load scaler
        scaler_path = 'models/scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded successfully")
        
        # Load feature names from metadata
        metadata_path = 'models/model_metadata.pkl'
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            feature_names = metadata.get('feature_names', [])
            print("✓ Feature names loaded")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False

# Load model when server starts
load_model()

def preprocess_input(data):
    """
    Preprocess input data to match model requirements
    """
    # Create DataFrame from input
    df = pd.DataFrame([data])
    
    # Map frontend field names to backend field names
    mapping = {
        'patient_id': 'Patient_ID',
        'gender': 'Gender',
        'tenure_months': 'Tenure_Months',
        'visits_last_year': 'Visits_Last_Year',
        'diseases': 'Chronic_Disease',
        'insurance_type': 'Insurance_Type',
        'satisfaction_score': 'Satisfaction_Score',
        'total_billing': 'Total_Bill_Amount',
        'missed_appointments': 'Missed_Appointments'
    }
    
    # Rename columns
    df.rename(columns=mapping, inplace=True)
    
    # Process diseases field (convert to Yes/No for Chronic_Disease)
    if 'Chronic_Disease' in df.columns:
        df['Chronic_Disease'] = df['Chronic_Disease'].apply(
            lambda x: 'Yes' if str(x).strip() and str(x).lower() != 'none' else 'No'
        )
    
    # Add Age column (since your model expects it, but frontend doesn't have it)
    # You can set a default or calculate based on some logic
    if 'Age' not in df.columns:
        df['Age'] = 45  # Default age, or you can add age to your frontend
    
    # One-hot encode categorical variables
    categorical_cols = ['Gender', 'Chronic_Disease', 'Insurance_Type']
    
    # Ensure all categorical columns exist
    for col in categorical_cols:
        if col not in df.columns:
            if col == 'Gender':
                df[col] = 'Male'
            elif col == 'Chronic_Disease':
                df[col] = 'No'
            elif col == 'Insurance_Type':
                df[col] = 'Private'
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Scale numerical features if scaler exists
    if scaler is not None:
        numerical_cols = ['Age', 'Tenure_Months', 'Visits_Last_Year', 
                         'Satisfaction_Score', 'Total_Bill_Amount', 'Missed_Appointments']
        
        existing_num_cols = [col for col in numerical_cols if col in df_encoded.columns]
        
        if existing_num_cols:
            df_encoded[existing_num_cols] = scaler.transform(df_encoded[existing_num_cols])
    
    return df_encoded

@app.route('/')
def home():
    return jsonify({
        'message': 'ML Model API for Patient Churn Prediction',
        'status': 'running',
        'model_loaded': model is not None
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        print(f"Received data: {data}")
        
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Preprocess input data
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        # Determine risk level
        churn_probability = float(probability[1])
        if churn_probability > 0.7:
            risk_level = 'High'
        elif churn_probability > 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Generate recommendation
        if prediction == 1:
            if churn_probability > 0.7:
                recommendation = "Immediate intervention needed - High churn risk"
            elif churn_probability > 0.4:
                recommendation = "Schedule follow-up - Medium churn risk"
            else:
                recommendation = "Monitor patient - Low churn risk"
        else:
            recommendation = "Continue current care plan"
        
        # Prepare response
        response = {
            'patient_id': data.get('patient_id', 'Unknown'),
            'churn': 'Yes' if prediction == 1 else 'No',
            'churn_prediction': int(prediction),
            'churn_probability': churn_probability,
            'no_churn_probability': float(probability[0]),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'confidence': float(max(probability)),
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Prediction error: {str(e)}")
        print(f"Error details: {error_details}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
        # Get CSV file from request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Check required columns
        required_cols = ['patient_id', 'gender', 'tenure_months', 'visits_last_year',
                        'diseases', 'insurance_type', 'satisfaction_score',
                        'total_billing', 'missed_appointments']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'}), 400
        
        predictions = []
        
        for _, row in df.iterrows():
            # Preprocess each row
            processed_data = preprocess_input(row.to_dict())
            
            # Make prediction
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0]
            
            churn_probability = float(probability[1])
            
            # Determine risk level
            if churn_probability > 0.7:
                risk_level = 'High'
            elif churn_probability > 0.4:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            predictions.append({
                'patient_id': row['patient_id'],
                'churn': 'Yes' if prediction == 1 else 'No',
                'churn_probability': churn_probability,
                'risk_level': risk_level
            })
        
        return jsonify({
            'predictions': predictions,
            'total': len(predictions),
            'churn_count': sum(1 for p in predictions if p['churn'] == 'Yes'),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """Reload the model (useful after retraining)"""
    try:
        success = load_model()
        if success:
            return jsonify({'message': 'Model reloaded successfully', 'status': 'success'})
        else:
            return jsonify({'error': 'Failed to reload model'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting ML Model API Server...")
    print("="*50)
    print(f"Model loaded: {model is not None}")
    print(f"Scaler loaded: {scaler is not None}")
    print("="*50)
    print("API Endpoints:")
    print("  GET  /           - Home page")
    print("  GET  /health     - Health check")
    print("  POST /predict    - Single prediction")
    print("  POST /predict-batch - Batch prediction from CSV")
    print("  POST /reload-model - Reload model")
    print("="*50)
    print("Server running on http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    app.run(debug=True, host='0.0.0.0', port=5000)