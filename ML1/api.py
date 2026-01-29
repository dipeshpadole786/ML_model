from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables
model = None
scaler = None
feature_names = []

def load_model_components():
    """Load model, scaler, and feature names"""
    global model, scaler, feature_names
    
    try:
        # Load model
        model_path = 'models/best_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("✓ ML Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load scaler
        scaler_path = 'models/scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded successfully")
        else:
            print("⚠ Warning: Scaler not found")
        
        # Try to get feature names from multiple sources
        possible_feature_sources = []
        
        # 1. From model attribute
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
            possible_feature_sources.append("model.feature_names_in_")
        
        # 2. From metadata
        if not feature_names:
            metadata_path = 'models/model_metadata.pkl'
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                if 'feature_names' in metadata:
                    feature_names = metadata['feature_names']
                    possible_feature_sources.append("metadata")
        
        # 3. Create default feature names based on training
        if not feature_names:
            # These are the expected features based on your dataset
            numerical_features = ['Age', 'Tenure_Months', 'Visits_Last_Year', 
                                 'Satisfaction_Score', 'Total_Bill_Amount', 'Missed_Appointments']
            categorical_features = ['Gender_Male', 'Chronic_Disease_Yes', 'Insurance_Type_Private']
            feature_names = numerical_features + categorical_features
            possible_feature_sources.append("default template")
        
        print(f"✓ Feature names loaded from: {possible_feature_sources}")
        print(f"  Total features: {len(feature_names)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model components: {str(e)}")
        return False

def prepare_features_for_prediction(input_data):
    """
    Prepare features from frontend data for model prediction
    """
    # Map frontend field names to model field names
    feature_map = {
        'patient_id': 'Patient_ID',  # Will be dropped
        'gender': 'Gender',
        'tenure_months': 'Tenure_Months',
        'visits_last_year': 'Visits_Last_Year',
        'diseases': 'Chronic_Disease',
        'insurance_type': 'Insurance_Type',
        'satisfaction_score': 'Satisfaction_Score',
        'total_billing': 'Total_Bill_Amount',
        'missed_appointments': 'Missed_Appointments'
    }
    
    # Create a dictionary with mapped names
    mapped_data = {}
    for frontend_key, model_key in feature_map.items():
        if frontend_key in input_data:
            mapped_data[model_key] = input_data[frontend_key]
        else:
            # Set defaults for missing fields
            if model_key == 'Gender':
                mapped_data[model_key] = 'Male'
            elif model_key == 'Chronic_Disease':
                mapped_data[model_key] = 'No'
            elif model_key == 'Insurance_Type':
                mapped_data[model_key] = 'Government'
            elif model_key == 'Age':  # Age not in frontend, use default
                mapped_data[model_key] = 45
            else:
                mapped_data[model_key] = 0
    
    # Add Age if not already added (frontend doesn't have it)
    if 'Age' not in mapped_data:
        mapped_data['Age'] = 45  # Default age
    
    print(f"Mapped data: {mapped_data}")
    
    # Convert to DataFrame
    df = pd.DataFrame([mapped_data])
    
    # Drop Patient_ID as it's not used in prediction
    if 'Patient_ID' in df.columns:
        df = df.drop('Patient_ID', axis=1)
    
    # One-hot encode categorical variables
    categorical_cols = ['Gender', 'Chronic_Disease', 'Insurance_Type']
    
    # Process diseases field
    if 'Chronic_Disease' in df.columns:
        df['Chronic_Disease'] = df['Chronic_Disease'].apply(
            lambda x: 'Yes' if str(x).strip() and str(x).lower() != 'none' 
                     and str(x).lower() != 'no' else 'No'
        )
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"Encoded columns: {df_encoded.columns.tolist()}")
    
    # Ensure we have all expected columns
    missing_cols = set(feature_names) - set(df_encoded.columns)
    extra_cols = set(df_encoded.columns) - set(feature_names)
    
    # Add missing columns with 0
    for col in missing_cols:
        df_encoded[col] = 0
    
    # Remove extra columns
    for col in extra_cols:
        if col != 'Patient_ID':  # Keep ID if present
            df_encoded = df_encoded.drop(col, axis=1)
    
    # Reorder columns to match expected order
    df_encoded = df_encoded[feature_names]
    
    # Scale numerical features
    if scaler is not None:
        numerical_cols = ['Age', 'Tenure_Months', 'Visits_Last_Year', 
                         'Satisfaction_Score', 'Total_Bill_Amount', 'Missed_Appointments']
        
        # Convert to numeric
        for col in numerical_cols:
            if col in df_encoded.columns:
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
        
        existing_num_cols = [col for col in numerical_cols if col in df_encoded.columns]
        
        if existing_num_cols:
            # Handle any NaN values
            df_encoded[existing_num_cols] = df_encoded[existing_num_cols].fillna(0)
            df_encoded[existing_num_cols] = scaler.transform(df_encoded[existing_num_cols])
    
    print(f"Final features for prediction: {df_encoded.columns.tolist()}")
    print(f"Data shape: {df_encoded.shape}")
    
    return df_encoded

@app.route('/')
def home():
    return jsonify({
        'message': 'Patient Churn Prediction API',
        'status': 'running',
        'model_loaded': model is not None,
        'feature_count': len(feature_names),
        'endpoints': {
            'GET /health': 'Check API health',
            'GET /features': 'Get expected features',
            'POST /predict': 'Make single prediction',
            'POST /predict-batch': 'Batch prediction from CSV'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features_loaded': len(feature_names) > 0
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Get expected feature names"""
    return jsonify({
        'features': feature_names,
        'count': len(feature_names),
        'status': 'success'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        print(f"\n📋 Received prediction request")
        print(f"Input data: {data}")
        
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Prepare features
        processed_data = prepare_features_for_prediction(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        churn_probability = float(probability[1])
        
        # Determine risk level
        if churn_probability > 0.7:
            risk_level = 'High'
            risk_color = '#ff4444'
        elif churn_probability > 0.4:
            risk_level = 'Medium'
            risk_color = '#ffaa44'
        else:
            risk_level = 'Low'
            risk_color = '#44cc44'
        
        # Generate recommendation
        if prediction == 1:
            if churn_probability > 0.7:
                recommendation = "⚠️ IMMEDIATE ACTION NEEDED: High churn risk detected. Schedule urgent follow-up with patient."
                action_items = [
                    "Contact patient within 24 hours",
                    "Schedule urgent appointment",
                    "Review treatment plan",
                    "Consider patient retention program"
                ]
            elif churn_probability > 0.4:
                recommendation = "🔄 FOLLOW-UP RECOMMENDED: Medium churn risk. Schedule appointment within 2 weeks."
                action_items = [
                    "Schedule follow-up appointment",
                    "Check patient satisfaction",
                    "Review insurance coverage",
                    "Monitor appointment attendance"
                ]
            else:
                recommendation = "📋 MONITOR: Low churn risk. Continue regular check-ups."
                action_items = [
                    "Continue current care plan",
                    "Regular check-ins",
                    "Monitor patient engagement"
                ]
        else:
            recommendation = "✅ LOW RISK: Patient is unlikely to churn. Maintain current care plan."
            action_items = [
                "Continue current treatment",
                "Regular wellness checks",
                "Maintain patient engagement"
            ]
        
        # Key factors influencing prediction
        key_factors = []
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            top_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for factor, importance in top_factors:
                if 'Missed_Appointments' in factor:
                    key_factors.append(f"Missed Appointments (impact: {importance:.2%})")
                elif 'Satisfaction_Score' in factor:
                    key_factors.append(f"Satisfaction Score (impact: {importance:.2%})")
                elif 'Tenure_Months' in factor:
                    key_factors.append(f"Tenure Duration (impact: {importance:.2%})")
                elif 'Chronic_Disease_Yes' in factor:
                    key_factors.append(f"Chronic Disease Presence (impact: {importance:.2%})")
                elif 'Insurance_Type' in factor:
                    key_factors.append(f"Insurance Type (impact: {importance:.2%})")
        
        # Prepare response
        response = {
            'patient_id': data.get('patient_id', 'Unknown'),
            'churn': 'Yes' if prediction == 1 else 'No',
            'churn_prediction': int(prediction),
            'churn_probability': churn_probability,
            'no_churn_probability': float(probability[0]),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendation': recommendation,
            'action_items': action_items,
            'key_factors': key_factors if key_factors else ["Feature importance not available"],
            'confidence': float(max(probability)),
            'status': 'success',
            'model_info': {
                'type': type(model).__name__,
                'features_used': len(feature_names)
            }
        }
        
        print(f"✅ Prediction completed: Will churn = {response['churn']}, Probability = {churn_probability:.1%}")
        
        return jsonify(response)
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Prediction error: {str(e)}")
        
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'status': 'error',
            'hint': 'Make sure all required fields are provided'
        }), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
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
        print(f"📁 Loaded {len(df)} patients from CSV")
        
        predictions = []
        errors = []
        
        for index, row in df.iterrows():
            try:
                # Convert row to dictionary
                row_data = row.to_dict()
                
                # Prepare features
                processed_data = prepare_features_for_prediction(row_data)
                
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
                    'patient_id': row_data.get('patient_id', f'Patient_{index+1}'),
                    'churn': 'Yes' if prediction == 1 else 'No',
                    'churn_probability': churn_probability,
                    'risk_level': risk_level,
                    'row_index': index + 1
                })
                
            except Exception as row_error:
                errors.append({
                    'row': index + 1,
                    'patient_id': row_data.get('patient_id', f'Patient_{index+1}'),
                    'error': str(row_error)
                })
                print(f"⚠ Error in row {index + 1}: {row_error}")
        
        # Calculate statistics
        total_patients = len(predictions)
        churn_count = sum(1 for p in predictions if p['churn'] == 'Yes')
        churn_rate = (churn_count / total_patients * 100) if total_patients > 0 else 0
        
        risk_distribution = {
            'High': sum(1 for p in predictions if p['risk_level'] == 'High'),
            'Medium': sum(1 for p in predictions if p['risk_level'] == 'Medium'),
            'Low': sum(1 for p in predictions if p['risk_level'] == 'Low')
        }
        
        return jsonify({
            'predictions': predictions,
            'summary': {
                'total_patients': total_patients,
                'predicted_to_churn': churn_count,
                'churn_rate_percent': round(churn_rate, 1),
                'risk_distribution': risk_distribution,
                'errors_count': len(errors)
            },
            'errors': errors if errors else [],
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sample-request', methods=['GET'])
def sample_request():
    """Get sample request structure"""
    return jsonify({
        'sample_request': {
            'patient_id': '123',
            'gender': 'Female',
            'tenure_months': 12,
            'visits_last_year': 5,
            'diseases': 'Diabetes',
            'insurance_type': 'Private',
            'satisfaction_score': 7,
            'total_billing': 15000,
            'missed_appointments': 2
        },
        'required_fields': [
            'patient_id',
            'gender',
            'tenure_months', 
            'visits_last_year',
            'diseases',
            'insurance_type',
            'satisfaction_score',
            'total_billing',
            'missed_appointments'
        ],
        'notes': [
            'gender: "Male" or "Female"',
            'diseases: Can be empty string for no diseases',
            'insurance_type: "Government", "Private", or "None"',
            'satisfaction_score: 1-10 scale'
        ]
    })

# Load model when starting
if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting Patient Churn Prediction API")
    print("="*60)
    
    # Load model components
    if load_model_components():
        print(f"✅ API Ready!")
        print(f"   Model: {type(model).__name__}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Scaler: {'Loaded' if scaler else 'Not loaded'}")
    else:
        print("❌ API Failed to start - Model loading failed")
        print("   Please train the model first: python main.py --mode train")
    
    print("\n" + "="*60)
    print("🌐 Available Endpoints:")
    print("  GET  /              - API Information")
    print("  GET  /health        - Health check")
    print("  GET  /features      - Get expected features")
    print("  GET  /sample-request - Sample request format")
    print("  POST /predict       - Single prediction")
    print("  POST /predict-batch - Batch prediction")
    print("="*60)
    print("📡 Server running on http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)