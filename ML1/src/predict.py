import pandas as pd
import numpy as np
import joblib
import os

class ChurnPredictor:
    def __init__(self, model_path='models/best_model.pkl', scaler_path='models/scaler.pkl'):
        """
        Initialize predictor with trained model
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        
        # Load model
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"✓ Model loaded from {model_path}")
            except:
                print(f"✗ Error loading model from {model_path}")
        
        # Load scaler
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Scaler loaded from {scaler_path}")
            except:
                print(f"✗ Error loading scaler from {scaler_path}")
        
        # Load metadata
        metadata_path = 'models/model_metadata.pkl'
        if os.path.exists(metadata_path):
            try:
                self.metadata = joblib.load(metadata_path)
                self.feature_names = self.metadata.get('feature_names', [])
                print(f"✓ Model metadata loaded")
                print(f"  Model: {self.metadata.get('model_name', 'Unknown')}")
                print(f"  Trained: {self.metadata.get('training_date', 'Unknown')}")
            except:
                print(f"✗ Error loading metadata")
        
        if self.model is None:
            print("⚠ Warning: No model loaded. Please train a model first.")
    
    def predict_single(self, patient_data):
        """
        Predict churn for a single patient
        """
        if self.model is None:
            print("✗ Cannot predict: Model not loaded")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Preprocess
        processed = self._preprocess_data(df)
        
        if processed is None:
            return None
        
        # Make prediction
        prediction = self.model.predict(processed)[0]
        probability = self.model.predict_proba(processed)[0]
        
        # Prepare result
        result = {
            'patient_id': patient_data.get('Patient_ID', 'Unknown'),
            'churn_prediction': int(prediction),
            'churn_probability': float(probability[1]),
            'no_churn_probability': float(probability[0]),
            'risk_level': self._get_risk_level(probability[1]),
            'recommendation': self._get_recommendation(prediction, probability[1])
        }
        
        return result
    
    def predict_csv(self, csv_path):
        """
        Predict churn for patients in a CSV file
        """
        if self.model is None:
            print("✗ Cannot predict: Model not loaded")
            return None
        
        if not os.path.exists(csv_path):
            print(f"✗ File not found: {csv_path}")
            return None
        
        try:
            # Load data
            df = pd.read_csv(csv_path)
            print(f"✓ Loaded {len(df)} patients from {csv_path}")
            
            # Check required columns
            required_cols = ['Age', 'Gender', 'Tenure_Months', 'Visits_Last_Year', 
                           'Chronic_Disease', 'Insurance_Type', 'Satisfaction_Score',
                           'Total_Bill_Amount', 'Missed_Appointments']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠ Missing columns: {missing_cols}")
                print("  Adding default values...")
                for col in missing_cols:
                    if col == 'Gender':
                        df[col] = 'Female'
                    elif col == 'Chronic_Disease':
                        df[col] = 'No'
                    elif col == 'Insurance_Type':
                        df[col] = 'Private'
                    else:
                        df[col] = 0
            
            # Preprocess
            processed = self._preprocess_data(df)
            
            if processed is None:
                return None
            
            # Make predictions
            predictions = self.model.predict(processed)
            probabilities = self.model.predict_proba(processed)
            
            # Add predictions to original dataframe
            results_df = df.copy()
            results_df['Churn_Prediction'] = predictions
            results_df['Churn_Probability'] = probabilities[:, 1]
            results_df['Risk_Level'] = results_df['Churn_Probability'].apply(self._get_risk_level)
            
            # Add recommendations
            results_df['Recommendation'] = results_df.apply(
                lambda row: self._get_recommendation(row['Churn_Prediction'], row['Churn_Probability']), 
                axis=1
            )
            
            # Summary
            churn_count = results_df['Churn_Prediction'].sum()
            total_count = len(results_df)
            churn_rate = (churn_count / total_count) * 100
            
            print(f"\n📊 Prediction Summary:")
            print(f"   Total Patients: {total_count}")
            print(f"   Predicted to Churn: {churn_count} ({churn_rate:.1f}%)")
            print(f"   High Risk: {len(results_df[results_df['Risk_Level'] == 'High'])}")
            print(f"   Medium Risk: {len(results_df[results_df['Risk_Level'] == 'Medium'])}")
            print(f"   Low Risk: {len(results_df[results_df['Risk_Level'] == 'Low'])}")
            
            return results_df
            
        except Exception as e:
            print(f"✗ Error processing CSV: {str(e)}")
            return None
    
    def _preprocess_data(self, df):
        """
        Preprocess input data
        """
        df_processed = df.copy()
        
        # Drop Patient_ID if present (not used in prediction)
        if 'Patient_ID' in df_processed.columns:
            df_processed = df_processed.drop('Patient_ID', axis=1)
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Chronic_Disease', 'Insurance_Type']
        
        # Ensure categorical columns exist
        for col in categorical_cols:
            if col not in df_processed.columns:
                if col == 'Gender':
                    df_processed[col] = 'Female'
                elif col == 'Chronic_Disease':
                    df_processed[col] = 'No'
                elif col == 'Insurance_Type':
                    df_processed[col] = 'Private'
        
        # One-hot encode
        df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
        
        # Ensure all expected features are present
        if self.feature_names:
            missing_features = set(self.feature_names) - set(df_encoded.columns)
            extra_features = set(df_encoded.columns) - set(self.feature_names)
            
            # Add missing features with 0
            for feature in missing_features:
                df_encoded[feature] = 0
            
            # Remove extra features
            for feature in extra_features:
                if feature not in ['Churn_Prediction', 'Churn_Probability', 'Risk_Level', 'Recommendation']:
                    df_encoded = df_encoded.drop(feature, axis=1)
            
            # Ensure correct order
            df_encoded = df_encoded[self.feature_names]
        
        # Scale numerical features
        if self.scaler is not None:
            numerical_cols = ['Age', 'Tenure_Months', 'Visits_Last_Year', 
                            'Satisfaction_Score', 'Total_Bill_Amount', 'Missed_Appointments']
            
            existing_num_cols = [col for col in numerical_cols if col in df_encoded.columns]
            
            if existing_num_cols:
                df_encoded[existing_num_cols] = self.scaler.transform(df_encoded[existing_num_cols])
        
        return df_encoded
    
    def _get_risk_level(self, probability):
        """
        Determine risk level based on probability
        """
        if probability > 0.7:
            return 'High'
        elif probability > 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_recommendation(self, prediction, probability):
        """
        Generate recommendation based on prediction
        """
        if prediction == 1:
            if probability > 0.7:
                return "Immediate intervention needed - High churn risk"
            elif probability > 0.4:
                return "Schedule follow-up - Medium churn risk"
            else:
                return "Monitor patient - Low churn risk"
        else:
            return "Continue current care plan"
    
    def get_model_info(self):
        """
        Get information about the loaded model
        """
        if self.metadata:
            return self.metadata
        return {"status": "No model loaded"}