import joblib
import pandas as pd

# Load model
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
metadata = joblib.load('models/model_metadata.pkl')

print("="*60)
print("MODEL INFORMATION")
print("="*60)

print(f"Model type: {type(model).__name__}")
print(f"Model name from metadata: {metadata.get('model_name', 'Unknown')}")

# Get feature names
if hasattr(model, 'feature_names_in_'):
    features = model.feature_names_in_.tolist()
    print(f"\nModel expects {len(features)} features:")
    for i, feature in enumerate(features, 1):
        print(f"  {i:2}. {feature}")
elif 'feature_names' in metadata:
    features = metadata['feature_names']
    print(f"\nMetadata has {len(features)} features:")
    for i, feature in enumerate(features[:10], 1):  # Show first 10
        print(f"  {i:2}. {feature}")
    if len(features) > 10:
        print(f"  ... and {len(features)-10} more features")

print(f"\nScaler type: {type(scaler).__name__}")
print(f"Scaler has {len(scaler.mean_)} features to scale")

print("\n" + "="*60)
print("SAMPLE PATIENT DATA TRANSFORMATION")
print("="*60)

# Create sample patient data matching your test
sample_data = {
    'Patient_ID': 89,
    'Age': 56,
    'Gender': 'Female',
    'Tenure_Months': 2,
    'Visits_Last_Year': 3,
    'Chronic_Disease': 'No',
    'Insurance_Type': 'Government',
    'Satisfaction_Score': 2.5,
    'Total_Bill_Amount': 12252.96,
    'Missed_Appointments': 9
}

# Show what features are created
df = pd.DataFrame([sample_data])
print("Original data:")
print(df.drop('Patient_ID', axis=1).to_string())

# One-hot encode
df_encoded = pd.get_dummies(df.drop('Patient_ID', axis=1), 
                           columns=['Gender', 'Chronic_Disease', 'Insurance_Type'], 
                           drop_first=True)

print(f"\nAfter one-hot encoding ({len(df_encoded.columns)} features):")
print(df_encoded.columns.tolist())

# Scale numerical features
numerical_cols = ['Age', 'Tenure_Months', 'Visits_Last_Year', 
                 'Satisfaction_Score', 'Total_Bill_Amount', 'Missed_Appointments']
existing_num_cols = [col for col in numerical_cols if col in df_encoded.columns]

df_scaled = df_encoded.copy()
if existing_num_cols:
    df_scaled[existing_num_cols] = scaler.transform(df_encoded[existing_num_cols])
    
print(f"\nAfter scaling ({len(df_scaled.columns)} features):")
print(df_scaled.columns.tolist())

# Make prediction
prediction = model.predict(df_scaled)[0]
probability = model.predict_proba(df_scaled)[0]

print(f"\nPrediction result:")
print(f"  Will churn: {'Yes' if prediction == 1 else 'No'}")
print(f"  Churn probability: {probability[1]:.1%}")
print(f"  No churn probability: {probability[0]:.1%}")

print("\n" + "="*60)