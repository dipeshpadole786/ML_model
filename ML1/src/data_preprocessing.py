import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(filepath='data/patient_churn_dataset.csv'):
    """
    Load the patient churn dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File not found at {filepath}")
        print("Please ensure the CSV file is in the 'data' folder")
        return None

def preprocess_data(df):
    """
    Preprocess the dataframe
    """
    if df is None:
        return None
    
    # Make a copy
    df_processed = df.copy()
    
    # Basic info
    print(f"\nDataset Overview:")
    print(f"Columns: {list(df_processed.columns)}")
    print(f"Missing values:\n{df_processed.isnull().sum()}")
    
    # Check churn distribution
    churn_dist = df_processed['Churn'].value_counts()
    churn_percent = df_processed['Churn'].value_counts(normalize=True) * 100
    print(f"\nChurn Distribution:")
    print(f"0 (No Churn): {churn_dist[0]} ({churn_percent[0]:.1f}%)")
    print(f"1 (Churn): {churn_dist[1]} ({churn_percent[1]:.1f}%)")
    
    return df_processed

def prepare_features(df):
    """
    Prepare features for modeling
    """
    if df is None:
        return None
    
    print("\n" + "="*50)
    print("Preparing features for modeling...")
    print("="*50)
    
    # Make a copy
    df_processed = df.copy()
    
    # Drop Patient_ID as it's not a predictive feature
    if 'Patient_ID' in df_processed.columns:
        df_processed = df_processed.drop('Patient_ID', axis=1)
        print("✓ Dropped Patient_ID column")
    
    # Check data types
    print(f"\nData Types:")
    print(df_processed.dtypes)
    
    # Encode categorical variables
    categorical_cols = ['Gender', 'Chronic_Disease', 'Insurance_Type']
    
    # Check which categorical columns exist
    existing_cat_cols = [col for col in categorical_cols if col in df_processed.columns]
    print(f"\nCategorical columns to encode: {existing_cat_cols}")
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df_processed, columns=existing_cat_cols, drop_first=True)
    print(f"✓ One-hot encoded categorical variables")
    print(f"  Shape after encoding: {df_encoded.shape}")
    
    # Separate features and target
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ Data split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Scale numerical features
    print(f"\nScaling numerical features...")
    
    # Get numerical columns (excluding the binary encoded ones)
    numerical_cols = ['Age', 'Tenure_Months', 'Visits_Last_Year', 
                     'Satisfaction_Score', 'Total_Bill_Amount', 'Missed_Appointments']
    
    # Filter to only columns that exist
    existing_num_cols = [col for col in numerical_cols if col in X_train.columns]
    
    if existing_num_cols:
        scaler = StandardScaler()
        
        # Scale features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[existing_num_cols] = scaler.fit_transform(X_train[existing_num_cols])
        X_test_scaled[existing_num_cols] = scaler.transform(X_test[existing_num_cols])
        
        print(f"✓ Scaled {len(existing_num_cols)} numerical features")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the scaler for later use
        scaler_path = 'models/scaler.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"✓ Scaler saved to {scaler_path}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()
    else:
        print("✗ No numerical columns found for scaling")
        return X_train, X_test, y_train, y_test, None, X.columns.tolist()

def handle_imbalance(X_train, y_train, method='smote'):
    """
    Handle class imbalance
    """
    print(f"\n" + "="*50)
    print("Handling class imbalance...")
    print("="*50)
    
    print(f"Original class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    if method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            print(f"\n✓ Applied SMOTE for balancing")
            print(f"Balanced class distribution:")
            unique, counts = np.unique(y_train_balanced, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"  Class {label}: {count} samples ({count/len(y_train_balanced)*100:.1f}%)")
            
            return X_train_balanced, y_train_balanced
        
        except ImportError:
            print("✗ imbalanced-learn not installed. Using class weights instead.")
            print("  Install with: pip install imbalanced-learn")
            return X_train, y_train
    
    return X_train, y_train