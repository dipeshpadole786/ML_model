import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

def analyze_dataset(df, save_plots=True):
    """
    Analyze the dataset and create visualizations
    """
    print("📊 Dataset Analysis")
    print("="*60)
    
    # Basic info
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    
    # Churn distribution
    churn_counts = df['Churn'].value_counts()
    churn_percent = df['Churn'].value_counts(normalize=True) * 100
    
    print(f"\nChurn Distribution:")
    print(f"0 (No Churn): {churn_counts[0]} ({churn_percent[0]:.1f}%)")
    print(f"1 (Churn): {churn_counts[1]} ({churn_percent[1]:.1f}%)")
    
    # Numerical features statistics
    numerical_cols = ['Age', 'Tenure_Months', 'Visits_Last_Year', 
                     'Satisfaction_Score', 'Total_Bill_Amount', 'Missed_Appointments']
    
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    if numerical_cols:
        print(f"\nNumerical Features Statistics:")
        print(df[numerical_cols].describe().round(2))
    
    # Categorical features
    categorical_cols = ['Gender', 'Chronic_Disease', 'Insurance_Type']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    if categorical_cols:
        print(f"\nCategorical Features:")
        for col in categorical_cols:
            print(f"\n{col}:")
            print(df[col].value_counts())
    
    # Create visualizations if requested
    if save_plots:
        create_visualizations(df)
    
    return df

def create_visualizations(df):
    """
    Create exploratory visualizations
    """
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Churn Distribution
    plt.figure(figsize=(8, 6))
    churn_counts = df['Churn'].value_counts()
    colors = ['lightblue', 'lightcoral']
    plt.bar(['No Churn', 'Churn'], churn_counts.values, color=colors)
    plt.title('Patient Churn Distribution')
    plt.ylabel('Number of Patients')
    
    # Add percentage labels
    total = len(df)
    for i, count in enumerate(churn_counts.values):
        percentage = (count / total) * 100
        plt.text(i, count + 5, f'{percentage:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/churn_distribution.png', dpi=300)
    plt.close()
    
    # 2. Churn by Gender (if column exists)
    if 'Gender' in df.columns:
        plt.figure(figsize=(8, 6))
        churn_by_gender = df.groupby('Gender')['Churn'].mean() * 100
        churn_by_gender.plot(kind='bar', color=['lightblue', 'lightpink'])
        plt.title('Churn Rate by Gender')
        plt.ylabel('Churn Rate (%)')
        plt.xlabel('Gender')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('visualizations/churn_by_gender.png', dpi=300)
        plt.close()
    
    # 3. Churn by Insurance Type (if column exists)
    if 'Insurance_Type' in df.columns:
        plt.figure(figsize=(10, 6))
        churn_by_insurance = df.groupby('Insurance_Type')['Churn'].mean() * 100
        churn_by_insurance.plot(kind='bar', color='lightgreen')
        plt.title('Churn Rate by Insurance Type')
        plt.ylabel('Churn Rate (%)')
        plt.xlabel('Insurance Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/churn_by_insurance.png', dpi=300)
        plt.close()
    
    # 4. Correlation Heatmap
    numerical_cols = ['Age', 'Tenure_Months', 'Visits_Last_Year', 
                     'Satisfaction_Score', 'Total_Bill_Amount', 'Missed_Appointments', 'Churn']
    
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png', dpi=300)
        plt.close()
    
    print(f"✓ Visualizations saved to 'visualizations/' folder")

def sample_patient_data():
    """
    Generate sample patient data for testing
    """
    samples = [
        {
            'Patient_ID': 1001,
            'Age': 45,
            'Gender': 'Female',
            'Tenure_Months': 24,
            'Visits_Last_Year': 8,
            'Chronic_Disease': 'No',
            'Insurance_Type': 'Private',
            'Satisfaction_Score': 4.2,
            'Total_Bill_Amount': 18500,
            'Missed_Appointments': 2
        },
        {
            'Patient_ID': 1002,
            'Age': 68,
            'Gender': 'Male',
            'Tenure_Months': 6,
            'Visits_Last_Year': 3,
            'Chronic_Disease': 'Yes',
            'Insurance_Type': 'Government',
            'Satisfaction_Score': 2.1,
            'Total_Bill_Amount': 32500,
            'Missed_Appointments': 7
        },
        {
            'Patient_ID': 1003,
            'Age': 32,
            'Gender': 'Female',
            'Tenure_Months': 36,
            'Visits_Last_Year': 12,
            'Chronic_Disease': 'No',
            'Insurance_Type': 'None',
            'Satisfaction_Score': 3.8,
            'Total_Bill_Amount': 12500,
            'Missed_Appointments': 1
        }
    ]
    
    return pd.DataFrame(samples)

def save_sample_data():
    """
    Save sample patient data to CSV for testing
    """
    df = sample_patient_data()
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_patients.csv', index=False)
    print("✓ Sample patient data saved to 'data/sample_patients.csv'")
    return df