import argparse
import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
try:
    from src.train_model import ModelTrainer, quick_train
    from src.predict import ChurnPredictor
    from src.utils import analyze_dataset, sample_patient_data, save_sample_data
    from src.data_preprocessing import load_data, preprocess_data
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running from the correct directory and all files are present.")
    sys.exit(1)

def setup_environment():
    """
    Setup the project environment
    """
    # Create necessary directories
    directories = ['data', 'models', 'visualizations']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Project directories created")
    
    # Check if data file exists
    data_file = 'data/patient_churn_dataset.csv'
    if not os.path.exists(data_file):
        print(f"⚠ Warning: {data_file} not found")
        print("Please place your dataset in the 'data' folder")
        return False
    
    return True

def train_command(args):
    """
    Handle train command
    """
    print("\n" + "="*60)
    print("TRAINING MODE")
    print("="*60)
    
    # Setup environment
    if not setup_environment():
        return
    
    # Check data file
    data_path = 'data/patient_churn_dataset.csv'
    if not os.path.exists(data_path):
        print(f"✗ Data file not found: {data_path}")
        return
    
    # Train models
    trainer = ModelTrainer()
    best_model = trainer.train_models(data_path)
    
    if best_model:
        print("\n✅ Training completed successfully!")
        print(f"   Model saved to: models/best_model.pkl")
        print(f"   Visualizations saved to: visualizations/")
    else:
        print("\n❌ Training failed!")

def predict_command(args):
    """
    Handle predict command
    """
    print("\n" + "="*60)
    print("PREDICTION MODE")
    print("="*60)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    if predictor.model is None:
        print("✗ No trained model found. Please train a model first.")
        print("   Run: python main.py --mode train")
        return
    
    # Check input source
    if args.input:
        # Predict from CSV file
        if args.input.endswith('.csv'):
            results = predictor.predict_csv(args.input)
            if results is not None:
                # Save results
                output_file = args.output if args.output else 'predictions.csv'
                results.to_csv(output_file, index=False)
                print(f"✓ Predictions saved to: {output_file}")
                
                # Show preview
                print(f"\n📋 Results Preview (first 5 patients):")
                print(results.head().to_string())
        else:
            print(f"✗ Input file must be a CSV file")
    else:
        # Interactive prediction
        print("\n🎯 Interactive Prediction Mode")
        print("Enter patient details (or 'quit' to exit):")
        
        while True:
            try:
                print("\n" + "-"*40)
                patient_data = {}
                
                # Get patient details
                patient_id = input("Patient ID (or 'quit'): ").strip()
                if patient_id.lower() == 'quit':
                    break
                
                patient_data['Patient_ID'] = patient_id
                patient_data['Age'] = int(input("Age: "))
                patient_data['Gender'] = input("Gender (Male/Female): ").capitalize()
                patient_data['Tenure_Months'] = int(input("Tenure (months): "))
                patient_data['Visits_Last_Year'] = int(input("Visits last year: "))
                patient_data['Chronic_Disease'] = input("Chronic Disease? (Yes/No): ").capitalize()
                patient_data['Insurance_Type'] = input("Insurance Type (Private/Government/None): ").capitalize()
                patient_data['Satisfaction_Score'] = float(input("Satisfaction Score (1-5): "))
                patient_data['Total_Bill_Amount'] = float(input("Total Bill Amount: "))
                patient_data['Missed_Appointments'] = int(input("Missed Appointments: "))
                
                # Make prediction
                result = predictor.predict_single(patient_data)
                
                if result:
                    print(f"\n🔍 Prediction Results:")
                    print(f"   Patient ID: {result['patient_id']}")
                    print(f"   Churn Prediction: {'Will Churn' if result['churn_prediction'] == 1 else 'Will Not Churn'}")
                    print(f"   Churn Probability: {result['churn_probability']:.1%}")
                    print(f"   Risk Level: {result['risk_level']}")
                    print(f"   Recommendation: {result['recommendation']}")
            
            except (ValueError, KeyboardInterrupt):
                print("\nExiting interactive mode...")
                break

def analyze_command(args):
    """
    Handle analyze command
    """
    print("\n" + "="*60)
    print("ANALYSIS MODE")
    print("="*60)
    
    # Setup environment
    if not setup_environment():
        return
    
    # Check data file
    data_path = 'data/patient_churn_dataset.csv'
    if not os.path.exists(data_path):
        print(f"✗ Data file not found: {data_path}")
        return
    
    # Load and analyze data
    df = load_data(data_path)
    if df is not None:
        analyze_dataset(df, save_plots=True)

def demo_command(args):
    """
    Handle demo command
    """
    print("\n" + "="*60)
    print("DEMONSTRATION MODE")
    print("="*60)
    
    # Setup environment
    setup_environment()
    
    # Create sample data
    print("\n1. Creating sample patient data...")
    save_sample_data()
    
    # Check if model exists, if not train one
    if not os.path.exists('models/best_model.pkl'):
        print("\n2. No trained model found. Training a model...")
        data_path = 'data/patient_churn_dataset.csv'
        
        if not os.path.exists(data_path):
            print(f"✗ Cannot train: {data_path} not found")
            print("Please add your dataset to the 'data' folder")
            return
        
        trainer = ModelTrainer()
        trainer.train_models(data_path)
    
    # Make predictions on sample data
    print("\n3. Making predictions on sample patients...")
    predictor = ChurnPredictor()
    
    if predictor.model is None:
        print("✗ Cannot make predictions: Model not loaded")
        return
    
    results = predictor.predict_csv('data/sample_patients.csv')
    
    if results is not None:
        print("\n📊 Sample Predictions:")
        print(results.to_string())
        
        # Save results
        results.to_csv('sample_predictions.csv', index=False)
        print(f"\n✓ Sample predictions saved to: sample_predictions.csv")

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description='Patient Churn Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train                  # Train model
  python main.py --mode predict                # Interactive prediction
  python main.py --mode predict --input data/sample.csv  # Batch prediction
  python main.py --mode analyze                # Analyze dataset
  python main.py --mode demo                   # Run demonstration
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'predict', 'analyze', 'demo'],
        default='train',
        help='Mode of operation'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input CSV file for batch prediction'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Output file for predictions (default: predictions.csv)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command based on mode
    if args.mode == 'train':
        train_command(args)
    elif args.mode == 'predict':
        predict_command(args)
    elif args.mode == 'analyze':
        analyze_command(args)
    elif args.mode == 'demo':
        demo_command(args)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏥 PATIENT CHURN PREDICTION SYSTEM")
    print("="*60)
    
    main()