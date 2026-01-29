#!/usr/bin/env python
"""
Setup script for Patient Churn Prediction System
"""

import subprocess
import sys
import os


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    if sys.version_info < (3, 7):
        print("✗ Python 3.7 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def install_requirements():
    """Install required packages"""
    print("\nInstalling requirements...")

    requirements = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
        "imbalanced-learn>=0.11.0",
        "xgboost>=1.7.0",
    ]

    for package in requirements:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"  ✓ {package} installed")
        except subprocess.CalledProcessError:
            print(f"  ✗ Failed to install {package}")

    print("\n✓ All requirements installed")


def setup_directories():
    """Create necessary directories"""
    print("\nSetting up directories...")

    directories = ["data", "models", "visualizations", "src"]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  Created: {directory}/")
        else:
            print(f"  Already exists: {directory}/")

    print("✓ Directory structure created")


def check_data_file():
    """Check if data file exists"""
    print("\nChecking data file...")

    data_file = "data/patient_churn_dataset.csv"
    if os.path.exists(data_file):
        print(f"✓ Data file found: {data_file}")
        return True
    else:
        print(f"⚠ Data file not found: {data_file}")
        print("  Please place your CSV file in the 'data' folder")
        return False


def create_sample_files():
    """Create sample configuration files"""
    print("\nCreating sample files...")

    # requirements.txt
    requirements_content = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
imbalanced-learn>=0.11.0
xgboost>=1.7.0
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    print("  Created: requirements.txt")

    # README.md
    readme_content = """# Patient Churn Prediction System

## Overview
This system predicts patient churn using machine learning models.

## Installation
1. Install Python 3.7 or higher
2. Run:
   python setup.py
3. Place your dataset at:
   data/patient_churn_dataset.csv

## Usage

Train the model:
python main.py --mode train

Make predictions (interactive):
python main.py --mode predict

Make batch predictions:
python main.py --mode predict --input data/sample.csv

Analyze dataset:
python main.py --mode analyze

Run demo:
python main.py --mode demo
"""

    with open("README.md", "w") as f:
        f.write(readme_content)
    print("  Created: README.md")


def main():
    print("=== Patient Churn Prediction System Setup ===")

    if not check_python_version():
        sys.exit(1)

    install_requirements()
    setup_directories()
    check_data_file()
    create_sample_files()

    print("\n🎉 Setup completed successfully!")


if __name__ == "__main__":
    main()
