# Patient Churn Prediction System

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
