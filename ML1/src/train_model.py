import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import from our modules
from .data_preprocessing import load_data, preprocess_data, prepare_features, handle_imbalance

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_models(self, data_path='data/patient_churn_dataset.csv'):
        """
        Main training pipeline
        """
        print("="*60)
        print("PATIENT CHURN PREDICTION - MODEL TRAINING")
        print("="*60)
        
        # Step 1: Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        df = load_data(data_path)
        
        if df is None:
            print("✗ Cannot proceed without data.")
            return None
        
        df_processed = preprocess_data(df)
        
        if df_processed is None:
            print("✗ Data preprocessing failed.")
            return None
        
        # Step 2: Prepare features
        print("\n2. Preparing features...")
        result = prepare_features(df_processed)
        
        if result is None:
            print("✗ Feature preparation failed.")
            return None
        
        X_train, X_test, y_train, y_test, scaler, feature_names = result
        
        # Step 3: Handle imbalance
        print("\n3. Handling class imbalance...")
        X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)
        
        # Step 4: Define models
        print("\n4. Initializing models...")
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42, 
                n_estimators=100,
                class_weight='balanced_subsample'
            ),
            'XGBoost': XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'SVM': SVC(
                probability=True, 
                random_state=42,
                class_weight='balanced'
            )
        }
        
        # Step 5: Train and evaluate each model
        print("\n5. Training and evaluating models...")
        print("-"*60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            self.results[name] = self._train_and_evaluate(
                model, name, X_train_balanced, X_test, y_train_balanced, y_test
            )
        
        # Step 6: Select best model
        print("\n6. Selecting best model...")
        self._select_best_model()
        
        # Step 7: Save best model
        print("\n7. Saving best model...")
        self._save_best_model(feature_names)
        
        # Step 8: Generate reports
        print("\n8. Generating evaluation reports...")
        self._generate_reports(X_test, y_test)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        return self.best_model
    
    def _train_and_evaluate(self, model, model_name, X_train, X_test, y_train, y_test):
        """
        Train and evaluate a single model
        """
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'model': model,
                'name': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'feature_names': X_train.columns.tolist()
            }
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            # Print results
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  CV AUC:    {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")
            
            return metrics
            
        except Exception as e:
            print(f"✗ Error training {model_name}: {str(e)}")
            return None
    
    def _select_best_model(self):
        """
        Select the best model based on ROC-AUC score
        """
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        if not valid_results:
            print("✗ No models were successfully trained.")
            return
        
        # Find best model by ROC-AUC
        self.best_model_name = max(valid_results, key=lambda x: valid_results[x]['roc_auc'])
        self.best_model = valid_results[self.best_model_name]['model']
        
        print(f"\n✓ Best Model: {self.best_model_name}")
        print(f"  ROC-AUC Score: {valid_results[self.best_model_name]['roc_auc']:.4f}")
        
        # Print comparison table
        print("\nModel Comparison:")
        print("-"*80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-"*80)
        
        for name, metrics in valid_results.items():
            if metrics:
                print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                      f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['roc_auc']:<10.4f}")
    
    def _save_best_model(self, feature_names):
        """
        Save the best model and metadata
        """
        if self.best_model is None:
            print("✗ No model to save.")
            return
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_path = 'models/best_model.pkl'
        joblib.dump(self.best_model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'feature_names': feature_names,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': type(self.best_model).__name__
        }
        
        metadata_path = 'models/model_metadata.pkl'
        joblib.dump(metadata, metadata_path)
        
        print(f"✓ Best model saved to: {model_path}")
        print(f"✓ Model metadata saved to: {metadata_path}")
    
    def _generate_reports(self, X_test, y_test):
        """
        Generate evaluation reports and visualizations
        """
        if self.best_model is None:
            return
        
        # Create visualizations directory
        os.makedirs('visualizations', exist_ok=True)
        
        # Get predictions from best model
        best_result = self.results[self.best_model_name]
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, best_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png', dpi=300)
        plt.close()
        
        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, best_result['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.best_model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('visualizations/roc_curve.png', dpi=300)
        plt.close()
        
        # 3. Feature Importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': best_result['feature_names'],
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', 
                       data=feature_importance.head(15))
            plt.title(f'Top 15 Feature Importance - {self.best_model_name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png', dpi=300)
            plt.close()
            
            # Save feature importance to CSV
            feature_importance.to_csv('visualizations/feature_importance.csv', index=False)
        
        # 4. Save classification report to text file
        report = classification_report(y_test, best_result['y_pred'], 
                                     target_names=['No Churn', 'Churn'])
        
        with open('visualizations/classification_report.txt', 'w') as f:
            f.write(f"Model: {self.best_model_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\nClassification Report:\n")
            f.write(report)
            f.write(f"\nROC-AUC Score: {best_result['roc_auc']:.4f}\n")
            f.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
        
        print(f"✓ Visualizations saved to 'visualizations/' folder")
        print(f"✓ Classification report saved to 'visualizations/classification_report.txt'")

def quick_train():
    """
    Quick training function for simple usage
    """
    trainer = ModelTrainer()
    return trainer.train_models()