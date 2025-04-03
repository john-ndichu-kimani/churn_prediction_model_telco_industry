import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple machine learning models and evaluate their performance"""
    print("\n----- Training Machine Learning Models -----")
    
    # Define the models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
    }
    
    # Results storage
    results = {}
    trained_models = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate AUC if the model supports probability predictions
        auc = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        
        # Store results
        training_time = time.time() - start_time
        results[name] = {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'training_time': training_time
        }
        
        trained_models[name] = model
        
        # Print results
        print(f"{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        if auc:
            print(f"  AUC: {auc:.4f}")
        print(f"  Training Time: {training_time:.2f} seconds")
    
    # Find best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    print(f"\nBest model based on F1 score: {best_model_name}")
    
    return trained_models, results, best_model_name

def evaluate_best_model(model, X_test, y_test, feature_names=None):
    """Evaluate the best model in more detail"""
    print("\n----- Detailed Evaluation of Best Model -----")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix plot saved as 'confusion_matrix.png'")
    
    # Plot feature importance if available
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort and get top 15 features
        indices = np.argsort(importances)[::-1][:15]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_importances, y=top_features)
        plt.title('Top 15 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print("Feature importance plot saved as 'feature_importance.png'")
    
    return report

if __name__ == "__main__":
    # This is just for testing the module
    from data_preparation import load_data, preprocess_data, split_dataset
    
    print("Loading and preprocessing data...")
    df = load_data("Telco-Customer-Churn.csv")
    processed_df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(processed_df)
    
    print("Training models...")
    trained_models, results, best_model_name = train_models(X_train, y_train, X_test, y_test)
    
    print("Evaluating best model...")
    best_model = trained_models[best_model_name]
    evaluate_best_model(best_model, X_test, y_test, feature_names=X_train.columns)
    
    print("Model training completed successfully!")