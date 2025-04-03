import os
import argparse
import pandas as pd
# Set non-interactive backend before any matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive backend

# Now import matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from data_preparation import load_data, explore_data, preprocess_data, split_dataset
from model_training import train_models, evaluate_best_model
from model_deployment import save_model, test_saved_model, create_fastapi_app

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Safaricom Customer Churn Prediction')
    
    parser.add_argument('--data', type=str, default='Telco-Customer-Churn.csv',
                        help='Path to the input CSV data file')
    
    parser.add_argument('--output', type=str, default='models',
                        help='Directory to save the trained model')
    
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split')
    
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    
    parser.add_argument('--explore-only', action='store_true',
                        help='Only explore the data without training models')
    
    parser.add_argument('--api', action='store_true',
                        help='Generate FastAPI code for model deployment')
    
    return parser.parse_args()

def run_pipeline(args):
    """Run the complete churn prediction pipeline"""
    print("=" * 80)
    print("SAFARICOM CUSTOMER CHURN PREDICTION PIPELINE")
    print("=" * 80)
    
    # Step 1: Load and explore data
    print("\n[Step 1/5] Loading and exploring data...")
    df = load_data(args.data)
    explore_data(df)
    
    if args.explore_only:
        print("\nExploration completed. Exiting as requested.")
        return
    
    # Step 2: Preprocess data
    print("\n[Step 2/5] Preprocessing data...")
    processed_df = preprocess_data(df)
    
    # Step 3: Split dataset
    print("\n[Step 3/5] Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(
        processed_df, test_size=args.test_size, random_state=args.random_state
    )
    
    # Step 4: Train and evaluate models
    print("\n[Step 4/5] Training and evaluating models...")
    trained_models, results, best_model_name = train_models(X_train, y_train, X_test, y_test)
    best_model = trained_models[best_model_name]
    
    # Generate evaluation report and visualizations
    evaluate_best_model(best_model, X_test, y_test, feature_names=X_train.columns)
    
    # Compare models with a Seaborn bar chart
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")  # Set Seaborn style
    model_names = list(results.keys())
    f1_scores = [results[name]['f1_score'] for name in model_names]

    # Create a DataFrame for Seaborn
    comparison_df = pd.DataFrame({'Model': model_names, 'F1 Score': f1_scores})
    ax = sns.barplot(x='Model', y='F1 Score', data=comparison_df, palette='viridis')

    # Add value labels on top of bars
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center')

    plt.title('Model F1 Score Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    print("Model comparison plot saved as 'model_comparison.png'")
    
    # Step 5: Save and deploy the model
    print("\n[Step 5/5] Saving and testing the best model...")
    model_path = save_model(best_model, f"churn_model_{best_model_name.replace(' ', '_').lower()}", args.output)
    test_saved_model(model_path, X_test)
    
    # Generate API code if requested
    if args.api:
        create_fastapi_app(model_path, X_train.columns.tolist())
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nBest model: {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"Saved to: {model_path}")
    
    if args.api:
        print("\nTo run the API:")
        print("1. Install FastAPI and uvicorn:")
        print("   pip install fastapi uvicorn")
        print("2. Start the API server:")
        print("   uvicorn app:app --reload")
        print("3. Access the API documentation:")
        print("   http://127.0.0.1:8000/docs")

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)