# Customer Churn Prediction For Telco

## Overview
This project implements a machine learning pipeline to predict customer churn for a telecommunications company like Safaricom. The model helps identify customers who are likely to discontinue services, enabling proactive retention strategies.

## Dataset
The analysis uses the Telco Customer Churn dataset from Kaggle, which contains information about 7,043 customers and various service-related attributes.

- **Source**: [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download)
- **Size**: 7,043 records with 21 features
- **Target Variable**: Churn (Yes/No)

## Features
The dataset includes information about:
- Customer demographics (gender, senior citizen status, partner, dependents)
- Account information (tenure, contract type, payment method)
- Services subscribed (phone, internet, streaming, security)
- Billing information (paperless billing, monthly charges, total charges)

## Pipeline Structure
1. **Data Loading and Exploration**: Initial examination of dataset characteristics
2. **Data Preprocessing**: 
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling
3. **Model Training and Evaluation**: 
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost
   - LightGBM
4. **Model Selection**: Best model chosen based on F1 score
5. **Model Persistence**: Saving trained model for deployment

## Results
- Best performing model: **LightGBM**
- Metrics:
  - Accuracy: 0.8013
  - Precision: 0.6497
  - Recall: 0.5455
  - F1 Score: 0.5930
  - AUC: 0.8350

## Visualizations
The pipeline generates several visualizations to aid in understanding:
- Confusion matrix
- Feature importance ranking
- Model performance comparison

## Installation and Usage
```bash
# Clone this repository
git clone https://github.com/yourusername/safaricom-churn-prediction.git

# Navigate to the project directory
cd safaricom-churn-prediction

# Install required packages
pip install -r requirements.txt

# Run the prediction pipeline
python main.py
```

## Future Improvements
- Hyperparameter tuning for models
- Handling class imbalance
- Feature engineering
- Deployment as a web service

## License
[MIT License](LICENSE)
