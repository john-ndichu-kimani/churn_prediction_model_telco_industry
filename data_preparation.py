import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the customer churn dataset"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def explore_data(df):
    """Explore the dataset and print key information"""
    print("\n----- Dataset Overview -----")
    print(f"First 5 rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDescriptive statistics:\n{df.describe()}")
    
    # Check target variable distribution
    if 'Churn' in df.columns:
        churn_counts = df['Churn'].value_counts(normalize=True) * 100
        print(f"\nChurn distribution:")
        for churn, percentage in churn_counts.items():
            print(f"  {churn}: {percentage:.2f}%")

def preprocess_data(df):
    """Preprocess the dataset for machine learning"""
    print("\n----- Preprocessing Data -----")
    
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Remove customer ID as it's not needed for prediction
    if 'customerID' in processed_df.columns:
        processed_df.drop(columns=['customerID'], inplace=True)
        print("Removed customerID column")
    
    # Convert TotalCharges to numeric
    if 'TotalCharges' in processed_df.columns:
        processed_df['TotalCharges'] = pd.to_numeric(processed_df['TotalCharges'], errors='coerce')
        # Fill missing values with median
        median_charge = processed_df['TotalCharges'].median()
        processed_df['TotalCharges'].fillna(median_charge, inplace=True)
        print(f"Converted TotalCharges to numeric and filled {df['TotalCharges'].isna().sum()} missing values with median ({median_charge:.2f})")
    
    # Encode binary categorical variables
    binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                     'PaperlessBilling', 'Churn']
    
    for col in binary_columns:
        if col in processed_df.columns:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col])
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"Encoded {col}: {mapping}")
    
    # One-hot encode multi-category variables
    categorical_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    
    # Filter to only include columns that exist in the dataframe
    cat_cols_present = [col for col in categorical_columns if col in processed_df.columns]
    
    if cat_cols_present:
        print(f"One-hot encoding {len(cat_cols_present)} categorical columns")
        processed_df = pd.get_dummies(processed_df, columns=cat_cols_present, drop_first=True)
    
    # Scale numerical features
    numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    num_cols_present = [col for col in numerical_columns if col in processed_df.columns]
    
    if num_cols_present:
        print(f"Scaling {len(num_cols_present)} numerical columns")
        scaler = MinMaxScaler()
        processed_df[num_cols_present] = scaler.fit_transform(processed_df[num_cols_present])
    
    print(f"Preprocessed dataframe has {processed_df.shape[1]} features")
    return processed_df

def split_dataset(df, target_col='Churn', test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets"""
    print("\n----- Splitting Dataset -----")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"Testing set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test the module
    df = load_data("Telco-Customer-Churn.csv")
    explore_data(df)
    processed_df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(processed_df)
    print("Data preparation completed successfully!")