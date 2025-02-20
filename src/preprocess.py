import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)

    # Separate categorical and numerical columns
    categorical_cols = ["Neighborhood"]
    numerical_cols = ['SquareFootage', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'BankValuation', 'Price']

    # Handle missing values (only for numerical columns)
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    # Encode categorical features
    label_encoder = LabelEncoder()
    df["Neighborhood"] = label_encoder.fit_transform(df["Neighborhood"])

    # Select features and target variable
    X = df[['SquareFootage', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'Neighborhood', 'BankValuation']]
    y = df['Price']

    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoder
