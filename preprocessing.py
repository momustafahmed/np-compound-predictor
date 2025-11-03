"""
Data preprocessing module for natural product compound analysis.
Handles data cleaning, feature engineering, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def load_data(filepath='np.csv'):
    """Load the dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    """
    Clean the dataset by handling missing values and data types.
    
    Args:
        df: Input dataframe
        
    Returns:
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Display initial info
    print(f"Initial shape: {df_clean.shape}")
    print(f"Missing values:\n{df_clean.isnull().sum()}")
    
    # Handle missing values in numerical features
    numerical_cols = ['mw', 'alogp', 'tpsa', 'hbd', 'hba', 'fsp3', 'dose_uM', 'ic50_uM']
    
    # Use median imputation for numerical features
    imputer = SimpleImputer(strategy='median')
    df_clean[numerical_cols] = imputer.fit_transform(df_clean[numerical_cols])
    
    print(f"\nAfter cleaning shape: {df_clean.shape}")
    print(f"Remaining missing values: {df_clean.isnull().sum().sum()}")
    
    return df_clean


def engineer_features(df):
    """
    Create additional features from existing data.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with engineered features
    """
    df_eng = df.copy()
    
    # Binary features - Define activity based on IC50 < 100 µM (common threshold in drug discovery)
    # This is more meaningful than the original 'active' column
    df_eng['is_active'] = (df_eng['ic50_uM'] < 100).astype(int)
    print(f"\n✓ Activity defined as IC50 < 100 µM")
    print(f"  Active compounds: {df_eng['is_active'].sum()} ({df_eng['is_active'].mean():.1%})")
    
    # Potency-related features
    df_eng['log_ic50'] = np.log10(df_eng['ic50_uM'] + 1)
    df_eng['log_dose'] = np.log10(df_eng['dose_uM'] + 1)
    df_eng['dose_ic50_ratio'] = df_eng['dose_uM'] / (df_eng['ic50_uM'] + 1)
    
    # Molecular property ratios
    df_eng['mw_tpsa_ratio'] = df_eng['mw'] / (df_eng['tpsa'] + 1)
    df_eng['hba_hbd_ratio'] = df_eng['hba'] / (df_eng['hbd'] + 1)
    df_eng['lipinski_violations'] = (
        (df_eng['mw'] > 500).astype(int) +
        (df_eng['alogp'] > 5).astype(int) +
        (df_eng['hbd'] > 5).astype(int) +
        (df_eng['hba'] > 10).astype(int)
    )
    
    # Drug-likeness score (simplified)
    df_eng['druglikeness_score'] = (
        (df_eng['mw'].between(150, 500)).astype(int) +
        (df_eng['alogp'].between(-0.4, 5.6)).astype(int) +
        (df_eng['tpsa'].between(20, 130)).astype(int) +
        (df_eng['hbd'] <= 5).astype(int) +
        (df_eng['hba'] <= 10).astype(int)
    ) / 5.0
    
    # Categorical encoding
    le_source = LabelEncoder()
    le_class = LabelEncoder()
    le_assay = LabelEncoder()
    
    df_eng['source_encoded'] = le_source.fit_transform(df_eng['source'])
    df_eng['class_encoded'] = le_class.fit_transform(df_eng['class'])
    df_eng['assay_encoded'] = le_assay.fit_transform(df_eng['assay'])
    
    return df_eng, le_source, le_class, le_assay


def prepare_features(df, feature_cols=None):
    """
    Prepare features for model training.
    
    Args:
        df: Input dataframe with engineered features
        feature_cols: List of columns to use as features (if None, use default)
        
    Returns:
        X (features), y (target), feature names
    """
    if feature_cols is None:
        # Default feature set
        feature_cols = [
            'mw', 'alogp', 'tpsa', 'hbd', 'hba', 'fsp3',
            'dose_uM', 'ic50_uM', 'log_ic50', 'log_dose',
            'dose_ic50_ratio', 'mw_tpsa_ratio', 'hba_hbd_ratio',
            'lipinski_violations', 'druglikeness_score',
            'source_encoded', 'class_encoded', 'assay_encoded'
        ]
    
    X = df[feature_cols]
    y = df['is_active']
    
    return X, y, feature_cols


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Check class distribution
    class_counts = y.value_counts()
    min_class_count = class_counts.min()
    
    # Only use stratify if we have at least 2 samples per class
    stratify_param = y if min_class_count >= 2 else None
    
    if stratify_param is None:
        print("⚠️  Warning: Class imbalance is too extreme for stratification. Using random split.")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")
    print(f"Positive class ratio in train: {y_train.mean():.3f}")
    print(f"Positive class ratio in test: {y_test.mean():.3f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_full_pipeline(filepath='np.csv', test_size=0.2, random_state=42):
    """
    Execute complete preprocessing pipeline.
    
    Args:
        filepath: Path to CSV file
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Dictionary containing all processed data and objects
    """
    # Load and clean
    df = load_data(filepath)
    df_clean = clean_data(df)
    
    # Engineer features
    df_eng, le_source, le_class, le_assay = engineer_features(df_clean)
    
    # Prepare features
    X, y, feature_cols = prepare_features(df_eng)
    
    # Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(
        X, y, test_size, random_state
    )
    
    return {
        'df_original': df,
        'df_clean': df_clean,
        'df_engineered': df_eng,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'label_encoders': {
            'source': le_source,
            'class': le_class,
            'assay': le_assay
        }
    }


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing preprocessing pipeline...")
    results = get_full_pipeline()
    print("\n✓ Preprocessing pipeline completed successfully!")
    print(f"Features shape: {results['X_train'].shape}")
    print(f"Number of features: {len(results['feature_cols'])}")
