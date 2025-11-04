import pandas as pd
import numpy as np

def validate_dataset(df):
    """
    Validate the input dataset for compatibility with the ADR model.
    """
    validation_results = {
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check if dataframe is empty
    if df.empty:
        validation_results['errors'].append('Dataset is empty')
        return False, validation_results
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        cols_with_missing = missing_values[missing_values > 0].to_dict()
        validation_results['errors'].append({
            'message': 'Missing values detected',
            'details': cols_with_missing
        })
    
    # Check data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if len(numeric_cols) == 0:
        validation_results['errors'].append('No numeric columns found in dataset')
    
    if len(non_numeric_cols) > 0:
        validation_results['warnings'].append({
            'message': 'Non-numeric columns detected',
            'columns': non_numeric_cols.tolist()
        })
    
    # Calculate basic statistics for numeric columns
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'missing': int(df[col].isnull().sum())
        }
    validation_results['stats'] = stats
    
    is_valid = len(validation_results['errors']) == 0
    return is_valid, validation_results

def prepare_dataset(df, target_column):
    """
    Prepare the dataset for training by separating features and target.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Remove any non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_column not in numeric_df.columns:
        raise ValueError(f"Target column '{target_column}' must be numeric")
    
    # Separate features and target
    X = numeric_df.drop(columns=[target_column])
    y = numeric_df[target_column]
    
    # Handle missing values if any
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    return X, y
