import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    """
    Cleans and prepares any uploaded dataset for clustering.
    Steps:
      - Removes rows/columns with too many NaNs
      - Encodes categorical columns
      - Fills missing values
      - Scales numerical features
    Returns:
      - Cleaned DataFrame ready for clustering
    """
    df = df.copy()

    # Drop columns where all values are NaN
    df.dropna(axis=1, how="all", inplace=True)

    # Handle categorical data (Label Encoding for non-numeric columns)
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            try:
                df[col] = le.fit_transform(df[col].astype(str))
            except Exception:
                df.drop(columns=[col], inplace=True)

    # Fill remaining NaNs with column mean
    df = df.fillna(df.mean(numeric_only=True))

    # Drop rows still containing NaNs (if any)
    df.dropna(inplace=True)

    # Scale features for better clustering
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df_scaled
