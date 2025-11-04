"""
Streamlit Web Application: HistGradientBoostingRegressor Demonstration
A comprehensive ML regression comparison tool with model explanations and visualizations
"""

# ============================================================================
# IMPORTS
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

# Scikit-learn imports
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

# XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost not installed. Install with: pip install xgboost")

# Hugging Face datasets (optional)
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Other utilities
import io
import joblib
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="HistGradientBoosting Regressor Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(dataset_choice, uploaded_file=None):
    """
    Load dataset from built-in scikit-learn datasets, Hugging Face, or user upload.
    
    Args:
        dataset_choice: Name of the dataset to load
        uploaded_file: Optional uploaded CSV file
        
    Returns:
        DataFrame with features and target
    """
    if dataset_choice == "California Housing":
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        target_col = 'MedHouseVal'
        
    elif dataset_choice == "Diabetes":
        data = load_diabetes(as_frame=True)
        df = data.frame
        target_col = 'target'
        
    elif dataset_choice == "Upload CSV" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Convert date columns to numeric features
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].notna().any():
                        # Convert to multiple numeric features
                        df[f'{col}_year'] = df[col].dt.year
                        df[f'{col}_month'] = df[col].dt.month
                        df[f'{col}_day'] = df[col].dt.day
                        df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                        # Drop original date column
                        df = df.drop(columns=[col])
                        st.info(f"✅ Converted date column '{col}' to numeric features (year, month, day, dayofweek)")
                except:
                    pass
        
        target_col = None  # Will be selected by user
        
    else:
        # Default to California Housing
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        target_col = 'MedHouseVal'
    
    return df, target_col


def detect_column_types(df, target_col):
    """
    Automatically detect numerical and categorical columns.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
    Returns:
        Tuple of (numerical_cols, categorical_cols)
    """
    # Exclude target column
    feature_cols = [col for col in df.columns if col != target_col]
    
    numerical_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        # Skip date/datetime columns (should have been converted already)
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
            
        if df[col].dtype in ['int64', 'float64']:
            # Check if it's actually categorical (few unique values)
            if df[col].nunique() < 10 and df[col].dtype == 'int64':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return numerical_cols, categorical_cols


def preprocess_data(df, target_col):
    """
    Preprocess data: handle missing values, encode categoricals, scale numericals.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
    Returns:
        X, y, preprocessor, numerical_cols, categorical_cols
    """
    # Validate target column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Check if target is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"Target column '{target_col}' must be numeric. Found type: {df[target_col].dtype}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle missing values in target
    missing_target_count = y.isnull().sum()
    if missing_target_count > 0:
        st.warning(f"⚠️ Found {missing_target_count} missing values in target column. Removing these rows...")
        # Remove rows with missing target
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        st.info(f"✅ Dataset size after removing missing targets: {len(X)} samples")
    
    # Detect column types
    numerical_cols, categorical_cols = detect_column_types(df, target_col)
    
    # Create preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    transformers = []
    if numerical_cols:
        transformers.append(('num', numerical_pipeline, numerical_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_pipeline, categorical_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    return X, y, preprocessor, numerical_cols, categorical_cols


def build_pipelines(preprocessor):
    """
    Build preprocessing + model pipelines for all regression models.
    
    Args:
        preprocessor: ColumnTransformer for preprocessing
        
    Returns:
        Dictionary of model pipelines
    """
    pipelines = {
        # HistGradientBoosting - IMPROVED with regularization and early stopping
        'HistGradientBoosting': Pipeline([
            ('preprocessor', preprocessor),
            ('model', HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.05,        # Reduced from 0.1 for less overfitting
                max_depth=4,               # Reduced from 5 for simpler trees
                min_samples_leaf=30,       # Increased from 20 for more regularization
                l2_regularization=1.0,     # Added L2 penalty
                max_bins=200,              # Reduced from 255 for less granularity
                early_stopping=True,       # Stop when validation doesn't improve
                n_iter_no_change=10,       # Stop after 10 iterations without improvement
                validation_fraction=0.1,   # Use 10% of training for validation
                random_state=42
            ))
        ]),
        # Linear Regression - Simple baseline model
        'LinearRegression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ]),
        # Ridge Regression - Better than LinearRegression with L2 regularization
        'Ridge': Pipeline([
            ('preprocessor', preprocessor),
            ('model', Ridge(
                alpha=10.0,                # L2 regularization strength
                random_state=42
            ))
        ]),
        # RandomForest - IMPROVED with strong regularization
        'RandomForest': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=6,               # Reduced from 10 to prevent overfitting
                min_samples_split=20,      # Increased from 2 for more conservative splits
                min_samples_leaf=10,       # Increased from 1 for larger leaf nodes
                max_features='sqrt',       # Changed from 'auto' to use fewer features
                bootstrap=True,
                oob_score=True,            # Out-of-bag score for validation
                random_state=42,
                n_jobs=-1
            ))
        ])
    }
    
    # Add XGBoost if available - IMPROVED with massive regularization
    if XGBOOST_AVAILABLE:
        pipelines['XGBoost'] = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(
                n_estimators=100,
                learning_rate=0.03,        # Much slower learning (reduced from 0.1)
                max_depth=3,               # Shallower trees (reduced from 5)
                min_child_weight=5,        # More regularization (increased from 1)
                subsample=0.7,             # Use 70% of samples per tree
                colsample_bytree=0.7,      # Use 70% of features per tree
                gamma=0.1,                 # Minimum loss reduction for split
                reg_alpha=0.5,             # L1 regularization
                reg_lambda=1.0,            # L2 regularization
                random_state=42
            ))
        ])
    
    return pipelines


def train_and_evaluate(pipelines, X_train, y_train, X_test, y_test):
    """
    Train all models and calculate evaluation metrics.
    
    Args:
        pipelines: Dictionary of model pipelines
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Dictionary with trained models and metrics
    """
    results = {}
    
    for name, pipeline in pipelines.items():
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'R² Score (Train)': r2_score(y_train, y_pred_train),
            'R² Score (Test)': r2_score(y_test, y_pred_test),
            'MSE (Train)': mean_squared_error(y_train, y_pred_train),
            'MSE (Test)': mean_squared_error(y_test, y_pred_test),
            'MAE (Train)': mean_absolute_error(y_train, y_pred_train),
            'MAE (Test)': mean_absolute_error(y_test, y_pred_test),
            'RMSE (Train)': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'RMSE (Test)': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
        
        results[name] = {
            'pipeline': pipeline,
            'metrics': metrics,
            'predictions_train': y_pred_train,
            'predictions_test': y_pred_test
        }
    
    return results


def plot_comparison(results):
    """
    Create comparison bar charts for model metrics.
    
    Args:
        results: Dictionary with model results
        
    Returns:
        Matplotlib figure
    """
    # Set style
    sns.set_style("whitegrid")
    
    # Extract metrics for comparison
    models = list(results.keys())
    r2_test = [results[m]['metrics']['R² Score (Test)'] for m in models]
    mse_test = [results[m]['metrics']['MSE (Test)'] for m in models]
    mae_test = [results[m]['metrics']['MAE (Test)'] for m in models]
    rmse_test = [results[m]['metrics']['RMSE (Test)'] for m in models]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison (Test Set)', fontsize=16, fontweight='bold')
    
    # R² Score
    axes[0, 0].bar(models, r2_test, color='steelblue', alpha=0.7)
    axes[0, 0].set_title('R² Score (Higher is Better)', fontweight='bold')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # MSE
    axes[0, 1].bar(models, mse_test, color='coral', alpha=0.7)
    axes[0, 1].set_title('Mean Squared Error (Lower is Better)', fontweight='bold')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # MAE
    axes[1, 0].bar(models, mae_test, color='mediumseagreen', alpha=0.7)
    axes[1, 0].set_title('Mean Absolute Error (Lower is Better)', fontweight='bold')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # RMSE
    axes[1, 1].bar(models, rmse_test, color='gold', alpha=0.7)
    axes[1, 1].set_title('Root Mean Squared Error (Lower is Better)', fontweight='bold')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def get_feature_importance(model, feature_names, model_name):
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
        
    Returns:
        DataFrame with feature importance
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importances)],
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    except Exception as e:
        st.warning(f"Could not extract feature importance for {model_name}: {str(e)}")
        return None


def plot_feature_importance(importance_df, top_n=10):
    """
    Plot feature importance bar chart.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to show
        
    Returns:
        Matplotlib figure
    """
    if importance_df is None or len(importance_df) == 0:
        return None
    
    # Select top N features
    plot_df = importance_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    sns.barplot(data=plot_df, y='Feature', x='Importance', color='steelblue', ax=ax)
    
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_predictions(y_true, y_pred, title="Predicted vs Actual"):
    """
    Create scatter plot of predicted vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Calculate R² for the plot
    r2 = r2_score(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, color='steelblue', edgecolors='navy', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_title(f'{title}\nR² Score: {r2:.4f}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_residuals(y_true, y_pred):
    """
    Create residual plot.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Matplotlib figure
    """
    residuals = y_true - y_pred
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(y_pred, residuals, alpha=0.6, s=30, color='coral', edgecolors='darkred', linewidth=0.5)
    
    # Zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
    
    ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_partial_dependence(pipeline, X_train, feature_names, features_to_plot):
    """
    Create partial dependence plots for top features.
    
    Args:
        pipeline: Trained pipeline
        X_train: Training data
        feature_names: List of feature names
        features_to_plot: Indices of features to plot
        
    Returns:
        Matplotlib figure or None
    """
    try:
        # Transform data
        X_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
        model = pipeline.named_steps['model']
        
        # Calculate partial dependence
        pd_results = partial_dependence(
            model, 
            X_transformed, 
            features=features_to_plot[:2],  # Plot top 2 features
            grid_resolution=50
        )
        
        # Create subplots
        n_features = len(features_to_plot[:2])
        fig, axes = plt.subplots(1, n_features, figsize=(12, 4))
        
        if n_features == 1:
            axes = [axes]
        
        for idx, feature_idx in enumerate(features_to_plot[:2]):
            axes[idx].plot(pd_results['grid_values'][idx], pd_results['average'][idx], 
                          linewidth=2, color='steelblue')
            axes[idx].set_title(f'{feature_names[feature_idx]}', fontweight='bold')
            axes[idx].set_xlabel('Feature Value', fontsize=10)
            axes[idx].set_ylabel('Partial Dependence', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
        
        fig.suptitle('Partial Dependence Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        st.warning(f"Could not create partial dependence plot: {str(e)}")
        return None


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("📊 HistGradientBoostingRegressor: Interactive Demo & Comparison")
    st.markdown("""
    This application demonstrates how **HistGradientBoostingRegressor** works and compares it 
    with other popular regression models. Upload your own dataset or use built-in examples!
    """)
    
    # ========================================================================
    # SIDEBAR: Dataset Selection & Configuration
    # ========================================================================
    
    st.sidebar.header("⚙️ Configuration")
    
    # Dataset selection
    dataset_options = ["California Housing", "Diabetes", "Upload CSV"]
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset",
        dataset_options,
        help="Choose a built-in dataset or upload your own CSV file"
    )
    
    uploaded_file = None
    if dataset_choice == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV file with numerical/categorical features and a target column"
        )
    
    # Load data
    if dataset_choice == "Upload CSV" and uploaded_file is None:
        st.info("👈 Please upload a CSV file from the sidebar to continue.")
        return
    
    with st.spinner("Loading dataset..."):
        df, default_target = load_data(dataset_choice, uploaded_file)
    
    # Display dataset info
    st.sidebar.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Target column selection
    if dataset_choice == "Upload CSV":
        # Filter to only show numeric columns for target selection
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.error("❌ No numeric columns found in the dataset. The target column must be numeric for regression.")
            return
        
        target_col = st.sidebar.selectbox(
            "Select Target Column (must be numeric)",
            numeric_columns,
            help="Choose the numeric column you want to predict"
        )
        
        # Validate selected target
        if df[target_col].isnull().all():
            st.error(f"❌ Target column '{target_col}' contains only missing values. Please select a different column.")
            return
        
        # Show target column statistics
        st.sidebar.info(f"""
        **Target: {target_col}**
        - Min: {df[target_col].min():.2f}
        - Max: {df[target_col].max():.2f}
        - Mean: {df[target_col].mean():.2f}
        - Missing: {df[target_col].isnull().sum()} ({df[target_col].isnull().sum()/len(df)*100:.1f}%)
        """)
    else:
        target_col = default_target
        st.sidebar.info(f"Target column: **{target_col}**")
    
    # Test size
    test_size = st.sidebar.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data to use for testing"
    ) / 100
    
    # Model selection (optional)
    st.sidebar.subheader("Model Selection")
    model_options = ["HistGradientBoosting", "LinearRegression", "Ridge", "RandomForest"]
    if XGBOOST_AVAILABLE:
        model_options.append("XGBoost")
    
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        model_options,
        default=model_options,
        help="Choose which models to train and compare"
    )
    
    if not selected_models:
        st.warning("Please select at least one model to continue.")
        return
    
    # ========================================================================
    # MAIN CONTENT: Tabs
    # ========================================================================
    
    tab1, tab2, tab3 = st.tabs([
        "📚 Model Explanation & Visualization",
        "🎯 Training & Evaluation",
        "📊 Model Comparison & Insights"
    ])
    
    # ========================================================================
    # TAB 1: Model Explanation & Visualization
    # ========================================================================
    
    with tab1:
        st.header("Understanding HistGradientBoostingRegressor")
        
        # What is HistGradientBoosting?
        with st.expander("🔍 What is HistGradientBoostingRegressor?", expanded=True):
            st.markdown("""
            **HistGradientBoostingRegressor** is a modern gradient boosting algorithm inspired by LightGBM. 
            It offers several advantages over traditional gradient boosting methods:
            
            ### Key Features:
            
            1. **Histogram-based Binning** 🎯
               - Continuous features are binned into discrete bins (typically 255 bins)
               - Dramatically speeds up training by reducing the number of split points to evaluate
               - Reduces memory usage significantly
            
            2. **Native Support for Missing Values** 🔧
               - Can handle NaN values directly without imputation
               - Learns optimal direction for missing values during training
            
            3. **Native Categorical Feature Support** 📊
               - Can process categorical features without one-hot encoding
               - More efficient for high-cardinality categorical variables
            
            4. **Gradient Boosting** 🚀
               - Builds an ensemble of decision trees sequentially
               - Each tree corrects errors made by previous trees
               - Uses gradient descent to minimize loss function
            
            ### How It Works:
            
            ```
            1. Start with an initial prediction (usually mean of target)
            2. For each iteration:
               a. Calculate residuals (errors) from current predictions
               b. Bin continuous features into histograms
               c. Train a decision tree to predict residuals
               d. Update predictions by adding tree predictions (weighted by learning rate)
            3. Final prediction = sum of all tree predictions
            ```
            
            ### Mathematical Foundation:
            
            The model minimizes a loss function L using gradient descent:
            
            $$F_m(x) = F_{m-1}(x) + \\eta \\cdot h_m(x)$$
            
            Where:
            - $F_m(x)$ is the model after m iterations
            - $\\eta$ is the learning rate
            - $h_m(x)$ is the m-th decision tree predicting negative gradients
            """)
        
        # Comparison with other models
        with st.expander("⚖️ How does it compare to other models?"):
            st.markdown("""
            | Feature | HistGradientBoosting | RandomForest | XGBoost | LinearRegression |
            |---------|---------------------|--------------|---------|------------------|
            | **Speed** | ⚡ Very Fast | 🚀 Fast | ⚡ Fast | ⚡⚡ Very Fast |
            | **Memory** | 💾 Low | 💾💾 Medium | 💾 Low | 💾 Very Low |
            | **Missing Values** | ✅ Native | ❌ Needs imputation | ✅ Native | ❌ Needs imputation |
            | **Categorical Features** | ✅ Native | ❌ Needs encoding | ✅ With encoding | ❌ Needs encoding |
            | **Overfitting Risk** | 🟡 Medium | 🟡 Medium | 🔴 High | 🟢 Low |
            | **Interpretability** | 🟡 Medium | 🟡 Medium | 🟡 Medium | 🟢 High |
            | **Performance** | 🟢 Excellent | 🟢 Excellent | 🟢 Excellent | 🟡 Good for linear data |
            
            ### When to use HistGradientBoostingRegressor?
            
            ✅ **Best for:**
            - Large datasets (>10,000 samples)
            - Mix of numerical and categorical features
            - Datasets with missing values
            - When you need a balance of speed and accuracy
            
            ❌ **Not ideal for:**
            - Very small datasets (<1,000 samples)
            - When maximum interpretability is required
            - Purely linear relationships
            """)
        
        # Architecture visualization
        with st.expander("🏗️ Algorithm Architecture"):
            st.markdown("""
            ### Gradient Boosting Process:
            
            ```
            Initial Model: F₀(x) = mean(y)
                    ↓
            Iteration 1: Calculate residuals r₁ = y - F₀(x)
                    ↓
            Tree h₁(x) learns to predict r₁
                    ↓
            Update: F₁(x) = F₀(x) + η·h₁(x)
                    ↓
            Iteration 2: Calculate residuals r₂ = y - F₁(x)
                    ↓
            Tree h₂(x) learns to predict r₂
                    ↓
            Update: F₂(x) = F₁(x) + η·h₂(x)
                    ↓
            ... repeat for max_iter iterations ...
                    ↓
            Final Model: F(x) = F₀(x) + η·Σhᵢ(x)
            ```
            
            ### Histogram Binning Example:
            
            Instead of evaluating all possible split points for a feature:
            
            **Traditional Method:**
            - Feature values: [1.2, 1.5, 1.8, 2.1, 2.4, ..., 100.5] (1000 values)
            - Must evaluate ~1000 split points ❌ Slow!
            
            **Histogram Method:**
            - Bin into 255 bins: [1-5]: bin 0, [5-10]: bin 1, ...
            - Only evaluate 255 split points ✅ Fast!
            """)
        
        # Hyperparameter guide
        with st.expander("🎛️ Key Hyperparameters"):
            st.markdown("""
            ### Important Hyperparameters:
            
            1. **max_iter** (default: 100)
               - Number of boosting iterations (trees to build)
               - More iterations = better fit but risk of overfitting
               - Typical range: 50-500
            
            2. **learning_rate** (default: 0.1)
               - Shrinkage parameter (η in formula)
               - Lower values need more iterations but often generalize better
               - Typical range: 0.01-0.3
            
            3. **max_depth** (default: None)
               - Maximum depth of each tree
               - Controls model complexity
               - Typical range: 3-10
            
            4. **max_bins** (default: 255)
               - Maximum number of bins for histogram
               - More bins = finer splits but slower training
               - Typical range: 128-255
            
            5. **min_samples_leaf** (default: 20)
               - Minimum samples required in a leaf node
               - Regularization parameter to prevent overfitting
               - Typical range: 10-100
            
            ### Tuning Tips:
            - Start with defaults
            - If overfitting: reduce max_iter, increase min_samples_leaf, reduce max_depth
            - If underfitting: increase max_iter, reduce min_samples_leaf, increase max_depth
            - Adjust learning_rate inversely with max_iter
            """)
    
    # ========================================================================
    # TAB 2: Training & Evaluation
    # ========================================================================
    
    with tab2:
        st.header("Model Training & Evaluation")
        
        # Show dataset preview
        st.subheader("📋 Dataset Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Samples", df.shape[0])
            st.metric("Features", df.shape[1] - 1)
        
        with col2:
            st.metric("Target Column", target_col)
            st.metric("Missing Values", df.isnull().sum().sum())
        
        with st.expander("View Data Sample"):
            st.dataframe(df.head(10), width='stretch')
        
        # Data preprocessing
        st.subheader("🔧 Data Preprocessing")
        
        with st.spinner("Preprocessing data..."):
            X, y, preprocessor, numerical_cols, categorical_cols = preprocess_data(df, target_col)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Numerical Features", len(numerical_cols))
        with col2:
            st.metric("Categorical Features", len(categorical_cols))
        with col3:
            st.metric("Total Features", len(numerical_cols) + len(categorical_cols))
        
        with st.expander("Feature Details"):
            if numerical_cols:
                st.write("**Numerical Features:**", numerical_cols)
            if categorical_cols:
                st.write("**Categorical Features:**", categorical_cols)
        
        # Train-test split
        st.subheader("📊 Train-Test Split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Test Samples", len(X_test))
        
        # Build pipelines
        st.subheader("🔨 Building Model Pipelines")
        
        with st.spinner("Building pipelines..."):
            all_pipelines = build_pipelines(preprocessor)
            # Filter based on user selection
            pipelines = {k: v for k, v in all_pipelines.items() if k in selected_models}
        
        st.success(f"✅ Built {len(pipelines)} model pipeline(s)")
        
        # Train models
        st.subheader("🎓 Training Models")
        
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training models... This may take a moment."):
                results = train_and_evaluate(pipelines, X_train, y_train, X_test, y_test)
                st.session_state['results'] = results
                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['numerical_cols'] = numerical_cols
                st.session_state['categorical_cols'] = categorical_cols
            
            st.success("✅ Training completed!")
        
        # Display results if available
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            st.subheader("📈 Model Performance Metrics")
            
            # Create metrics DataFrame
            metrics_data = []
            for model_name, result in results.items():
                metrics_data.append({
                    'Model': model_name,
                    'R² (Train)': f"{result['metrics']['R² Score (Train)']:.4f}",
                    'R² (Test)': f"{result['metrics']['R² Score (Test)']:.4f}",
                    'RMSE (Train)': f"{result['metrics']['RMSE (Train)']:.4f}",
                    'RMSE (Test)': f"{result['metrics']['RMSE (Test)']:.4f}",
                    'MAE (Test)': f"{result['metrics']['MAE (Test)']:.4f}"
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, width='stretch')
            
            # Highlight best model
            best_r2_model = max(results.items(), key=lambda x: x[1]['metrics']['R² Score (Test)'])
            st.info(f"🏆 **Best Model (by R² Score):** {best_r2_model[0]} with R² = {best_r2_model[1]['metrics']['R² Score (Test)']:.4f}")
            
            # Model selection for detailed visualization
            st.subheader("🔍 Detailed Model Analysis")
            selected_model = st.selectbox(
                "Select Model for Detailed Visualization",
                list(results.keys())
            )
            
            if selected_model:
                model_result = results[selected_model]
                pipeline = model_result['pipeline']
                
                # Feature importance
                st.markdown("### 📊 Feature Importance")
                
                # Get feature names after preprocessing
                try:
                    # Get transformed feature names
                    feature_names = []
                    if numerical_cols:
                        feature_names.extend(numerical_cols)
                    if categorical_cols:
                        # Get encoded feature names
                        encoder = pipeline.named_steps['preprocessor'].named_transformers_.get('cat')
                        if encoder and hasattr(encoder.named_steps['encoder'], 'get_feature_names_out'):
                            cat_features = encoder.named_steps['encoder'].get_feature_names_out(categorical_cols)
                            feature_names.extend(cat_features)
                    
                    model = pipeline.named_steps['model']
                    importance_df = get_feature_importance(model, feature_names, selected_model)
                    
                    if importance_df is not None:
                        fig_importance = plot_feature_importance(importance_df, top_n=15)
                        if fig_importance:
                            st.pyplot(fig_importance)
                            st.caption("Feature importance shows which features have the most influence on predictions.")
                    else:
                        st.info("Feature importance not available for this model.")
                
                except Exception as e:
                    st.warning(f"Could not display feature importance: {str(e)}")
                
                # Predicted vs Actual
                st.markdown("### 🎯 Predicted vs Actual Values")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_train = plot_predictions(
                        st.session_state['y_train'],
                        model_result['predictions_train'],
                        f"{selected_model} - Training Set"
                    )
                    st.pyplot(fig_train)
                
                with col2:
                    fig_test = plot_predictions(
                        st.session_state['y_test'],
                        model_result['predictions_test'],
                        f"{selected_model} - Test Set"
                    )
                    st.pyplot(fig_test)
                
                st.caption("Points closer to the red dashed line indicate better predictions. The R² score measures how well predictions match actual values.")
                
                # Residual plot
                st.markdown("### 📉 Residual Analysis")
                fig_residuals = plot_residuals(
                    st.session_state['y_test'],
                    model_result['predictions_test']
                )
                st.pyplot(fig_residuals)
                st.caption("Residuals should be randomly scattered around zero. Patterns in residuals indicate model bias.")
                
                # Partial dependence (for tree-based models)
                if selected_model in ['HistGradientBoosting', 'RandomForest', 'XGBoost']:
                    st.markdown("### 🔄 Partial Dependence Plots")
                    
                    if importance_df is not None and len(importance_df) >= 2:
                        # Get indices of top 2 features
                        top_features = importance_df.head(2)['Feature'].tolist()
                        
                        # Find indices in original feature list
                        feature_indices = [i for i, f in enumerate(feature_names) if f in top_features]
                        
                        if len(feature_indices) >= 2:
                            fig_pd = plot_partial_dependence(
                                pipeline,
                                st.session_state['X_train'],
                                feature_names,
                                feature_indices
                            )
                            
                            if fig_pd:
                                st.pyplot(fig_pd)
                                st.caption("Partial dependence plots show the marginal effect of features on predictions, holding other features constant.")
    
    # ========================================================================
    # TAB 3: Model Comparison & Insights
    # ========================================================================
    
    with tab3:
        st.header("Model Comparison & Insights")
        
        if 'results' not in st.session_state:
            st.info("👈 Please train the models first in the 'Training & Evaluation' tab.")
            return
        
        results = st.session_state['results']
        
        # Comparison chart
        st.subheader("📊 Performance Comparison")
        fig_comparison = plot_comparison(results)
        st.pyplot(fig_comparison)
        
        # Detailed metrics comparison
        st.subheader("📋 Detailed Metrics Comparison")
        
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name,
                'R² Score (Test)': result['metrics']['R² Score (Test)'],
                'RMSE (Test)': result['metrics']['RMSE (Test)'],
                'MAE (Test)': result['metrics']['MAE (Test)'],
                'Overfit Gap (R²)': result['metrics']['R² Score (Train)'] - result['metrics']['R² Score (Test)']
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('R² Score (Test)', ascending=False)
        
        # Style the dataframe
        st.dataframe(
            comparison_df.style.format({
                'R² Score (Test)': '{:.4f}',
                'RMSE (Test)': '{:.4f}',
                'MAE (Test)': '{:.4f}',
                'Overfit Gap (R²)': '{:.4f}'
            }).background_gradient(subset=['R² Score (Test)'], cmap='RdYlGn')
            .background_gradient(subset=['RMSE (Test)', 'MAE (Test)'], cmap='RdYlGn_r'),
            width='stretch'
        )
        
        # Insights
        st.subheader("💡 Key Insights")
        
        best_model = comparison_df.iloc[0]
        worst_model = comparison_df.iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **🏆 Best Performing Model: {best_model['Model']}**
            - R² Score: {best_model['R² Score (Test)']:.4f}
            - RMSE: {best_model['RMSE (Test)']:.4f}
            - Overfitting Gap: {best_model['Overfit Gap (R²)']:.4f}
            """)
        
        with col2:
            st.warning(f"""
            **📉 Lowest Performing Model: {worst_model['Model']}**
            - R² Score: {worst_model['R² Score (Test)']:.4f}
            - RMSE: {worst_model['RMSE (Test)']:.4f}
            - Overfitting Gap: {worst_model['Overfit Gap (R²)']:.4f}
            """)
        
        # Analysis
        with st.expander("📊 Detailed Analysis"):
            st.markdown(f"""
            ### Performance Analysis
            
            **1. Model Ranking (by R² Score):**
            """)
            
            for idx, row in comparison_df.iterrows():
                st.markdown(f"   {comparison_df.index.get_loc(idx) + 1}. **{row['Model']}** - R²: {row['R² Score (Test)']:.4f}")
            
            st.markdown("""
            **2. Overfitting Analysis:**
            
            The 'Overfit Gap' shows the difference between training and test R² scores:
            - **< 0.05**: Well-generalized model ✅
            - **0.05 - 0.10**: Moderate overfitting ⚠️
            - **> 0.10**: Significant overfitting ❌
            """)
            
            for _, row in comparison_df.iterrows():
                gap = row['Overfit Gap (R²)']
                if gap < 0.05:
                    status = "✅ Well-generalized"
                elif gap < 0.10:
                    status = "⚠️ Moderate overfitting"
                else:
                    status = "❌ Overfitting"
                st.markdown(f"   - **{row['Model']}**: Gap = {gap:.4f} - {status}")
        
        # Recommendations
        st.subheader("💭 Recommendations")
        
        with st.expander("🎯 Model Selection Recommendations", expanded=True):
            if 'HistGradientBoosting' in results:
                hgb_result = results['HistGradientBoosting']['metrics']
                st.markdown(f"""
                ### HistGradientBoostingRegressor Performance
                
                - **R² Score:** {hgb_result['R² Score (Test)']:.4f}
                - **RMSE:** {hgb_result['RMSE (Test)']:.4f}
                
                **When to use HistGradientBoosting:**
                - ✅ Large datasets with many features
                - ✅ Mix of numerical and categorical features
                - ✅ Presence of missing values
                - ✅ Need for fast training with good performance
                - ✅ Memory constraints
                """)
            
            st.markdown("""
            ### General Recommendations:
            
            1. **For Production Use:**
               - Choose the model with best test R² score and low overfitting gap
               - Consider training time and inference speed
               - Evaluate model interpretability requirements
            
            2. **For Further Improvement:**
               - Perform hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
               - Try feature engineering (polynomial features, interactions)
               - Collect more data if possible
               - Handle outliers and anomalies
               - Consider ensemble methods (combining multiple models)
            
            3. **If Models Underperform:**
               - Check for data quality issues
               - Verify feature relevance
               - Try different feature scaling methods
               - Consider non-linear transformations
               - Check for data leakage
            """)
        
        # Download options
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download metrics as CSV
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics (CSV)",
                data=csv,
                file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download best model
            if st.button("💾 Save Best Model"):
                best_model_name = comparison_df.iloc[0]['Model']
                best_pipeline = results[best_model_name]['pipeline']
                
                # Save to bytes
                buffer = io.BytesIO()
                joblib.dump(best_pipeline, buffer)
                buffer.seek(0)
                
                st.download_button(
                    label=f"📥 Download {best_model_name} Model",
                    data=buffer,
                    file_name=f"{best_model_name}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream"
                )


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
