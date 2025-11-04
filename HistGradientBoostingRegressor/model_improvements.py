"""
Model Improvement Strategies for Better Performance
Apply these changes to reduce overfitting and improve test scores
"""

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

# ============================================================================
# STRATEGY 1: IMPROVED HYPERPARAMETERS (Reduce Overfitting)
# ============================================================================

def build_improved_pipelines(preprocessor):
    """
    Build pipelines with optimized hyperparameters to reduce overfitting
    """
    pipelines = {
        # HistGradientBoosting - Add regularization
        'HistGradientBoosting': Pipeline([
            ('preprocessor', preprocessor),
            ('model', HistGradientBoostingRegressor(
                max_iter=100,              # Keep same
                learning_rate=0.05,        # REDUCED from 0.1 (slower learning = less overfit)
                max_depth=4,               # REDUCED from 5 (simpler trees)
                min_samples_leaf=30,       # INCREASED from 20 (more regularization)
                l2_regularization=1.0,     # ADDED (L2 penalty)
                max_bins=200,              # REDUCED from 255 (less granular splits)
                early_stopping=True,       # ADDED (stop when validation doesn't improve)
                n_iter_no_change=10,       # Stop after 10 iterations without improvement
                validation_fraction=0.1,   # Use 10% of training for validation
                random_state=42
            ))
        ]),
        
        # LinearRegression - Use Ridge (L2 regularization)
        'Ridge': Pipeline([
            ('preprocessor', preprocessor),
            ('model', Ridge(
                alpha=10.0,                # Regularization strength
                random_state=42
            ))
        ]),
        
        # RandomForest - Strong regularization
        'RandomForest': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(
                n_estimators=100,          # Keep same
                max_depth=6,               # REDUCED from 10 (prevent deep trees)
                min_samples_split=20,      # INCREASED from 2 (more conservative splits)
                min_samples_leaf=10,       # INCREASED from 1 (larger leaf nodes)
                max_features='sqrt',       # CHANGED from 'auto' (use fewer features)
                bootstrap=True,            # Ensure bootstrap sampling
                oob_score=True,            # Out-of-bag score for validation
                random_state=42,
                n_jobs=-1
            ))
        ]),
        
        # XGBoost - Add strong regularization
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(
                n_estimators=100,
                learning_rate=0.03,        # REDUCED from 0.1 (much slower)
                max_depth=3,               # REDUCED from 5 (shallower trees)
                min_child_weight=5,        # INCREASED from 1 (more regularization)
                subsample=0.7,             # ADDED (use 70% of samples per tree)
                colsample_bytree=0.7,      # ADDED (use 70% of features per tree)
                gamma=0.1,                 # ADDED (min loss reduction for split)
                reg_alpha=0.5,             # ADDED (L1 regularization)
                reg_lambda=1.0,            # ADDED (L2 regularization)
                early_stopping_rounds=10,  # ADDED (stop if no improvement)
                random_state=42
            ))
        ])
    }
    
    return pipelines


# ============================================================================
# STRATEGY 2: FEATURE ENGINEERING
# ============================================================================

def create_polynomial_features(X, degree=2):
    """
    Create polynomial features to capture non-linear relationships
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly, poly.get_feature_names_out()


def remove_outliers(X, y, contamination=0.1):
    """
    Remove outliers using Isolation Forest
    """
    from sklearn.ensemble import IsolationForest
    
    iso = IsolationForest(contamination=contamination, random_state=42)
    mask = iso.fit_predict(X) == 1  # 1 means inlier
    
    return X[mask], y[mask]


def select_best_features(X, y, k=10):
    """
    Select top K features using mutual information
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    
    selector = SelectKBest(mutual_info_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    
    return X_selected, selector


# ============================================================================
# STRATEGY 3: ROBUST PREPROCESSING
# ============================================================================

def build_robust_preprocessor(numerical_cols, categorical_cols):
    """
    Use RobustScaler instead of StandardScaler (better for outliers)
    """
    from sklearn.preprocessing import RobustScaler
    
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # CHANGED from StandardScaler
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    transformers = []
    if numerical_cols:
        transformers.append(('num', numerical_pipeline, numerical_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_pipeline, categorical_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    return preprocessor


# ============================================================================
# STRATEGY 4: CROSS-VALIDATION
# ============================================================================

def evaluate_with_cv(pipeline, X, y, cv=5):
    """
    Use cross-validation for more reliable performance estimates
    """
    cv_scores = cross_val_score(
        pipeline, X, y, 
        cv=cv, 
        scoring='r2',
        n_jobs=-1
    )
    
    return {
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'all_cv_scores': cv_scores
    }


# ============================================================================
# STRATEGY 5: ENSEMBLE METHODS
# ============================================================================

def create_voting_regressor(pipelines):
    """
    Combine multiple models using voting
    """
    from sklearn.ensemble import VotingRegressor
    
    # Extract just the models (not full pipelines for voting)
    estimators = [
        ('hist', pipelines['HistGradientBoosting']),
        ('ridge', pipelines['Ridge']),
        ('rf', pipelines['RandomForest'])
    ]
    
    voting = VotingRegressor(estimators=estimators)
    
    return voting


def create_stacking_regressor(pipelines, meta_learner=None):
    """
    Stack multiple models with a meta-learner
    """
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge
    
    if meta_learner is None:
        meta_learner = Ridge(alpha=1.0)
    
    estimators = [
        ('hist', pipelines['HistGradientBoosting']),
        ('ridge', pipelines['Ridge']),
        ('rf', pipelines['RandomForest'])
    ]
    
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5
    )
    
    return stacking


# ============================================================================
# STRATEGY 6: DATA QUALITY IMPROVEMENTS
# ============================================================================

def check_data_quality(X, y):
    """
    Check for common data quality issues
    """
    issues = []
    
    # Check for duplicate rows
    duplicates = X.duplicated().sum()
    if duplicates > 0:
        issues.append(f"⚠️ Found {duplicates} duplicate rows")
    
    # Check for constant features
    constant_features = X.columns[X.nunique() <= 1].tolist()
    if constant_features:
        issues.append(f"⚠️ Found {len(constant_features)} constant features: {constant_features}")
    
    # Check for high correlation
    if hasattr(X, 'corr'):
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        if high_corr:
            issues.append(f"⚠️ Found {len(high_corr)} highly correlated features: {high_corr}")
    
    # Check target distribution
    target_skew = y.skew() if hasattr(y, 'skew') else 0
    if abs(target_skew) > 1:
        issues.append(f"⚠️ Target is highly skewed (skewness: {target_skew:.2f}). Consider log transform.")
    
    return issues


def apply_target_transform(y):
    """
    Apply log transform to target if it's skewed
    """
    import numpy as np
    
    # Add small constant to avoid log(0)
    y_transformed = np.log1p(y)  # log(1 + y)
    
    return y_transformed


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """
    How to apply these improvements
    """
    
    # Example 1: Use improved hyperparameters
    print("=" * 50)
    print("EXAMPLE 1: Improved Hyperparameters")
    print("=" * 50)
    print("""
    # In your app.py, replace build_pipelines() function with:
    pipelines = build_improved_pipelines(preprocessor)
    
    # This will use regularized models with:
    # - Lower learning rates
    # - Shallower trees
    # - More samples per leaf
    # - L1/L2 regularization
    # - Early stopping
    """)
    
    # Example 2: Check data quality
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Data Quality Check")
    print("=" * 50)
    print("""
    # Add this before training:
    issues = check_data_quality(X, y)
    for issue in issues:
        print(issue)
    
    # Remove duplicates
    X = X.drop_duplicates()
    y = y[X.index]
    
    # Remove constant features
    constant_cols = X.columns[X.nunique() <= 1]
    X = X.drop(columns=constant_cols)
    """)
    
    # Example 3: Use cross-validation
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Cross-Validation")
    print("=" * 50)
    print("""
    # Instead of single train-test split, use CV:
    cv_results = evaluate_with_cv(pipeline, X_train, y_train, cv=5)
    print(f"CV R² Score: {cv_results['mean_cv_score']:.4f} (+/- {cv_results['std_cv_score']:.4f})")
    """)
    
    # Example 4: Feature engineering
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Feature Engineering")
    print("=" * 50)
    print("""
    # Create polynomial features:
    X_poly, feature_names = create_polynomial_features(X, degree=2)
    
    # Or select best features:
    X_selected, selector = select_best_features(X, y, k=15)
    """)
    
    # Example 5: Ensemble methods
    print("\n" + "=" * 50)
    print("EXAMPLE 5: Ensemble Methods")
    print("=" * 50)
    print("""
    # Create voting ensemble:
    voting_model = create_voting_regressor(pipelines)
    voting_model.fit(X_train, y_train)
    
    # Or stacking:
    stacking_model = create_stacking_regressor(pipelines)
    stacking_model.fit(X_train, y_train)
    """)


if __name__ == "__main__":
    example_usage()
