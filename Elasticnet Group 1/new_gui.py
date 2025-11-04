import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing, load_diabetes

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Regularization Model Explorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ App Header ------------------
st.markdown('<p class="main-header">🎯 Regularization Model Explorer</p>', unsafe_allow_html=True)
st.markdown(
    "Compare **Ridge**, **Lasso**, and **ElasticNet** models on multiple datasets. "
    "Train models and click the tabs to compare their performance!"
)

# ------------------ Initialize Session State ------------------
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
# --- OPTIMIZATION ---
# This dictionary will store each dataset *once*
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

# ------------------ Dataset Loading Functions ------------------
@st.cache_data
def load_california_housing_data():
    """Loads the California Housing dataset."""
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="MedHouseVal")
    description = "Predict California housing prices based on location and demographics (20,640 samples, 8 features)"
    return X, y, "California Housing", description

@st.cache_data
def load_diabetes_data():
    """Loads the Diabetes dataset."""
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name="Disease_Progression")
    description = "Predict diabetes disease progression from medical measurements (442 samples, 10 features)"
    return X, y, "Diabetes Progression", description

@st.cache_data
def load_boston_synthetic():
    """Creates a synthetic housing dataset similar to Boston Housing."""
    np.random.seed(42)
    n_samples = 506
    
    CRIM = np.random.exponential(3.5, n_samples)
    ZN = np.random.choice([0, 12.5, 25, 50, 80], n_samples)
    INDUS = np.random.uniform(0.5, 27, n_samples)
    NOX = np.random.uniform(0.3, 0.9, n_samples)
    RM = np.random.normal(6.3, 0.7, n_samples)
    AGE = np.random.uniform(10, 100, n_samples)
    DIS = np.random.uniform(1, 12, n_samples)
    RAD = np.random.choice([1, 2, 3, 4, 5, 8, 24], n_samples)
    TAX = np.random.uniform(180, 710, n_samples)
    PTRATIO = np.random.uniform(12, 22, n_samples)
    LSTAT = np.random.uniform(2, 38, n_samples)
    
    y = (9 * RM - 0.3 * LSTAT - 0.1 * CRIM + 0.02 * ZN - 
         15 * NOX - 0.05 * AGE + 1.5 * DIS - 0.01 * TAX - 
         0.5 * PTRATIO + np.random.normal(0, 3, n_samples))
    y = np.clip(y, 5, 50)
    
    X = pd.DataFrame({
        'CRIM': CRIM, 'ZN': ZN, 'INDUS': INDUS, 'NOX': NOX,
        'RM': RM, 'AGE': AGE, 'DIS': DIS, 'RAD': RAD,
        'TAX': TAX, 'PTRATIO': PTRATIO, 'LSTAT': LSTAT
    })
    y = pd.Series(y, name="MEDV")
    description = "Synthetic housing dataset with 11 features (506 samples, similar to Boston Housing)"
    return X, y, "Synthetic Housing", description

def load_user_dataset(uploaded_file):
    """Loads a user-uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        
        if len(df) < 10:
            st.error("Dataset must have at least 10 rows")
            return None, None, None, None
        
        if len(df.columns) < 2:
            st.error("Dataset must have at least 2 columns (features + target)")
            return None, None, None, None
        
        st.info("✅ Dataset loaded successfully!")
        target_col = st.selectbox(
            "Select the target (y) column:",
            options=df.columns.tolist(),
            key="target_selector"
        )
        
        if target_col:
            y = df[target_col]
            X = df.drop(columns=[target_col])
            
            non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                st.warning(f"Dropping non-numeric columns: {non_numeric}")
                X = X.select_dtypes(include=[np.number])
            
            if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
                st.warning("Dropping rows with missing values")
                X = X.dropna()
                y = y[X.index]
            
            name = uploaded_file.name.replace('.csv', '')
            description = f"User uploaded dataset ({len(X)} samples, {X.shape[1]} features)"
            return X, y, name, description
        
        return None, None, None, None
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None, None, None

# ------------------ Model Training Function ------------------
@st.cache_data
def train_model_on_dataset(X, y, model_type, alpha, l1_ratio, test_size):
    """
    Trains a model on the provided dataset.
    """
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    if model_type == "Ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "Lasso":
        model = Lasso(alpha=alpha, max_iter=10000)
    else: # ElasticNet
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return model, scaler, train_mse, test_mse, train_r2, test_r2, X.columns.tolist()

@st.cache_data
def get_regression_curves(X, y, model_type, l1_ratio, test_size):
    """
    Calculates error curves for a range of alphas.
    """
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    alphas = np.logspace(-2, 1, 50)  
    train_errors, test_errors = [], []
    
    for a in alphas:
        if model_type == "Ridge":
            model = Ridge(alpha=a)
        elif model_type == "Lasso":
            model = Lasso(alpha=a, max_iter=10000)
        else: # ElasticNet
            model = ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=10000)
            
        model.fit(X_train_scaled, y_train)
        train_errors.append(mean_squared_error(y_train, model.predict(X_train_scaled)))
        test_errors.append(mean_squared_error(y_test, model.predict(X_test_scaled)))
        
    return alphas, train_errors, test_errors

def get_step(series):
    """Calculates an appropriate step size for st.number_input."""
    if pd.api.types.is_integer_dtype(series):
        return 1.0
    
    range_val = series.max() - series.min()
    if range_val == 0:
        return 0.01
    
    if range_val < 2:
        return 0.01
    elif range_val < 20:
        return 0.1
    elif range_val < 200:
        return 1.0
    else:
        return 10.0

# ------------------ Sidebar: Dataset Selection ------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/data-configuration.png", width=80)
    st.title("📊 Dataset Manager")
    
    st.markdown("---")
    st.header("1️⃣ Select/Upload Dataset")
    
    dataset_option = st.radio(
        "Choose data source:",
        ["Built-in Datasets", "Upload Your Own CSV"],
        key="dataset_source"
    )
    
    if dataset_option == "Built-in Datasets":
        builtin_choice = st.selectbox(
            "Select a built-in dataset:",
            ["California Housing", "Diabetes Progression", "Synthetic Housing"],
            key="builtin_dataset"
        )
        
        if builtin_choice == "California Housing":
            X, y, name, desc = load_california_housing_data()
        elif builtin_choice == "Diabetes Progression":
            X, y, name, desc = load_diabetes_data()
        else:
            X, y, name, desc = load_boston_synthetic()
        
        st.success(f"✅ Loaded: {name}")
        st.info(desc)
        
    else:  # Upload CSV
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file:
            X, y, name, desc = load_user_dataset(uploaded_file)
            if X is not None:
                st.success(f"✅ Loaded: {name}")
                st.info(desc)
        else:
            st.warning("⬆️ Please upload a CSV file")
            X, y, name, desc = None, None, None, None
    
    # --- OPTIMIZATION ---
    # Store the dataset *once* in the session state 'datasets' dict
    if X is not None and y is not None:
        st.session_state.datasets[name] = { 'X': X, 'y': y, 'description': desc }
        current_dataset_name = name
    else:
        current_dataset_name = None
    
    st.markdown("---")
    
    if current_dataset_name:
        st.header("2️⃣ Train Model")
        
        model_type = st.selectbox(
            "Regularization Type",
            ("ElasticNet", "Lasso", "Ridge"),
            help="Choose the regularization method",
            key="model_type"
        )
        
        alpha = st.slider(
            "Alpha (λ) - Regularization Strength",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            help="Higher values = stronger regularization",
            key="alpha_slider"
        )
        
        if model_type == "ElasticNet":
            l1_ratio = st.slider(
                "L1 Ratio",
                0.0, 1.0, 0.5,
                help="0 = Ridge (L2), 1 = Lasso (L1)",
                key="l1_ratio_slider"
            )
        else:
            l1_ratio = 0.0 if model_type == "Ridge" else 1.0
        
        test_size = st.slider(
            "Test Set Size (%)",
            10, 50, 20,
            help="Percentage of data for testing",
            key="test_size_slider"
        ) / 100
        
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner(f"Training {model_type} on {current_dataset_name}..."):
                try:
                    # --- OPTIMIZATION ---
                    # Retrieve the single copy of the dataset
                    X_current = st.session_state.datasets[current_dataset_name]['X']
                    y_current = st.session_state.datasets[current_dataset_name]['y']
                    
                    model, scaler, train_mse, test_mse, train_r2, test_r2, features = train_model_on_dataset(
                        X_current,
                        y_current,
                        model_type,
                        alpha,
                        l1_ratio,
                        test_size
                    )
                    
                    if model_type == "ElasticNet":
                        model_key = f"{current_dataset_name}_{model_type} (α={alpha:.2f}, L1={l1_ratio:.2f})"
                    else:
                        model_key = f"{current_dataset_name}_{model_type} (α={alpha:.2f})"
                    
                    # --- OPTIMIZATION ---
                    # Don't store X and y again. Just store the model, scaler, metrics,
                    # and the *name* of the dataset.
                    st.session_state.trained_models[model_key] = {
                        'model': model,
                        'scaler': scaler,
                        'model_type': model_type,
                        'alpha': alpha,
                        'l1_ratio': l1_ratio,
                        'test_size': test_size,
                        'train_mse': train_mse,
                        'test_mse': test_mse,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'features': features,
                        'dataset_name': current_dataset_name, # Store the name
                        # 'X': X_current, # <-- REMOVED (Saves Memory)
                        # 'y': y_current  # <-- REMOVED (Saves Memory)
                    }
                    
                    st.success(f"✅ Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"Training error: {str(e)}")

# ------------------ Main Content ------------------
if not st.session_state.trained_models:
    st.info("👈 Select a dataset and train a model to get started!")
    
    if current_dataset_name and current_dataset_name in st.session_state.datasets:
        st.markdown('<p class="sub-header">Dataset Preview</p>', unsafe_allow_html=True)
        # --- OPTIMIZATION ---
        # Retrieve dataset from the 'datasets' dict
        X_preview = st.session_state.datasets[current_dataset_name]['X']
        y_preview = st.session_state.datasets[current_dataset_name]['y']
        df_preview = pd.concat([X_preview, y_preview], axis=1)
        st.dataframe(df_preview.head(10), use_container_width=True) # Added container width
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Samples", len(X_preview))
        with col2: st.metric("Features", X_preview.shape[1])
        with col3: st.metric("Target Mean", f"{y_preview.mean():.2f}")

else:
    model_keys = list(st.session_state.trained_models.keys())
    tabs = st.tabs([f"📊 {key}" for key in model_keys])
    
    for idx, (model_key, tab) in enumerate(zip(model_keys, tabs)):
        with tab:
            model_info = st.session_state.trained_models[model_key]
            
            # --- OPTIMIZATION ---
            # Retrieve the correct X and y data using the stored dataset_name
            dataset_name = model_info['dataset_name']
            X_data = st.session_state.datasets[dataset_name]['X']
            y_data = st.session_state.datasets[dataset_name]['y']
            
            header_cols = [3, 1]
            if model_info['model_type'] == "ElasticNet":
                header_cols.append(1)
            header_cols.append(1)
            
            col1, *metric_cols, col_last = st.columns(header_cols)
            
            with col1:
                st.markdown(f"### {model_info['dataset_name']} — **{model_info['model_type']}**")
            with metric_cols[0]:
                st.metric("Alpha (λ)", f"{model_info['alpha']:.2f}")
            if model_info['model_type'] == "ElasticNet":
                with metric_cols[1]:
                    st.metric("L1 Ratio", f"{model_info['l1_ratio']:.2f}")
            
            with col_last:
                if st.button("🗑️ Remove Model", key=f"remove_{model_key}"):
                    del st.session_state.trained_models[model_key]
                    st.rerun()
            
            st.markdown("---")
            
            perf_tab, coef_tab, pred_tab = st.tabs([
                "📈 Performance", "⚖️ Coefficients", "🔮 Prediction"
            ])
            
            with perf_tab:
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Train MSE", f"{model_info['train_mse']:.4f}")
                with col2: st.metric("Test MSE", f"{model_info['test_mse']:.4f}")
                with col3: st.metric("Train R²", f"{model_info['train_r2']:.4f}")
                with col4: st.metric("Test R²", f"{model_info['test_r2']:.4f}")
                
                # --- OPTIMIZATION ---
                # Pass the retrieved X_data and y_data to the cached function
                alphas, train_errors, test_errors = get_regression_curves(
                    X_data,
                    y_data,
                    model_info['model_type'],
                    model_info['l1_ratio'],
                    model_info['test_size']
                )
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(alphas, train_errors, label="Training Error", marker='o', lw=2, ms=4, color='#2ecc71')
                ax.plot(alphas, test_errors, label="Testing Error", marker='s', lw=2, ms=4, color='#e74c3c')
                
                ax.axvline(x=model_info['alpha'], color='purple', linestyle='--', lw=2, 
                           label=f'Current α = {model_info["alpha"]:.2f}')
                ax.scatter([model_info['alpha']], [model_info['train_mse']], 
                           color='purple', s=200, zorder=5, marker='o', edgecolors='white', lw=2)
                ax.scatter([model_info['alpha']], [model_info['test_mse']], 
                           color='purple', s=200, zorder=5, marker='s', edgecolors='white', lw=2)
                
                ax.set_xscale("log")
                ax.set_xlabel("Alpha (λ) - Regularization Strength", fontsize=12, fontweight='bold')
                ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12, fontweight='bold')
                
                if model_info['model_type'] == "ElasticNet":
                    title = f"ElasticNet Regression — Effect of Regularization (L1 Ratio = {model_info['l1_ratio']:.2f})"
                else:
                    title = f"{model_info['model_type']} Regression — Effect of Regularization"
                ax.set_title(title, fontsize=14, fontweight='bold')
                
                ax.legend(fontsize=10, loc='best')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, use_container_width=True) # Added container width
                
                best_alpha_idx = np.argmin(test_errors)
                best_alpha = alphas[best_alpha_idx]
                min_test_error = test_errors[best_alpha_idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    **📊 Current Model:**
                    - Alpha: **{model_info['alpha']:.2f}**
                    - Train/Test Gap: **{abs(model_info['train_mse'] - model_info['test_mse']):.4f}**
                    """)
                with col2:
                    st.markdown(f"""
                    **💡 Optimal Alpha (from scan):**
                    - Best α: **{best_alpha:.2f}**
                    - Min Test Error: **{min_test_error:.4f}**
                    """)
            
            with coef_tab:
                coefs = model_info['model'].coef_
                intercept = model_info['model'].intercept_
                non_zero = np.sum(np.abs(coefs) > 1e-6)
                
                st.markdown(f"""
                **Feature Selection:** {non_zero} / {len(model_info['features'])} features active
                """)
                
                coef_df = pd.DataFrame({'Feature': model_info['features'], 'Weight': coefs})
                coef_df = coef_df.sort_values('Weight', key=abs, ascending=False)
                
                intercept_df = pd.DataFrame({'Feature': ['(Intercept)'], 'Weight': [intercept]})
                final_df = pd.concat([intercept_df, coef_df], ignore_index=True).set_index('Feature')
                
                st.dataframe(final_df.style.format({'Weight': '{:,.4f}'}), use_container_width=True) # Added container width
                
                top_n = min(10, non_zero)
                if top_n > 0:
                    top_coefs = coef_df.head(top_n)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_coefs['Weight']]
                    ax.barh(top_coefs['Feature'], top_coefs['Weight'], color=colors)
                    ax.set_xlabel('Coefficient Value', fontweight='bold')
                    ax.set_title(f'Top {top_n} Feature Weights (by magnitude)', fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig, use_container_width=True) # Added container width
                else:
                    st.info("All feature weights are zero. Try decreasing Alpha (λ).")
            
            with pred_tab:
                st.markdown("### Make a Prediction")
                input_values = {}
                cols = st.columns(3)
                
                for i, feature in enumerate(model_info['features']):
                    col_idx = i % 3
                    with cols[col_idx]:
                        # --- OPTIMIZATION ---
                        # Get feature data from the retrieved X_data
                        feature_data = X_data[feature] 
                        min_val, max_val, mean_val = (
                            float(feature_data.min()), 
                            float(feature_data.max()), 
                            float(feature_data.mean())
                        )
                        step_val = get_step(feature_data) 
                        decimals = max(0, -int(np.floor(np.log10(step_val)))) if step_val > 0 else 0
                        format_str = f"%.{decimals}f"

                        input_values[feature] = st.number_input(
                            feature,
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=step_val,
                            format=format_str,
                            key=f"input_{model_key}_{feature}"
                        )
                
                if st.button("🔮 Predict", key=f"predict_{model_key}", type="primary"):
                    input_df = pd.DataFrame([input_values])[model_info['features']]
                    input_scaled = model_info['scaler'].transform(input_df)
                    prediction = model_info['model'].predict(input_scaled)[0]
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2>🎯 Predicted Value</h2>
                        <h1>{prediction:.4f}</h1>
                        <p>Target: {y_data.name}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # --- OPTIMIZATION ---
                    # Get mean from retrieved y_data
                    y_mean = y_data.mean() 
                    diff_percent = ((prediction - y_mean) / y_mean) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Dataset Mean", f"{y_mean:.4f}")
                    with col2: st.metric("Prediction", f"{prediction:.4f}")
                    with col3: st.metric("Difference", f"{diff_percent:+.2f}%")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎯 Regularization Model Explorer | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
