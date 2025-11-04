import streamlit as st
import sys
import subprocess
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           confusion_matrix, silhouette_score,
                           calinski_harabasz_score, davies_bouldin_score)
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
import joblib
from datetime import datetime

# Page config
st.set_page_config(
    page_title="T-SNE Visualization and Analysis",
    layout="wide"
)

# Functions from the original code
def load_dataset(choice_or_path):
    """Load a dataset from built-in sklearn datasets or from a CSV file."""
    if isinstance(choice_or_path, str):  # For built-in datasets
        if choice_or_path == 'iris':
            data = load_iris()
            df_raw = pd.DataFrame(data.data, columns=data.feature_names)
            df_raw['target'] = data.target
            X = data.data
            y = data.target
        elif choice_or_path == 'wine':
            data = load_wine()
            df_raw = pd.DataFrame(data.data, columns=data.feature_names)
            df_raw['target'] = data.target
            X = data.data
            y = data.target
        elif choice_or_path == 'breast_cancer':
            data = load_breast_cancer()
            df_raw = pd.DataFrame(data.data, columns=data.feature_names)
            df_raw['target'] = data.target
            X = data.data
            y = data.target
    else:  # For uploaded files
        try:
            # Try different pandas read_csv parameters
            try:
                df_raw = pd.read_csv(choice_or_path)
            except:
                # Try with different encoding if default fails
                df_raw = pd.read_csv(choice_or_path, encoding='latin1')
            
            if df_raw.empty:
                raise ValueError("The uploaded file is empty")
                
            # Display DataFrame info for debugging
            st.write("DataFrame Info:")
            st.write(f"Shape: {df_raw.shape}")
            st.write("Columns:", df_raw.columns.tolist())
            
            # Let user select the target column
            target_column = st.selectbox(
                "Select target column",
                df_raw.columns.tolist(),
                index=len(df_raw.columns)-1
            )
            
            # Split features and target
            y = df_raw[target_column]
            X = df_raw.drop(columns=[target_column])
            
            # Check if target is numerical and has too many unique values
            if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 10:
                st.info(f"Target column '{target_column}' has {len(y.unique())} unique values. Converting to categories...")
                
                # Let user choose number of bins
                n_bins = st.slider(
                    "Number of categories to create",
                    min_value=3,
                    max_value=10,
                    value=5,
                    help="Choose how many categories to group the values into"
                )
                
                try:
                    # Create bins and labels
                    y_numeric = pd.to_numeric(y, errors='coerce')
                    bins = pd.qcut(y_numeric, q=n_bins, labels=[f'Group {i+1}' for i in range(n_bins)], duplicates='drop')
                    
                    # Show distribution of new categories
                    st.write("Distribution of categorized values:")
                    dist = bins.value_counts().sort_index()
                    st.write(dist)
                    
                    # Update y with binned values
                    y = bins
                    
                    # Create a mapping dictionary for reference
                    unique_vals = sorted(y.unique())
                    mapping = {}
                    for label in unique_vals:
                        if pd.notna(label):  # Check if label is not NaN
                            values_in_bin = y_numeric[bins == label]
                            if not values_in_bin.empty:
                                mapping[str(label)] = f"Range: {values_in_bin.min():.1f} - {values_in_bin.max():.1f}"
                    
                    st.write("\nCategory Ranges:")
                    for cat, range_val in mapping.items():
                        st.write(f"{cat}: {range_val}")
                except Exception as e:
                    st.warning(f"Could not bin values automatically: {str(e)}")
                    st.write("Using original values for target column.")
            
            # Ensure y is properly encoded for categorical values
            if not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = le.fit_transform(y)
                
                # Show mapping
                st.write("\nCategory Encoding:")
                for i, label in enumerate(le.classes_):
                    st.write(f"{label} → {i}")
            
            if X.empty:
                raise ValueError("No feature columns found after removing target column")
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.write("Please ensure your CSV file:")
            st.write("- Is not empty")
            st.write("- Has proper column headers")
            st.write("- Contains both feature and target columns")
            st.write("- Is properly formatted (try opening it in Excel to verify)")
            raise Exception("Failed to load CSV file")
    
    return X, y, df_raw

def preprocess_data(X, strategy="auto"):
    """Preprocess the input data with automatic detection of feature types."""
    if isinstance(X, pd.DataFrame):
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
    else:
        numeric_features = list(range(X.shape[1]))
        categorical_features = []
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))  # Changed from sparse to sparse_output
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    
    info_dict = {
        'n_numeric_features': len(numeric_features),
        'n_categorical_features': len(categorical_features),
        'total_features_after': X_processed.shape[1]
    }
    
    return X_processed, preprocessor, info_dict

def reduce_dimensions(X, method, **kwargs):
    """Reduce dimensionality using specified method."""
    start_time = time.perf_counter()
    metadata = {}
    
    if method == 'TSNE':
        perplexity = min(kwargs.get('perplexity', 30), (X.shape[0] - 1) / 3)
        n_components = kwargs.get('n_components', 2)
        
        if kwargs.get('pca_prestep', True) and X.shape[1] > 50:
            pca = PCA(n_components=50)
            X = pca.fit_transform(X)
            metadata['pca_variance_ratio'] = pca.explained_variance_ratio_.sum()
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=kwargs.get('n_iter', 1000),
            learning_rate=kwargs.get('learning_rate', 'auto'),
            random_state=42
        )
        X_reduced = tsne.fit_transform(X)
        metadata['kl_divergence'] = tsne.kl_divergence_
        
    elif method == 'PCA':
        pca = PCA(n_components=kwargs.get('n_components', 2))
        X_reduced = pca.fit_transform(X)
        metadata['explained_variance_ratio'] = pca.explained_variance_ratio_
        
    elif method in ['UMAP', 'PCA->UMAP']:
        if method == 'PCA->UMAP' and X.shape[1] > 50:
            pca = PCA(n_components=50)
            X = pca.fit_transform(X)
            metadata['pca_variance_ratio'] = pca.explained_variance_ratio_.sum()
            
        reducer = umap.UMAP(
            n_components=kwargs.get('n_components', 2),
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=kwargs.get('min_dist', 0.1),
            random_state=42
        )
        X_reduced = reducer.fit_transform(X)
    
    timing_seconds = time.perf_counter() - start_time
    return X_reduced, timing_seconds, metadata

def train_and_evaluate(X, y, models_list):
    """Train and evaluate selected models."""
    # Check class distribution
    class_counts = pd.Series(y).value_counts()
    min_samples_per_class = 2  # Minimum required samples per class
    
    # Check if we have enough samples per class
    if class_counts.min() < min_samples_per_class:
        st.warning(f"""
        Insufficient samples in some classes for reliable model evaluation.
        Minimum samples per class: {class_counts.min()}
        Class distribution:
        {class_counts.to_string()}
        
        Consider:
        1. Using a different dataset
        2. Removing classes with too few samples
        3. Collecting more data for underrepresented classes
        """)
        return pd.DataFrame(), {}, {}

    # Proceed with training if we have enough samples
    try:
        # Stratified split if possible
        if len(np.unique(y)) > 1 and all(class_counts >= min_samples_per_class):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        models_dict = {
            'svm': SVC(probability=True, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'lr': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        results = []
        for model_name in models_list:
            if model_name not in models_dict:
                continue
                
            model = models_dict[model_name]
            train_start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - train_start
            
            pred_start = time.perf_counter()
            y_pred = model.predict(X_test)
            pred_time = time.perf_counter() - pred_start
            
            # Handle binary and multiclass cases
            if len(np.unique(y)) > 2:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='macro'
                )
            else:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='binary'
                )
            
            results.append({
                'model': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'train_time': train_time,
                'pred_time': pred_time
            })
            
            # Save model
            os.makedirs('results/models', exist_ok=True)
            joblib.dump(model, f'results/models/{model_name}_model.joblib')
        
        metrics_df = pd.DataFrame(results)
        splits_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        return metrics_df, models_dict, splits_dict
        
    except Exception as e:
        st.error(f"""
        Error during model training: {str(e)}
        
        Dataset statistics:
        - Total samples: {len(y)}
        - Number of classes: {len(np.unique(y))}
        - Class distribution:
        {class_counts.to_string()}
        """)
        return pd.DataFrame(), {}, {}
    
def compute_embedding_quality(X_reduced, y):
    """Compute quality metrics for the embedding."""
    metrics = {}
    if len(np.unique(y)) >= 2:
        try:
            metrics['silhouette'] = silhouette_score(X_reduced, y)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_reduced, y)
            metrics['davies_bouldin'] = davies_bouldin_score(X_reduced, y)
        except Exception as e:
            metrics['error'] = str(e)
    else:
        metrics['error'] = 'Need at least 2 classes for clustering metrics'
    return metrics

def main():
    st.title("T-SNE Visualization and Analysis")
    st.markdown("""
    This application performs dimensionality reduction and classification using T-SNE
    and compares it with other methods like PCA and UMAP.
    """)

    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Dataset selection
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset",
        ["Iris", "Wine", "Breast Cancer", "Upload CSV"],
        help="Choose a dataset to analyze"
    )

    # File upload
    uploaded_file = None
    if dataset_choice == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is None:
            st.warning("Please upload a CSV file")
            return

    # Preprocessing options
    st.sidebar.subheader("Preprocessing")
    do_impute = st.sidebar.checkbox("Impute missing values", value=True)
    do_scale = st.sidebar.checkbox("Scale features", value=True)
    do_encode = st.sidebar.checkbox("Encode categorical variables", value=True)

    # Reduction method and parameters
    st.sidebar.subheader("Dimensionality Reduction")
    method = st.sidebar.selectbox(
        "Method",
        ["TSNE", "PCA", "UMAP", "PCA->UMAP"]
    )

    # Method-specific parameters
    params = {}
    if method == "TSNE":
        perplexity = st.sidebar.slider(
            "Perplexity",
            min_value=5,
            max_value=50,
            value=30
        )
        n_iter = st.sidebar.slider(
            "Iterations",
            min_value=250,
            max_value=2000,
            value=1000
        )
        learning_rate = st.sidebar.number_input(
            "Learning rate",
            min_value=10.0,
            max_value=1000.0,
            value=200.0
        )
        pca_prestep = st.sidebar.checkbox("Use PCA pre-step", value=True)
        params = {
            'perplexity': perplexity,
            'n_iter': n_iter,
            'learning_rate': learning_rate,
            'pca_prestep': pca_prestep
        }
    elif method in ["UMAP", "PCA->UMAP"]:
        n_neighbors = st.sidebar.slider(
            "n_neighbors",
            min_value=2,
            max_value=100,
            value=15
        )
        min_dist = st.sidebar.slider(
            "min_dist",
            min_value=0.0,
            max_value=0.99,
            value=0.1
        )
        params = {
            'n_neighbors': n_neighbors,
            'min_dist': min_dist
        }
    else:  # PCA
        n_components = st.sidebar.slider(
            "Components",
            min_value=2,
            max_value=10,
            value=2
        )
        params = {'n_components': n_components}

    # Model selection
    st.sidebar.subheader("Models")
    models_to_train = []
    if st.sidebar.checkbox("Support Vector Machine", value=True):
        models_to_train.append('svm')
    if st.sidebar.checkbox("Random Forest", value=True):
        models_to_train.append('rf')
    if st.sidebar.checkbox("Logistic Regression", value=True):
        models_to_train.append('lr')

    # Main content area
    col1, col2 = st.columns([2, 1])

    try:
        # Load data based on selection
        if dataset_choice == "Upload CSV":
            X, y, df = load_dataset(uploaded_file)
        else:
            X, y, df = load_dataset(dataset_choice.lower())

        with col1:
            st.subheader("Dataset Overview")
            st.write(f"Shape: {X.shape}")
            st.dataframe(df.head())

            # Show class distribution
            if len(np.unique(y)) > 1:
                st.subheader("Class Distribution")
                class_counts = pd.Series(y).value_counts()
                fig, ax = plt.subplots()
                class_counts.plot(kind='bar')
                plt.title("Class Distribution")
                plt.xlabel("Class")
                plt.ylabel("Count")
                st.pyplot(fig)
                plt.close()

                # Display class distribution details
                st.write("Class Distribution Details:")
                st.write(class_counts)

    except Exception as e:
        st.error(f"An error occurred while loading the data: {str(e)}")
        return

    # Process button
    if st.button("Run Analysis"):
        try:
            with st.spinner("Processing data..."):
                # Preprocess
                X_processed, preprocessor, preprocess_info = preprocess_data(X)
                
                with col2:
                    st.subheader("Preprocessing Summary")
                    for key, value in preprocess_info.items():
                        st.write(f"{key}: {value}")

                # Reduce dimensions
                X_reduced, timing, metadata = reduce_dimensions(X_processed, method, **params)
                st.write(f"Reduction completed in {timing:.2f} seconds")

                # Quality metrics
                quality_metrics = compute_embedding_quality(X_reduced, y)
                st.subheader("Embedding Quality Metrics")
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    for metric, value in quality_metrics.items():
                        if metric != 'error':
                            st.metric(metric, f"{value:.3f}")

                # Visualizations
                st.subheader("Visualization")
                fig = px.scatter(
                    x=X_reduced[:, 0],
                    y=X_reduced[:, 1],
                    color=y.astype(str),
                    title=f"{method} Embedding",
                    labels={'x': 'Component 1', 'y': 'Component 2'}
                )
                fig.update_layout(
                    width=800,
                    height=600,
                    title_x=0.5
                )
                st.plotly_chart(fig, use_container_width=True)

                # Train and evaluate models if selected
                if models_to_train:
                    st.subheader("Model Performance")
                    metrics_df, models_dict, splits = train_and_evaluate(
                        X_reduced, y, models_to_train
                    )
                    
                    # Only proceed if we got valid results
                    if not metrics_df.empty:
                        # Display metrics
                        st.dataframe(metrics_df)

                        # Confusion matrices
                        if splits:  # Check if we have valid splits
                            st.subheader("Confusion Matrices")
                            cm_cols = st.columns(len(models_to_train))
                            for idx, (model_name, model) in enumerate(models_dict.items()):
                                with cm_cols[idx]:
                                    y_pred = model.predict(splits['X_test'])
                                    cm = confusion_matrix(splits['y_test'], y_pred)
                                    fig, ax = plt.subplots()
                                    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                                    plt.title(f"{model_name.upper()}")
                                    st.pyplot(fig)
                                    plt.close()

                        # Save results if we have valid metrics
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        save_path = f'results/{timestamp}'
                        os.makedirs(save_path, exist_ok=True)
                        
                        # Save metrics summary
                        metrics_df['embedding_method'] = method
                        metrics_df['timestamp'] = timestamp
                        metrics_df.to_csv(f'{save_path}/metrics_summary.csv', index=False)
                        
                        # Create download button
                        st.download_button(
                            "Download Results Summary",
                            metrics_df.to_csv(index=False),
                            f"metrics_summary_{timestamp}.csv",
                            "text/csv",
                            key='download-csv'
                        )

                        # Display additional analysis
                        st.subheader("Analysis Summary")
                        st.write("""
                        #### Key Findings:
                        """)
                        best_model = metrics_df.loc[metrics_df['f1'].idxmax()]
                        st.write(f"- Best performing model: {best_model['model'].upper()} (F1-score: {best_model['f1']:.3f})")
                        st.write(f"- Total processing time: {timing:.2f} seconds")
                        
                        if 'explained_variance_ratio' in metadata:
                            st.write(f"- Explained variance ratio: {metadata['explained_variance_ratio']:.3f}")
                        if 'kl_divergence' in metadata:
                            st.write(f"- KL divergence: {metadata['kl_divergence']:.3f}")
                    else:
                        st.warning("Model evaluation was skipped due to insufficient data in some classes.")

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.write("Error details:", str(e))

if __name__ == "__main__":
    main()

