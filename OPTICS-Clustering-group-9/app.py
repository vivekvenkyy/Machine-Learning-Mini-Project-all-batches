import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
from sklearn.datasets import load_wine

st.set_page_config(layout="wide")

def calculate_metrics(X_scaled, labels):
    """Calculate clustering metrics, handling cases with single cluster or noise."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    metrics = {
        'clusters': n_clusters,
        'noise_points': n_noise,
        'silhouette': np.nan,
        'davies_bouldin': np.nan,
        'calinski': np.nan
    }

    if n_clusters > 1 and n_clusters < len(X_scaled) and n_noise < len(X_scaled):
        valid_mask = labels != -1
        if valid_mask.sum() > 1:
            try:
                metrics['silhouette'] = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
                metrics['davies_bouldin'] = davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask])
                metrics['calinski'] = calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask])
            except Exception as e:
                st.warning(f"Could not calculate all metrics: {e}")
    return metrics

st.title("🔍 OPTICS Clustering Comparison Tool")

st.sidebar.header("📁 Data Loading")
dataset_option = st.sidebar.radio("Choose a dataset:", ("Upload CSV", "Inbuilt Wine Dataset"))

df = None
if dataset_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded: {len(df)} rows")
        except Exception as e:
            st.sidebar.error(f"Failed to load dataset: {e}")
elif dataset_option == "Inbuilt Wine Dataset":
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    st.sidebar.success(f"Loaded Inbuilt Wine Dataset: {len(df)} rows")

if df is not None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must have at least 2 numeric columns for analysis.")
    else:
        st.sidebar.header("🎯 Feature Selection")
        x_feature = st.sidebar.selectbox("X-Axis Feature:", options=numeric_cols, index=0)
        y_feature = st.sidebar.selectbox("Y-Axis Feature:", options=numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

        st.sidebar.header("⚙️ Algorithm Parameters")

        st.sidebar.subheader("OPTICS Parameters")
        optics_min_samples = st.sidebar.slider("min_samples", min_value=2, max_value=20, value=5)
        optics_xi = st.sidebar.slider("xi", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
        optics_min_cluster_size = st.sidebar.slider("min_cluster_size (fraction)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

        st.sidebar.subheader("K-Means & Hierarchical Parameters")
        kmeans_clusters = st.sidebar.slider("n_clusters", min_value=2, max_value=10, value=5)

        st.sidebar.subheader("DBSCAN Parameters")
        dbscan_eps = st.sidebar.slider("eps", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        dbscan_min_samples = st.sidebar.slider("min_samples (DBSCAN)", min_value=2, max_value=20, value=5)

        if st.sidebar.button("▶️ Run Comparison Analysis"):
            X = df[[x_feature, y_feature]].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            results = {}

            # 1. OPTICS
            optics = OPTICS(
                min_samples=optics_min_samples,
                xi=optics_xi,
                min_cluster_size=optics_min_cluster_size,
                metric='euclidean'
            )
            optics_labels = optics.fit_predict(X_scaled)
            results['OPTICS'] = {
                'model': optics,
                'labels': optics_labels,
                'name': 'OPTICS',
                'metrics': calculate_metrics(X_scaled, optics_labels)
            }

            # 2. K-Means
            kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            results['K-Means'] = {
                'model': kmeans,
                'labels': kmeans_labels,
                'name': 'K-Means',
                'metrics': calculate_metrics(X_scaled, kmeans_labels)
            }

            # 3. DBSCAN
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            dbscan_labels = dbscan.fit_predict(X_scaled)
            results['DBSCAN'] = {
                'model': dbscan,
                'labels': dbscan_labels,
                'name': 'DBSCAN',
                'metrics': calculate_metrics(X_scaled, dbscan_labels)
            }

            # 4. Hierarchical
            hierarchical = AgglomerativeClustering(n_clusters=kmeans_clusters)
            hierarchical_labels = hierarchical.fit_predict(X_scaled)
            results['Hierarchical'] = {
                'model': hierarchical,
                'labels': hierarchical_labels,
                'name': 'Hierarchical',
                'metrics': calculate_metrics(X_scaled, hierarchical_labels)
            }

            st.subheader("Clustering Comparison Results")

            st.write("### Dataset Preview")
            st.dataframe(df)

            st.subheader("Preprocessing")
            with st.expander("Show Processing Details"):
                st.write("#### Descriptive Statistics")
                st.dataframe(df.describe())

                st.write("#### Missing Values Count")
                st.dataframe(df.isnull().sum().rename("Missing Values"))

                st.write("#### Data Types")
                st.dataframe(df.dtypes.rename("Data Type"))

            tab1, tab2, tab3, tab4 = st.tabs(["Clustering Results", "Reachability Plots", "Performance Metrics", "Comparison"])

            with tab1:
                st.write("### OPTICS Clustering Scatter Plot")
                fig1, ax = plt.subplots(figsize=(7, 5))
                optics_result = results['OPTICS']
                scatter = ax.scatter(
                    X[:, 0],
                    X[:, 1],
                    c=optics_result['labels'],
                    cmap='viridis',
                    s=50,
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=0.5
                )
                ax.set_xlabel(x_feature, fontsize=10)
                ax.set_ylabel(y_feature, fontsize=10)
                ax.set_title(f"{optics_result['name']} Clustering", fontsize=12, weight='bold')
                ax.grid(True, alpha=0.3)
                fig1.colorbar(scatter, ax=ax, label='Cluster')
                st.pyplot(fig1)

            with tab2:
                st.write("### Reachability Plot (OPTICS) & Cluster Size Distribution")
                fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                # OPTICS reachability
                optics_model = results['OPTICS']['model']
                if hasattr(optics_model, 'reachability_') and hasattr(optics_model, 'ordering_'):
                    reachability = optics_model.reachability_[optics_model.ordering_]
                    ax1.bar(range(len(reachability)), reachability, color='steelblue', edgecolor='black')
                    ax1.set_xlabel('Order', fontsize=10)
                    ax1.set_ylabel('Reachability Distance', fontsize=10)
                    ax1.set_title('OPTICS Reachability Plot', fontsize=12, weight='bold')
                    ax1.grid(True, alpha=0.3, axis='y')
                else:
                    ax1.text(0.5, 0.5, "OPTICS reachability data not available", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
                    ax1.set_title('OPTICS Reachability Plot', fontsize=12, weight='bold')

                # Cluster distribution (Improved)
                cluster_data_for_plot = []
                for algo_name, result in results.items():
                    labels = result['labels']
                    for label in labels:
                        cluster_data_for_plot.append({
                            'Algorithm': algo_name,
                            'Cluster_Label': f"C{label}" if label != -1 else "Noise"
                        })
                
                if cluster_data_for_plot:
                    cluster_df = pd.DataFrame(cluster_data_for_plot)
                    # Use seaborn.countplot for better visualization of categorical distributions
                    sns.countplot(
                        data=cluster_df,
                        x='Algorithm',
                        hue='Cluster_Label',
                        ax=ax2,
                        palette='tab10', # A good categorical palette
                        edgecolor='black'
                    )
                    ax2.set_xlabel('Algorithm', fontsize=10)
                    ax2.set_ylabel('Number of Points', fontsize=10)
                    ax2.set_title('Cluster Size Distribution', fontsize=12, weight='bold')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.grid(True, alpha=0.3, axis='y')
                    ax2.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    ax2.text(0.5, 0.5, "No cluster data available for distribution plot", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
                    ax2.set_title('Cluster Size Distribution', fontsize=12, weight='bold')
                
                plt.tight_layout() # Adjust layout to prevent labels from overlapping
                st.pyplot(fig2)
                st.write("### OPTICS Reachability Plot Explanation")
                st.image("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.semanticscholar.org%2Fpaper%2FOPTICS%253A-ordering-points-to-identify-the-clustering-Ankerst-Breunig%2F80c983b2f36e3db461e35a5e8836d4b20b485d4f&psig=AOvVaw1F0tyU6WnHYAajuG8Mgj5j&ust=1761717706481000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCLCFxoKcxpADFQAAAAAdAAAAABAV", caption="OPTICS Reachability Plot Interpretation")

            with tab3:
                st.write("### Performance Metrics Comparison")
                metrics_data = {
                    'Algorithm': [],
                    'Silhouette': [],
                    'Davies-Bouldin': [],
                    'Calinski-Harabasz': []
                }

                for algo_name, result in results.items():
                    metrics_data['Algorithm'].append(algo_name)
                    metrics_data['Silhouette'].append(result['metrics']['silhouette'])
                    metrics_data['Davies-Bouldin'].append(result['metrics']['davies_bouldin'])
                    metrics_data['Calinski-Harabasz'].append(result['metrics']['calinski'])
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df.set_index('Algorithm'))

                fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
                
                # Silhouette Score
                ax = axes3[0]
                ax.bar(metrics_df['Algorithm'], metrics_df['Silhouette'], color='skyblue', edgecolor='black')
                ax.set_ylabel('Score', fontsize=10)
                ax.set_title('Silhouette Score\n(Higher is Better)', fontsize=11, weight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Davies-Bouldin Index
                ax = axes3[1]
                ax.bar(metrics_df['Algorithm'], metrics_df['Davies-Bouldin'], color='salmon', edgecolor='black')
                ax.set_ylabel('Score', fontsize=10)
                ax.set_title('Davies-Bouldin Index\n(Lower is Better)', fontsize=11, weight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Calinski-Harabasz Score
                ax = axes3[2]
                ax.bar(metrics_df['Algorithm'], metrics_df['Calinski-Harabasz'], color='lightgreen', edgecolor='black')
                ax.set_ylabel('Score', fontsize=10)
                ax.set_title('Calinski-Harabasz Score\n(Higher is Better)', fontsize=11, weight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig3)

            with tab4:
                st.write("### Clustering Scatter Plots Comparison")
                fig_comp, axes_comp = plt.subplots(2, 2, figsize=(14, 10))
                axes_comp = axes_comp.flatten()

                for idx, (algo_name, result) in enumerate(results.items()):
                    ax = axes_comp[idx]
                    scatter = ax.scatter(
                        X[:, 0],
                        X[:, 1],
                        c=result['labels'],
                        cmap='viridis',
                        s=50,
                        alpha=0.6,
                        edgecolors='black',
                        linewidths=0.5
                    )
                    ax.set_xlabel(x_feature, fontsize=10)
                    ax.set_ylabel(y_feature, fontsize=10)
                    ax.set_title(f"{algo_name} Clustering", fontsize=12, weight='bold')
                    ax.grid(True, alpha=0.3)
                    fig_comp.colorbar(scatter, ax=ax, label='Cluster')
                st.pyplot(fig_comp)

else:
    st.info("Please upload a CSV file to begin the clustering analysis.")
