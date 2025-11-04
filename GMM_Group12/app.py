import streamlit as st
from dataset_loader import load_dataset
from algorithms import run_algorithm
from visualization import plot_clusters, plot_actual_classes
from utils import compute_silhouette_score
from data_preprocessing import preprocess_data
from theory import get_algorithm_theory

st.set_page_config(page_title="Clustering Algorithm Comparison", layout="wide")

# -------------------
# Title
# -------------------
st.title("Clustering Algorithm Comparison")
st.write("### Compare Different Clustering Algorithms on Built-in or Custom Datasets")

# -------------------
# Sidebar Options
# -------------------
st.sidebar.header("Options")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ["Iris", "Wine", "Upload your own CSV"]
)

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["K-Means", "Gaussian Mixture Model", "DBSCAN", "Hierarchical Clustering", "Spectral Clustering"]
)

# -------------------
# Load Dataset
# -------------------
X, y_labels = load_dataset(dataset_name)

if X is not None:
    st.write("#### Dataset Preview")
    st.dataframe(X.head())
else:
    st.stop()

# -------------------
# Algorithm Theory Section
# -------------------
st.write("### 📘 Algorithm Theory")
theory_text = get_algorithm_theory(algorithm)
st.markdown(theory_text, unsafe_allow_html=True)

# -------------------
# Algorithm Parameters
# -------------------
n_clusters = None
eps = None
min_samples = None

if algorithm in ["K-Means", "Gaussian Mixture Model", "Hierarchical Clustering", "Spectral Clustering"]:
    n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)

if algorithm == "DBSCAN":
    eps = st.sidebar.slider("DBSCAN: eps (neighborhood size)", 0.1, 5.0, 0.5, 0.1)
    min_samples = st.sidebar.slider("DBSCAN: min_samples", 1, 20, 5)

# -------------------
# Run Clustering
# -------------------
labels = run_algorithm(algorithm, X, n_clusters, eps, min_samples)

# -------------------
# Evaluate
# -------------------
sil_score = compute_silhouette_score(X, labels)
st.write(f"**Silhouette Score:** {sil_score}")

# -------------------
# Visualizations (smaller)
# -------------------
st.write(f"### {algorithm} Clustering Result (PCA 2D Projection)")
plot_clusters(X, labels, algorithm, dataset_name, fig_size=(5, 4))

if y_labels is not None:
    st.write(f"### Actual Classification of {dataset_name}")
    plot_actual_classes(X, y_labels, dataset_name, fig_size=(5, 4))
