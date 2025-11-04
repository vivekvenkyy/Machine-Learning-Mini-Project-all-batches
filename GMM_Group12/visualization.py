import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.decomposition import PCA

def plot_clusters(X, labels, algorithm, dataset_name, fig_size=(4, 3)):
    """Plot PCA-reduced clustering result."""
    pca = PCA(2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=fig_size)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=30, alpha=0.8)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", loc="best", fontsize=8)
    ax.add_artist(legend1)

    ax.set_xlabel("PCA 1", fontsize=9)
    ax.set_ylabel("PCA 2", fontsize=9)
    ax.set_title(f"{algorithm} on {dataset_name}", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)


def plot_actual_classes(X, y_labels, dataset_name, fig_size=(4, 3)):
    """Plot PCA-reduced actual labels for comparison."""
    pca = PCA(2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=fig_size)
    unique_labels = np.unique(y_labels)
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    for i, label in enumerate(unique_labels):
        ax.scatter(
            X_pca[np.array(y_labels) == label, 0],
            X_pca[np.array(y_labels) == label, 1],
            label=label.capitalize(),
            color=colors[i % len(colors)],
            s=30,
            alpha=0.8
        )

    ax.legend(title="Actual Classes", fontsize=8)
    ax.set_xlabel("PCA 1", fontsize=9)
    ax.set_ylabel("PCA 2", fontsize=9)
    ax.set_title(f"Actual Classification of {dataset_name}", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
