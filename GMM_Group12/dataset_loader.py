import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from data_preprocessing import preprocess_data

def load_dataset(dataset_name):
    """
    Loads and preprocesses the selected dataset.
    Supports: Iris, Wine, and user-uploaded CSVs.
    Returns:
        X (DataFrame): Cleaned feature data
        y_labels (list or None): Labels (if available)
    """
    if dataset_name == "Iris":
        data = datasets.load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
        target_names = data.target_names
        y_labels = [target_names[i] for i in y]
        return X, y_labels

    elif dataset_name == "Wine":
        data = datasets.load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
        target_names = data.target_names
        y_labels = [target_names[i] for i in y]
        return X, y_labels

    elif dataset_name == "Upload your own CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            X = pd.read_csv(uploaded_file)
            st.write("#### Raw Uploaded Dataset")
            st.dataframe(X.head())

            X = preprocess_data(X)  # ✅ Clean automatically
            y_labels = None
            return X, y_labels
        else:
            st.warning("Please upload a CSV file to continue.")
            st.stop()

    else:
        st.error("Unknown dataset selected.")
        st.stop()
