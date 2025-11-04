import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# App Title
# ----------------------------
st.title("Explainable AI (XAI) for Regression")
st.markdown("""
Train regression models and explore explainability insights using SHAP and correlation heatmaps.
Select a dataset, pick a model, and see predictions and explanations below.
""")

# ----------------------------
# Dataset Selection
# ----------------------------
st.sidebar.header("Dataset Selection")
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    ["California Housing (Default)", "Upload your own CSV"]
)

if dataset_choice == "California Housing (Default)":
    from sklearn.datasets import fetch_california_housing

    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    st.sidebar.success("Using California Housing dataset")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="latin1")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        target_col = st.sidebar.selectbox("Select Target Column (y)", df.columns)
        y = df[target_col]
        X = df.drop(columns=[target_col])
        st.sidebar.success("Custom dataset loaded")
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

# ----------------------------
# Model Selection
# ----------------------------
st.sidebar.header("Choose Regression Model")
model_choice = st.sidebar.selectbox(
    "Select a Model",
    ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
)

# ----------------------------
# Train & Explain
# ----------------------------
if st.sidebar.button("Train & Explain"):

    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Splitting data...")
    progress_bar.progress(10)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    status_text.text("Training model...")
    progress_bar.progress(30)

    # Train model
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Decision Tree Regressor":
        model = DecisionTreeRegressor(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    status_text.text("Evaluating model...")
    progress_bar.progress(50)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # ----------------------------
    # Model Evaluation Metrics
    # ----------------------------
    st.subheader("Model Evaluation")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MSE", f"{mse:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("MAE", f"{mae:.2f}")
    col4.metric("R² Score", f"{r2:.3f}")

    # Performance interpretation
    with st.expander("What do these metrics mean?"):
        st.markdown(f"""
        - **MSE (Mean Squared Error)**: {mse:.2f} - Average squared difference between predictions and actual values. Lower is better.
        - **RMSE (Root Mean Squared Error)**: {rmse:.2f} - Square root of MSE, in same units as target variable. Easier to interpret.
        - **MAE (Mean Absolute Error)**: {mae:.2f} - Average absolute difference. More robust to outliers than MSE.
        - **R² Score**: {r2:.3f} - Proportion of variance explained by the model. Range: 0 to 1 (higher is better).
          - R² > 0.7: Good model
          - R² 0.4-0.7: Moderate model
          - R² < 0.4: Poor model
        """)

    # Regression Plot
    st.subheader("Regression Plot: Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=y_test, y=y_pred, color="dodgerblue", label="Predicted", alpha=0.6)
    sns.lineplot(x=y_test, y=y_test, color="orange", label="Ideal Fit", linewidth=2)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{model_choice} Regression Plot")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # ----------------------------
    # SHAP Explainability
    # ----------------------------
    st.subheader("SHAP Explainability Analysis")

    with st.expander("What is SHAP?"):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** is a unified approach to explain the output of machine learning models.

        - **Global Importance**: Shows which features are most important across all predictions
        - **Local Explanation**: Shows how each feature contributed to a specific prediction
        - **Positive SHAP values**: Feature pushes prediction higher
        - **Negative SHAP values**: Feature pushes prediction lower
        """)

    # Initialize variables for use later
    top_3_features = []
    shap_success = False

    try:
        # Use background sampling for speed - OPTIMIZED
        if model_choice in ["Decision Tree Regressor", "Random Forest Regressor"]:
            # TreeExplainer is fast, no background needed
            explainer = shap.TreeExplainer(model)
            # Only explain a subset for speed
            shap_values = explainer.shap_values(X_test[:min(500, len(X_test))])
            X_test_shap = X_test.iloc[:min(500, len(X_test))]
        else:  # Linear Regression
            # KernelExplainer is slow, use small background
            background = shap.sample(X_train, min(50, len(X_train)))
            explainer = shap.KernelExplainer(model.predict, background)
            # Only explain a small subset for speed
            shap_values = explainer.shap_values(X_test[:min(100, len(X_test))])
            X_test_shap = X_test.iloc[:min(100, len(X_test))]

        # SHAP summary plot (global importance)
        st.write("**Global Feature Importance (SHAP Summary Plot):**")
        with st.spinner("Generating SHAP summary plot..."):
            fig1 = plt.figure()
            shap.summary_plot(shap_values, X_test_shap, show=False)
            st.pyplot(fig1)
            plt.close(fig1)

        st.markdown("""
        **Interpretation**: Each point represents a prediction. Colors show feature values (red=high, blue=low).
        Position on x-axis shows impact on prediction.
        """)

        # SHAP bar plot for mean importance
        st.write("**Mean Absolute SHAP Values (Average Impact):**")
        fig_bar = plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X_test_shap, plot_type="bar", show=False)
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        # Local explanation for first test sample
        st.write("**Local Explanation (First Sample Waterfall Plot):**")

        sample_idx = 0  # Always use first sample

        fig2 = plt.figure()
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value if isinstance(explainer.expected_value, (int, float)) else
            explainer.expected_value[0],
            data=X_test_shap.iloc[sample_idx].values,
            feature_names=X_test_shap.columns.tolist()
        ), show=False)
        st.pyplot(fig2)
        plt.close(fig2)

        # Show actual prediction details
        actual_val = y_test.iloc[sample_idx]
        pred_val = y_pred[sample_idx]
        st.markdown(f"""
        **Sample #{sample_idx} Details:**
        - Actual Value: `{actual_val:.2f}`
        - Predicted Value: `{pred_val:.2f}`
        - Prediction Error: `{abs(actual_val - pred_val):.2f}`
        - Base Value (average prediction): `{explainer.expected_value if isinstance(explainer.expected_value, (int, float)) else explainer.expected_value[0]:.2f}`
        """)

        # SHAP dependence plots for top features
        st.write("**SHAP Dependence Plots (Top 3 Features):**")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features = np.argsort(mean_abs_shap)[-3:][::-1]

        for i, feat_idx in enumerate(top_features):
            feat_name = X_test_shap.columns[feat_idx]
            with st.spinner(f"Generating dependence plot for {feat_name}..."):
                fig_dep, ax_dep = plt.subplots(figsize=(8, 4))
                shap.dependence_plot(feat_idx, shap_values, X_test_shap, show=False, ax=ax_dep)
                st.pyplot(fig_dep)
                plt.close(fig_dep)
                st.caption(f"Shows how {feat_name} values affect predictions and interactions with other features")

        # Feature interaction analysis
        st.write("**Key Insights from SHAP Analysis:**")
        top_3_features = [X_test_shap.columns[i] for i in top_features]
        st.markdown(f"""
        **Most Important Features**: {', '.join(top_3_features)}

        These features have the highest average impact on predictions across all samples.
        The model relies heavily on these features to make accurate predictions.
        """)

        shap_success = True

    except Exception as e:
        st.error(f"SHAP explainability failed: {e}")
        # Fallback: use model-based feature importance for top features
        if model_choice in ["Decision Tree Regressor", "Random Forest Regressor"]:
            importance = model.feature_importances_
        else:
            importance = np.abs(model.coef_)

        top_features_idx = np.argsort(importance)[-3:][::-1]
        top_3_features = [X.columns[i] for i in top_features_idx]

    # ----------------------------
    # Correlation Heatmap
    # ----------------------------
    st.subheader("Feature Correlation Analysis")

    with st.expander("Understanding Correlation"):
        st.markdown("""
        Correlation measures linear relationships between features:
        - **+1**: Perfect positive correlation
        - **0**: No linear correlation
        - **-1**: Perfect negative correlation
        - **|r| > 0.7**: Strong correlation (may indicate multicollinearity)
        """)

    corr = X.corr()
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True, ax=ax3)
    ax3.set_title("Feature Correlation Heatmap")
    st.pyplot(fig3)
    plt.close(fig3)

    # Identify highly correlated features
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.7:
                high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    if high_corr_pairs:
        st.warning("**High Correlations Detected:**")
        for feat1, feat2, corr_val in high_corr_pairs:
            st.write(f"- {feat1} <-> {feat2}: {corr_val:.3f}")
        st.info("High correlation between features may indicate redundancy. Consider feature selection.")

    # ----------------------------
    # Feature Importance
    # ----------------------------
    st.subheader("Model-Based Feature Importance")

    if model_choice in ["Decision Tree Regressor", "Random Forest Regressor"]:
        importance = model.feature_importances_
        importance_type = "Gini Importance" if model_choice == "Decision Tree Regressor" else "Mean Decrease in Impurity"
    else:  # Linear Regression
        importance = np.abs(model.coef_)
        importance_type = "Absolute Coefficient Values"

    fi_df = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", ax=ax4)
    ax4.set_title(f"Feature Importance ({importance_type})")
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown(f"""
    **Top 3 Most Important Features:**
    1. {fi_df.iloc[0]['Feature']}: {fi_df.iloc[0]['Importance']:.4f}
    2. {fi_df.iloc[1]['Feature']}: {fi_df.iloc[1]['Importance']:.4f}
    3. {fi_df.iloc[2]['Feature']}: {fi_df.iloc[2]['Importance']:.4f}
    """)

    # ----------------------------
    # Residual Analysis
    # ----------------------------
    st.subheader("Residual Analysis")

    with st.expander("What are Residuals?"):
        st.markdown("""
        Residuals are the differences between actual and predicted values.
        - **Good model**: Residuals randomly distributed around zero
        - **Patterns in residuals**: Suggest model isn't capturing all relationships
        """)

    residuals = y_test - y_pred

    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 4))

    # Residual plot
    ax5a.scatter(y_pred, residuals, alpha=0.5, color='purple')
    ax5a.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax5a.set_xlabel('Predicted Values')
    ax5a.set_ylabel('Residuals')
    ax5a.set_title('Residual Plot')
    ax5a.grid(True, alpha=0.3)

    # Histogram of residuals
    ax5b.hist(residuals, bins=30, color='teal', alpha=0.7, edgecolor='black')
    ax5b.set_xlabel('Residuals')
    ax5b.set_ylabel('Frequency')
    ax5b.set_title('Distribution of Residuals')
    ax5b.axvline(x=0, color='red', linestyle='--', linewidth=2)

    st.pyplot(fig5)
    plt.close(fig5)

    # Residual statistics
    st.markdown(f"""
    **Residual Statistics:**
    - Mean: {residuals.mean():.4f} (should be close to 0)
    - Std Dev: {residuals.std():.4f}
    - Min: {residuals.min():.4f}
    - Max: {residuals.max():.4f}
    """)

    # ----------------------------
    # Executive Summary
    # ----------------------------
    st.subheader("Executive Summary")

    # Determine model quality
    if r2 > 0.7:
        quality = "**Good**"
        recommendation = "The model performs well and can be used for predictions with confidence."
    elif r2 > 0.4:
        quality = "**Moderate**"
        recommendation = "The model shows moderate performance. Consider feature engineering or trying different algorithms."
    else:
        quality = "**Poor**"
        recommendation = "The model needs significant improvement. Review feature selection and consider more complex models."

    # Use top_3_features if available, otherwise use fallback
    if len(top_3_features) == 0:
        top_3_features = list(fi_df.head(3)['Feature'])

    st.markdown(f"""
    ### Model Performance: {quality}

    **Selected Model:** {model_choice}

    **Performance Metrics:**
    - R² Score: {r2:.3f}
    - RMSE: {rmse:.2f}
    - MAE: {mae:.2f}

    **Key Findings:**
    - The model explains **{r2 * 100:.1f}%** of the variance in the target variable
    - Average prediction error: {mae:.2f} units
    - Most influential features: {', '.join(top_3_features[:3])}

    **Recommendation:** {recommendation}

    **Model Interpretability:**
    - {'SHAP values reveal how each feature contributes to individual predictions' if shap_success else 'Model-based feature importance shows key predictive features'}
    - {'Strong correlations detected between features - consider feature selection' if high_corr_pairs else 'No concerning multicollinearity detected'}
    - Residual analysis shows {'random distribution (good)' if abs(residuals.mean()) < 0.1 else 'some systematic bias (needs attention)'}
    """)

