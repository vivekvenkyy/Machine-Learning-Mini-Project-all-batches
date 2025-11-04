
# ----------- PART 1: Simple KNN Regression Demo -----------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Print dataset description
print(diabetes.DESCR)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the KNN regressor
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ideal fit')
plt.title('KNN Regression: Predicted vs Actual')
plt.xlabel('Actual Disease Progression')
plt.ylabel('Predicted Disease Progression')
plt.legend()
plt.show()

import gradio as gr
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, fetch_california_housing

# Load built-in datasets
def load_builtin_dataset(choice):
    if choice == "Diabetes Dataset":
        data = load_diabetes(as_frame=True)
        return data.frame, "target"
    elif choice == "California Housing":
        data = fetch_california_housing(as_frame=True)
        return data.frame, "MedHouseVal"
    else:
        return pd.DataFrame(), None

# Core comparison function with charts
def compare_regressors(dataset_choice, file, target_column, n_neighbors=5):
    # Load dataset
    if dataset_choice != "Upload your own CSV":
        df, target_column = load_builtin_dataset(dataset_choice)
    else:
        if file is None:
            return "⚠️ Please upload a CSV file.", None, None, None, None
        df = pd.read_csv(file.name)

    if target_column not in df.columns:
        return f"❌ Target column '{target_column}' not found! Columns:\n{list(df.columns)}", None, None, None, None

    # Split features and target
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models
    models = {
        "KNN Regressor": KNeighborsRegressor(n_neighbors=n_neighbors),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Support Vector Regressor (SVR)": SVR()
    }

    results = {}
    y_pred_knn = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "MSE": round(mean_squared_error(y_test, y_pred), 3),
            "R²": round(r2_score(y_test, y_pred), 3)
        }
        if name == "KNN Regressor":
            y_pred_knn = y_pred

    # Convert to DataFrame
    df_results = pd.DataFrame(results).T.sort_values("R²", ascending=False)

    # Identify best model
    best_model_name = df_results["R²"].idxmax()
    knn_r2 = results["KNN Regressor"]["R²"]
    summary_text = f"### Dataset: {dataset_choice}\n**Target:** {target_column}\n\n"
    summary_text += df_results.to_markdown()
    summary_text += f"\n\n✅ Best performing model: **{best_model_name}**"
    summary_text += f"\n🔹 KNN Regressor R²: {knn_r2}"

    # Create charts using Plotly
    fig_r2 = px.bar(df_results, x=df_results.index, y="R²", text="R²",
                    title="R² Scores by Model", color="R²")
    fig_mse = px.bar(df_results, x=df_results.index, y="MSE", text="MSE",
                     title="MSE by Model", color="MSE")

    # Scatter plot for KNN predictions vs actual
    fig_scatter = px.scatter(x=y_test, y=y_pred_knn, labels={"x":"Actual", "y":"Predicted"},
                             title="KNN Predicted vs Actual")
    fig_scatter.add_shape(type="line", x0=min(y_test), x1=max(y_test),
                          y0=min(y_test), y1=max(y_test), line=dict(color="red", dash="dash"))

    return summary_text, fig_r2, fig_mse, fig_scatter

# Auto-update target column
def update_target_box(dataset_choice):
    if dataset_choice == "Upload your own CSV":
        return gr.update(value="", visible=True, interactive=True, label="Target Column Name (y)")
    else:
        _, target_col = load_builtin_dataset(dataset_choice)
        return gr.update(value=target_col, visible=True, interactive=False, label="Auto Target Column")

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## ⚙️ Regression Model Comparison Dashboard")
    gr.Markdown("Upload or select a dataset and compare regression models (KNN, Random Forest, SVR, etc.)")

    dataset_choice = gr.Radio(
        ["Upload your own CSV", "Diabetes Dataset", "California Housing"],
        label="Choose Dataset Source",
        value="Diabetes Dataset"
    )
    file_input = gr.File(file_types=[".csv"], label="Upload CSV (if selected above)")
    target_input = gr.Textbox(label="Target Column", value="", visible=True)
    k_slider = gr.Slider(1, 20, value=5, step=1, label="K (for KNN)")

    output_table = gr.Markdown()
    output_r2 = gr.Plot()
    output_mse = gr.Plot()
    output_scatter = gr.Plot()

    dataset_choice.change(fn=update_target_box, inputs=dataset_choice, outputs=target_input)

    gr.Button("Compare Models 🚀").click(
        fn=compare_regressors,
        inputs=[dataset_choice, file_input, target_input, k_slider],
        outputs=[output_table, output_r2, output_mse, output_scatter]
    )

demo.launch()
