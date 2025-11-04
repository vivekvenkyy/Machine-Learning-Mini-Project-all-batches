import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    log_loss, precision_score, recall_score, f1_score
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from xgboost import XGBClassifier  # ✅ XGBoost import

# ---------------------------
# Streamlit Configuration
# ---------------------------
st.set_page_config(page_title="Enhanced ML Classifier App", layout="wide", initial_sidebar_state="expanded")
st.title("🤖 Gaussian Process + XGBoost Classifier Dashboard")

# ---------------------------
# Sidebar Dataset Selection
# ---------------------------
st.sidebar.header("📂 Dataset Options")
dataset_choice = st.sidebar.radio("Choose a dataset source:", ("Upload Custom CSV", "Use Sample Dataset"))

df = None
dataset_name = None

if dataset_choice == "Use Sample Dataset":
    sample_file = st.sidebar.selectbox("Select a sample dataset", ("moons.csv", "iris.csv"))
    df = pd.read_csv(sample_file)
    dataset_name = sample_file
    st.write(f"### Loaded Sample Dataset: {sample_file}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        dataset_name = "custom"
        st.write("### Uploaded Dataset Preview")
    else:
        st.info("📥 Upload a CSV file or select a sample dataset to begin.")
        st.stop()

st.dataframe(df.head(), width="stretch")

# ---------------------------
# Preprocessing
# ---------------------------
st.sidebar.header("🧹 Data Preprocessing")

if dataset_name == "moons.csv":
    target_col = "label"
elif dataset_name == "iris.csv":
    target_col = "Species"
else:
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)

if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in dataset.")
    st.stop()

if st.sidebar.checkbox("Handle Missing Values", value=True):
    df = df.fillna(df.mean(numeric_only=True))
    st.sidebar.success("Missing values filled with column mean.")

X = df.drop(columns=[target_col])
y = df[target_col]

if y.dtype == 'object':
    y = pd.factorize(y)[0]

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
st.sidebar.success("Features scaled automatically for better convergence.")

split_ratio = st.sidebar.slider("Train-Test Split (test size %)", 10, 50, 30)
test_size = split_ratio / 100.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# ---------------------------
# Gaussian Process Specific Parameters
# ---------------------------
st.sidebar.header("🎛️ Gaussian Process Tuning")

kernel_choice = st.sidebar.selectbox("Select Kernel Type", ("RBF", "Matern", "RationalQuadratic"))
length_scale = st.sidebar.slider("Kernel Length Scale", 0.1, 10.0, 1.0)
restarts = st.sidebar.slider("Optimizer Restarts", 0, 10, 3)
max_iter_predict = st.sidebar.slider("Max Iter Predict", 50, 1000, 100)

if kernel_choice == "RBF":
    kernel = 1.0 * RBF(length_scale=length_scale)
elif kernel_choice == "Matern":
    kernel = 1.0 * Matern(length_scale=length_scale, nu=1.5)
else:
    kernel = 1.0 * RationalQuadratic(length_scale=length_scale, alpha=0.5)

# ---------------------------
# Classifiers (including XGBoost)
# ---------------------------
classifiers = {
    "Gaussian Process": GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=restarts, max_iter_predict=max_iter_predict),
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "SVM": SVC(C=1.0, probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "XGBoost": XGBClassifier(  # ✅ Added XGBoost here
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
}

# ---------------------------
# Model Training & Evaluation
# ---------------------------
results = {}
warnings.filterwarnings("ignore", category=ConvergenceWarning)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

for name, clf in classifiers.items():
    try:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    except Exception as e:
        results[name] = np.nan
        st.warning(f"{name} failed: {e}")

results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
results_df["Accuracy"] = pd.to_numeric(results_df["Accuracy"], errors="coerce")
results_df = results_df.sort_values(by="Accuracy", ascending=False)

st.write("## 📊 Accuracy Comparison")
st.dataframe(results_df, width="stretch")

fig, ax = plt.subplots(figsize=(4, 2))
results_df["Accuracy"].plot(kind="bar", ax=ax, color="skyblue")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)

# ---------------------------
# Gaussian Process Detailed Metrics
# ---------------------------
st.write("---")
st.subheader("🔍 Gaussian Process Classifier Performance")

gpc = classifiers["Gaussian Process"]
try:
    y_pred = gpc.predict(X_test)
    y_proba = gpc.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    ll = log_loss(y_test, y_proba)

    st.write(f"**Accuracy:** {acc:.3f}")
    st.write(f"**Precision:** {prec:.3f}")
    st.write(f"**Recall:** {rec:.3f}")
    st.write(f"**F1-score:** {f1:.3f}")
    st.write(f"**Log Loss:** {ll:.3f}")

    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    if st.checkbox("Run 5-Fold CV for GPC"):
        cv_scores = cross_val_score(gpc, X, y, cv=5)
        st.write(f"**Average CV Accuracy:** {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

except Exception as e:
    st.error(f"Gaussian Process evaluation failed: {e}")

# ---------------------------
# 🌟 XGBoost Feature Importance Visualization
# ---------------------------
if "XGBoost" in classifiers:
    st.write("---")
    st.subheader("🌟 XGBoost Feature Importance")
    try:
        xgb_model = classifiers["XGBoost"]
        if hasattr(xgb_model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": xgb_model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
            ax.set_title("XGBoost Feature Importance")
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Feature importance plot failed: {e}")

# ---------------------------
# 🧭 Class Separation Visualization
# ---------------------------
st.write("## 🎨 Data Visualization: Class Separation")

visual_method = st.radio("Select Visualization Method", ("PCA", "t-SNE"), horizontal=True)

try:
    if X.shape[1] > 2:
        if visual_method == "PCA":
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//5))
        X_vis = reducer.fit_transform(X)
    else:
        X_vis = X.values

    vis_df = pd.DataFrame(X_vis, columns=["Dim1", "Dim2"])
    vis_df["label"] = y

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=vis_df, x="Dim1", y="Dim2", hue="label", palette="viridis", s=50, alpha=0.8, ax=ax)
    ax.set_title(f"{visual_method} Projection of Dataset")
    st.pyplot(fig)

except Exception as e:
    st.warning(f"Visualization failed: {e}")

# ---------------------------
# 🧩 Decision Boundary Visualization
# ---------------------------
st.write("## 🧩 Decision Boundary Visualization (2D)")

if X.shape[1] > 2:
    X_plot = X_vis
else:
    X_plot = X.values

try:
    gpc_2d = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=restarts, max_iter_predict=max_iter_predict)
    gpc_2d.fit(X_plot, y)

    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = gpc_2d.predict(grid)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=y, palette="viridis", s=50, edgecolor="k", ax=ax)
    ax.set_title(f"{visual_method} Projection with GPC Decision Boundary")
    st.pyplot(fig)

except Exception as e:
    st.warning(f"Decision boundary visualization failed: {e}")

# ---------------------------
# Save Model
# ---------------------------
if st.button("💾 Save Gaussian Process Model"):
    joblib.dump(gpc, "GaussianProcessClassifier.pkl")
    st.success("✅ Gaussian Process Classifier saved as `GaussianProcessClassifier.pkl`")
