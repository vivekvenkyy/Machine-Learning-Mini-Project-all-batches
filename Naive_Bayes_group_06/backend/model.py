import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import LabelEncoder # <-- Import LabelEncoder
import os
from sklearn.svm import SVC  # <-- ADD THIS IMPORT
import chardet

# --- Setup ---
os.makedirs("results", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')


# --- Data Loading and Preprocessing ---
try:
    dataset_path = sys.argv[1]
    # data = pd.read_csv(dataset_path)
    # --- Auto-detect file encoding (fix for UnicodeDecodeError) ---
    with open(dataset_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    encoding = result['encoding'] or 'utf-8'
    print(f"Detected encoding: {encoding}")

    data = pd.read_csv(dataset_path, encoding=encoding)

    # --- (NEW) Smart Preprocessing Step ---
    
    # 1. Drop completely empty columns (like 'Unnamed: 32')
    data = data.dropna(axis=1, how='all')
    
    # 2. Drop any 'id' columns (case-insensitive)
    id_col = [col for col in data.columns if col.lower() == 'id']
    if id_col:
        data = data.drop(id_col[0], axis=1)
    
    # 3. (NEW) Find the target column
    # Best guess: Find the *only* text-based column.
    object_cols = data.select_dtypes(include='object').columns
    
    if len(object_cols) == 1:
        # If there's exactly one text column, assume it's the target.
        target_column = object_cols[0]
    else:
        # Otherwise, fall back to the original logic:
        # Assumes target is the last column.
        target_column = data.columns[-1]

    # 4. Identify Target (y) and Features (X)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # 5. Encode Target (y) if it's text
    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)

    # 6. Encode Features (X) that are text
    # (This handles any *other* text columns in X if logic fell back)
    le_x = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = le_x.fit_transform(X[col])

    # 7. Drop any rows with remaining missing values
    # This must be done *after* X and y are separated
    data_combined = X.join(pd.Series(y, name=target_column))
    data_combined = data_combined.dropna()
    
    X = data_combined.drop(target_column, axis=1)
    y = data_combined[target_column]
    # --- End of Preprocessing ---

except Exception as e:
    print(f"Error loading or processing data: {e}")
    sys.exit(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- (FIX) Check if the problem is binary or multiclass ---
n_classes = len(np.unique(y_test)) # Check the test set
is_binary = n_classes == 2

# --- Model Definitions ---
models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SVC": SVC(probability=True, random_state=42)  # <-- ADD THIS LINE
}

# --- Training and Evaluation ---
results_data = {"models": {}}
accuracies = {}

# --- (FIX) Plot for combined ROC Curve (only if binary) ---
if is_binary:
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Baseline') # Baseline

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- (FIX) Get probabilities only if binary ---
    y_prob = None
    if is_binary:
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            print(f"Warning: Could not get probabilities for {name}: {e}")
            y_prob = None # Ensure it's None if proba fails

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # --- Confusion Matrix Visualization (Unchanged) ---
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, annot_kws={"size": 14})
    ax_cm.set_title(f"{name} Confusion Matrix", fontsize=16)
    ax_cm.set_xlabel("Predicted", fontsize=12)
    ax_cm.set_ylabel("Actual", fontsize=12)
    cm_path = f"results/cm_{name.replace(' ', '_').lower()}.png"
    fig_cm.savefig(cm_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_cm)

    # --- (FIX) ROC Curve Data (only if binary and proba worked) ---
    if is_binary and y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    # --- Store Results (Unchanged) ---
    results_data["models"][name] = {
        "accuracy": acc,
        "classification_report": report,
        "plots": {
            "confusion_matrix": f"/{cm_path}"
        }
    }

# --- (FIX) Finalize and Save Combined ROC Curve Plot ---
results_data["comparison_plots"] = {}  # Always create the key

if is_binary:
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.legend(loc='lower right')
    roc_path = "results/roc_curves_comparison.png"
    fig_roc.savefig(roc_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_roc)
    # Only add the path if the plot was actually created
    results_data["comparison_plots"]["roc_curves"] = f"/{roc_path}" 

# --- Save Final JSON Output ---
results_data["accuracies"] = accuracies

with open("results/metrics.json", "w") as f:
    json.dump(results_data, f, indent=4)

print("Analysis complete. Metrics and plots saved successfully.")