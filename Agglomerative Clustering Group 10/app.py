"""
Streamlit Agglomerative Clustering Dashboard
-------------------------------------------
Run locally:
  1) pip install -r requirements.txt  (see list below)
  2) streamlit run app.py

Minimal requirements (pin loosely for stability):
  streamlit>=1.36
  scikit-learn>=1.3
  scipy>=1.10
  pandas>=2.0
  numpy>=1.24
  matplotlib>=3.8
  plotly>=5.20

If you can't use requirements.txt, install manually:
  pip install streamlit scikit-learn scipy pandas numpy matplotlib plotly

References (shown in-app as well):
  - Scikit-learn AgglomerativeClustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
  - Clustering overview (sklearn): https://scikit-learn.org/stable/modules/clustering.html
  - Dendrogram basics (SciPy): https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
  - Silhouette score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
  - Davies–Bouldin: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html
  - Calinski–Harabasz: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
"""

import io
import math
import textwrap
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import plotly.express as px

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="Agglomerative Clustering Lab",
    layout="wide",
)

# Constants for upload restrictions
MAX_FILE_MB = 5
MAX_ROWS = 10000
MAX_COLS = 50

# Sampling caps for heavy visuals
DENDRO_MAX_SAMPLES = 2000
SPECTRAL_MAX_SAMPLES = 800  # spectral clustering can be heavy

# ---------------------------
# Utility Functions
# ---------------------------

def _human_readable_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    size = float(n)
    for u in units:
        if size < 1024:
            return f"{size:.1f} {u}"
        size /= 1024
    return f"{size:.1f} PB"


def validate_dataset(uploaded) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Validate uploaded CSV and return (df, errors)."""
    errors = []
    if uploaded is None:
        return None, []

    # size check
    size_bytes = getattr(uploaded, "size", None)
    if size_bytes is not None and size_bytes > MAX_FILE_MB * 1024 * 1024:
        errors.append(
            f"File too large: {_human_readable_bytes(size_bytes)} (limit {MAX_FILE_MB} MB)."
        )
        return None, errors

    # load CSV
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        errors.append(f"Failed to read CSV: {e}")
        return None, errors

    # basic shape checks
    if df.shape[0] > MAX_ROWS:
        errors.append(f"Too many rows: {df.shape[0]} (limit {MAX_ROWS}).")
    if df.shape[1] > MAX_COLS:
        errors.append(f"Too many columns: {df.shape[1]} (limit {MAX_COLS}).")

    # "Supervised" heuristic: disallow clear label columns
    label_like_names = {"target","label","class","y","response","outcome"}
    found_label_cols = [c for c in df.columns if str(c).strip().lower() in label_like_names]

    # Also consider categorical small-cardinality columns as likely labels
    n = len(df)
    small_cardinality_cols = []
    for c in df.columns:
        nunique = df[c].nunique(dropna=True)
        if nunique <= max(20, int(0.02*n)):
            # avoid flagging pure numeric IDs with huge range
            if not pd.api.types.is_float_dtype(df[c]) and not pd.api.types.is_integer_dtype(df[c]):
                small_cardinality_cols.append(c)

    likely_supervised = found_label_cols or small_cardinality_cols

    if found_label_cols:
        errors.append(
            "Dataset appears to contain label/target columns: " + ", ".join(found_label_cols)
        )
    if small_cardinality_cols:
        errors.append(
            "Dataset seems to include categorical columns with very few unique values (possible labels): "
            + ", ".join(map(str, small_cardinality_cols[:5]))
            + (" ..." if len(small_cardinality_cols) > 5 else "")
        )
        errors.append("Please upload purely unsupervised feature data (no ground-truth labels).")

    return (None if errors else df), errors


def describe_restrictions():
    st.markdown(
        """
        **Upload restrictions**
        - Only CSV files (no labels/targets). We reject common label names (e.g., `target`, `label`, `class`, `y`).
        - Avoid supervised data: low-cardinality categorical columns are flagged.
        - Size ≤ **5 MB**; ≤ **10,000 rows**; ≤ **50 columns**.
        - Mixed dtypes are okay — we'll auto-select numeric features. Consider one-hot encoding offline if needed.
        """
    )


def pick_numeric_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    """Select numeric columns, scale, return (df_num, X_scaled, used_cols, dropped_cols)."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dropped = [c for c in df.columns if c not in num_cols]
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found. Please upload numeric features.")

    df_num = df[num_cols].copy()

    # Impute simple (median) for safety
    for c in num_cols:
        if df_num[c].isna().any():
            df_num[c] = df_num[c].fillna(df_num[c].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num.values)
    return df_num, X_scaled, num_cols, dropped


def pca_project(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)


def cluster_and_scores(labels: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    # Some metrics undefined for 1 cluster or too small samples
    metrics = {}
    n_clusters = len(np.unique(labels))
    if n_clusters <= 1 or X.shape[0] < 5:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    try:
        metrics["silhouette"] = float(silhouette_score(X, labels))
    except Exception:
        metrics["silhouette"] = np.nan
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception:
        metrics["davies_bouldin"] = np.nan
    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception:
        metrics["calinski_harabasz"] = np.nan
    return metrics


def run_agglomerative(X: np.ndarray, n_clusters: int, linkage_method: str, metric: str) -> np.ndarray:
    # sklearn API changed "affinity"->"metric"; try new first, fallback to old
    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method, metric=metric)
    except TypeError:
        # Older sklearn
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method, affinity=metric)
    labels = model.fit_predict(X)
    return labels


def make_dendrogram(X: np.ndarray, method: str, metric: str):
    # Downsample for dendrogram if needed
    X_plot = X
    if X.shape[0] > DENDRO_MAX_SAMPLES:
        idx = np.random.RandomState(42).choice(X.shape[0], DENDRO_MAX_SAMPLES, replace=False)
        X_plot = X[idx]
    # SciPy linkage requires condensed distance for some metrics; let linkage handle pairs
    Z = linkage(X_plot, method=method, metric=metric)
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(Z, no_labels=True, color_threshold=None, ax=ax)
    ax.set_title("Dendrogram (sampled)")
    st.pyplot(fig, clear_figure=True)


def generate_synthetic(name: str, n_samples: int, noise: float, random_state: int = 42) -> pd.DataFrame:
    from sklearn.datasets import make_blobs, make_moons, make_circles

    if name == "blobs":
        X, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=1.2, random_state=random_state)
    elif name == "moons":
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == "circles":
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    else:
        X, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=random_state)
    df = pd.DataFrame(X, columns=["x1","x2"])
    return df


# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Data Source")
choice = st.sidebar.radio(
    "Choose an option:",
    ("Upload your dataset (CSV)", "Try our dataset"),
)

if choice == "Upload your dataset (CSV)":
    uploaded = st.sidebar.file_uploader("Upload CSV (unsupervised features only)", type=["csv"], accept_multiple_files=False)
    describe_restrictions()
    df, errors = validate_dataset(uploaded)
    if errors:
        with st.sidebar.expander("Upload issues", expanded=True):
            for e in errors:
                st.error(e)
else:
    with st.sidebar.expander("Synthetic dataset options", expanded=True):
        synth_type = st.selectbox("Shape", ["blobs","moons","circles"], index=0)
        synth_n = st.slider("Samples", min_value=200, max_value=5000, value=800, step=100)
        synth_noise = st.slider("Noise (moons/circles)", 0.0, 0.3, 0.05, 0.01)
    df = generate_synthetic(synth_type, synth_n, synth_noise)

# ---------------------------
# Preprocess
# ---------------------------
if df is not None:
    try:
        df_num, X_scaled, used_cols, dropped_cols = pick_numeric_matrix(df)
        with st.sidebar.expander("Feature selection", expanded=False):
            st.caption("Using numeric columns only:")
            st.code(", ".join(map(str, used_cols)) or "(none)")
            if dropped_cols:
                st.caption("Dropped non-numeric:")
                st.code(", ".join(map(str, dropped_cols)))
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        df_num, X_scaled = None, None

# ---------------------------
# Agglomerative Controls
# ---------------------------
st.sidebar.header("Agglomerative Clustering")
linkage_method = st.sidebar.selectbox("Linkage", ["ward","average","complete","single"], index=0,
    help="'ward' requires Euclidean distance; others allow different metrics.")

metric_options = ["euclidean","l1","l2","manhattan","cosine"]
if linkage_method == "ward":
    metric = "euclidean"
    st.sidebar.caption("Metric fixed to Euclidean for Ward linkage.")
else:
    metric = st.sidebar.selectbox("Distance metric", metric_options, index=0)

n_clusters = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=15, value=4)

# ---------------------------
# Layout Tabs
# ---------------------------
st.title("Agglomerative Clustering Lab")

if df is None:
    st.info("Provide data via the sidebar to begin.")
    st.stop()

TabIntro, TabAgglo, TabCompare, TabData, TabRefs = st.tabs([
    "Overview",
    "Agglomerative Results",
    "Compare Algorithms",
    "Data Preview",
    "About & References",
])

with TabIntro:
    st.markdown(
        """
        This dashboard lets you run **Agglomerative (Hierarchical) Clustering** on your own data (with restrictions)
        or on built-in synthetic datasets. You can:
        - Inspect a **2D projection** (PCA) with colored cluster assignments.
        - View a **dendrogram** (sampled) to understand the hierarchy.
        - Compare against **KMeans, DBSCAN, Spectral,** and **Gaussian Mixture** using internal metrics.

        ⚠️ **No supervised labels.** We block likely label columns to keep this an unsupervised workflow.
        """
    )

with TabAgglo:
    st.subheader("Run Agglomerative Clustering")
    if X_scaled is None:
        st.warning("No usable numeric features.")
    else:
        labels = run_agglomerative(X_scaled, n_clusters=n_clusters, linkage_method=linkage_method, metric=metric)
        metrics = cluster_and_scores(labels, X_scaled)

        # PCA projection
        proj2d = pca_project(X_scaled, 2)
        fig_scatter = px.scatter(
            x=proj2d[:,0], y=proj2d[:,1], color=labels.astype(str),
            labels={"x":"PC1","y":"PC2","color":"cluster"},
            title="PCA Projection (colored by Agglomerative labels)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Metrics
        st.markdown("**Internal validation metrics**")
        mtab = pd.DataFrame([
            {"metric":"silhouette (higher=better)", "value": metrics.get("silhouette")},
            {"metric":"davies_bouldin (lower=better)", "value": metrics.get("davies_bouldin")},
            {"metric":"calinski_harabasz (higher=better)", "value": metrics.get("calinski_harabasz")},
        ])
        st.dataframe(mtab, use_container_width=True)

        # Dendrogram
        with st.expander("Show dendrogram (sampled)"):
            try:
                make_dendrogram(X_scaled, method=linkage_method, metric=metric if linkage_method != "ward" else "euclidean")
                st.caption("For readability we sample up to %d points." % DENDRO_MAX_SAMPLES)
            except Exception as e:
                st.error(f"Dendrogram failed: {e}")

with TabCompare:
    st.subheader("Compare Clustering Algorithms")
    if X_scaled is None:
        st.warning("No usable numeric features.")
    else:
        # ---------------------------
        # Helpers
        # ---------------------------
        def _safe_scores(labels, X):
            try:
                return cluster_and_scores(labels, X)
            except Exception:
                return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}

        def _valid_for_silhouette(labels):
            labs = np.asarray(labels)
            uniq = np.unique(labs)
            if len(uniq) < 2:
                return False
            if (len(uniq) == 2) and (-1 in uniq):
                return False
            return True

        def _get_objective_value(scores, metric_key):
            v = scores.get(metric_key, np.nan)
            return v

        def _is_better(a, b, metric_key):
            # a and b are floats; decide maximize/minimize depending on metric
            if np.isnan(a): 
                return False
            if np.isnan(b): 
                return True
            if metric_key == "davies_bouldin":
                return a < b   # lower better
            # silhouette or calinski_harabasz
            return a > b       # higher better

        # ---------------------------
        # Existing controls (unchanged)
        # ---------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            k_k = st.number_input("KMeans: k", min_value=2, max_value=15, value=min(6, n_clusters))
        with col2:
            db_eps = st.number_input("DBSCAN: eps", min_value=0.01, max_value=10.0, value=0.5, step=0.05)
        with col3:
            db_min = st.number_input("DBSCAN: min_samples", min_value=3, max_value=50, value=5, step=1)

        spectral_on = st.checkbox("Enable Spectral Clustering (may be slow on large n)", value=(X_scaled.shape[0] <= SPECTRAL_MAX_SAMPLES))
        gmm_on = st.checkbox("Enable Gaussian Mixture (soft clustering)", value=True)

        # NEW: global best toggle + metric
        colx1, colx2 = st.columns([1,1])
        with colx1:
            best_mode = st.checkbox("Give best answer for my dataset", value=False)
        with colx2:
            metric_choice_label = st.selectbox(
                "Optimization metric",
                ["silhouette (higher=better)", "davies_bouldin (lower=better)", "calinski_harabasz (higher=better)"],
                index=0
            )
        metric_map = {
            "silhouette (higher=better)": "silhouette",
            "davies_bouldin (lower=better)": "davies_bouldin",
            "calinski_harabasz (higher=better)": "calinski_harabasz",
        }
        metric_key = metric_map[metric_choice_label]

        # Precompute PCA for plotting
        proj2d = pca_project(X_scaled, 2)

        results = []
        labels_manual = {}   # algorithm -> labels for plotting (manual)
        params_manual = {}   # algorithm -> manual params (for titles)

        # ---------------------------
        # Agglomerative (manual)
        # ---------------------------
        try:
            lab_ag = run_agglomerative(X_scaled, n_clusters=n_clusters, linkage_method=linkage_method, metric=metric)
            res_ag = _safe_scores(lab_ag, X_scaled)
            results.append({"algorithm":"Agglomerative (manual)", **res_ag, "n_clusters": int(len(np.unique(lab_ag)))})
            labels_manual["Agglomerative"] = lab_ag
            params_manual["Agglomerative"] = {"linkage": linkage_method, "metric": metric, "k": n_clusters}
        except Exception as e:
            st.error(f"Agglomerative error: {e}")

        # ---------------------------
        # KMeans (manual)
        # ---------------------------
        lab_km_manual = None
        try:
            km_manual = KMeans(n_clusters=int(k_k), n_init="auto", random_state=42)
            lab_km_manual = km_manual.fit_predict(X_scaled)
            res_km_manual = _safe_scores(lab_km_manual, X_scaled)
            results.append({"algorithm":"KMeans (manual)", **res_km_manual, "n_clusters": int(len(np.unique(lab_km_manual)))})
            labels_manual["KMeans"] = lab_km_manual
            params_manual["KMeans"] = {"k": int(k_k)}
        except Exception as e:
            st.error(f"KMeans error: {e}")

        # ---------------------------
        # DBSCAN (manual)
        # ---------------------------
        try:
            db_manual = DBSCAN(eps=float(db_eps), min_samples=int(db_min))
            lab_db_manual = db_manual.fit_predict(X_scaled)
            res_db_manual = _safe_scores(lab_db_manual, X_scaled)
            results.append({"algorithm":"DBSCAN (manual)", **res_db_manual, "n_clusters": int(len(np.unique(lab_db_manual)))})
            labels_manual["DBSCAN"] = lab_db_manual
            params_manual["DBSCAN"] = {"eps": float(db_eps), "min_samples": int(db_min)}
        except Exception as e:
            st.error(f"DBSCAN error: {e}")

        # ---------------------------
        # Spectral (manual) — only if enabled
        # ---------------------------
        if spectral_on:
            try:
                if X_scaled.shape[0] > SPECTRAL_MAX_SAMPLES:
                    st.info(f"Sampling to {SPECTRAL_MAX_SAMPLES} rows for Spectral to keep it responsive.")
                    rng = np.random.RandomState(42)
                    idx = rng.choice(X_scaled.shape[0], SPECTRAL_MAX_SAMPLES, replace=False)
                    X_spec = X_scaled[idx]
                    proj2d_spec_manual = proj2d[idx]
                else:
                    X_spec = X_scaled
                    proj2d_spec_manual = proj2d

                sp_manual = SpectralClustering(n_clusters=int(k_k), assign_labels="kmeans", random_state=42, affinity="nearest_neighbors")
                lab_sp_manual = sp_manual.fit_predict(X_spec)
                res_sp_manual = _safe_scores(lab_sp_manual, X_spec)
                results.append({"algorithm":"Spectral (manual)", **res_sp_manual, "n_clusters": int(len(np.unique(lab_sp_manual)))})
                labels_manual["Spectral"] = (lab_sp_manual, proj2d_spec_manual)  # sampled labels + projected coords
                params_manual["Spectral"] = {"k": int(k_k), "affinity": "nearest_neighbors"}
            except Exception as e:
                st.error(f"Spectral error: {e}")

        # ---------------------------
        # GMM (manual) — only if enabled
        # ---------------------------
        if gmm_on:
            try:
                gm_manual = GaussianMixture(n_components=int(k_k), random_state=42)
                lab_gm_manual = gm_manual.fit_predict(X_scaled)
                res_gm_manual = _safe_scores(lab_gm_manual, X_scaled)
                results.append({"algorithm":"GaussianMixture (manual)", **res_gm_manual, "n_clusters": int(len(np.unique(lab_gm_manual)))})
                labels_manual["GaussianMixture"] = lab_gm_manual
                params_manual["GaussianMixture"] = {"k": int(k_k)}
            except Exception as e:
                st.error(f"GaussianMixture error: {e}")

        # ---------------------------
        # Auto-tuned variants (comparison only; plots stay manual unless best_mode)
        # ---------------------------
        # Agglomerative (auto k around current)
        try:
            k_ag_candidates = list(range(2, min(15, max(3, int(n_clusters)+4))+1))
            best_ag = {"val": np.nan, "k": None, "scores": None}
            for kk in k_ag_candidates:
                lab = run_agglomerative(X_scaled, n_clusters=int(kk), linkage_method=linkage_method, metric=metric)
                sc = _safe_scores(lab, X_scaled)
                # if silhouette selected but invalid, skip
                if metric_key == "silhouette" and not _valid_for_silhouette(lab):
                    continue
                val = _get_objective_value(sc, metric_key)
                if _is_better(val, best_ag["val"], metric_key):
                    best_ag = {"val": val, "k": kk, "scores": sc, "labels": lab}
            if best_ag["k"] is not None:
                results.append({"algorithm":"Agglomerative (auto)", **best_ag["scores"], "n_clusters": int(len(np.unique(best_ag["labels"]))), "note": f"k={best_ag['k']}"})
        except Exception:
            pass

        # KMeans (auto k)
        try:
            k_candidates = list(range(2, min(15, max(3, int(k_k)+4))+1))
            best = {"val": np.nan, "k": None, "scores": None, "labels": None}
            for kk in k_candidates:
                km = KMeans(n_clusters=int(kk), n_init="auto", random_state=42)
                lab = km.fit_predict(X_scaled)
                sc = _safe_scores(lab, X_scaled)
                if metric_key == "silhouette" and np.isnan(sc.get("silhouette", np.nan)):
                    continue
                val = _get_objective_value(sc, metric_key)
                if _is_better(val, best["val"], metric_key):
                    best = {"val": val, "k": kk, "scores": sc, "labels": lab}
            if best["k"] is not None:
                results.append({"algorithm":"KMeans (auto)", **best["scores"], "n_clusters": int(len(np.unique(best["labels"]))), "note": f"k={best['k']}"})
        except Exception:
            pass

        # DBSCAN (auto around manual)
        try:
            eps_list = [max(0.01, float(db_eps)*f) for f in (0.5, 0.75, 1.0, 1.25, 1.5)]
            ms_base = int(db_min)
            ms_list = sorted(set([max(3, ms_base-2), ms_base, min(50, ms_base+2)]))
            best = {"val": np.nan, "params": None, "scores": None, "labels": None}
            for e_ in eps_list:
                for m_ in ms_list:
                    db = DBSCAN(eps=float(e_), min_samples=int(m_))
                    lab = db.fit_predict(X_scaled)
                    sc = _safe_scores(lab, X_scaled)
                    if metric_key == "silhouette" and not _valid_for_silhouette(lab):
                        continue
                    val = _get_objective_value(sc, metric_key)
                    if _is_better(val, best["val"], metric_key):
                        best = {"val": val, "params": (e_, m_), "scores": sc, "labels": lab}
            if best["params"] is not None:
                e_, m_ = best["params"]
                results.append({"algorithm":"DBSCAN (auto)", **best["scores"], "n_clusters": int(len(np.unique(best["labels"]))), "note": f"eps={e_:.3g}, min_samples={int(m_)}"})
        except Exception:
            pass

        # Spectral (auto k) — only if enabled
        if spectral_on:
            try:
                if X_scaled.shape[0] > SPECTRAL_MAX_SAMPLES:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(X_scaled.shape[0], SPECTRAL_MAX_SAMPLES, replace=False)
                    X_spec_auto = X_scaled[idx]
                else:
                    X_spec_auto = X_scaled
                k_candidates = list(range(2, min(15, max(3, int(k_k)+4))+1))
                best = {"val": np.nan, "k": None, "scores": None, "labels": None}
                for kk in k_candidates:
                    sp = SpectralClustering(n_clusters=int(kk), assign_labels="kmeans", random_state=42, affinity="nearest_neighbors")
                    lab = sp.fit_predict(X_spec_auto)
                    sc = _safe_scores(lab, X_spec_auto)
                    if metric_key == "silhouette" and np.isnan(sc.get("silhouette", np.nan)):
                        continue
                    val = _get_objective_value(sc, metric_key)
                    if _is_better(val, best["val"], metric_key):
                        best = {"val": val, "k": kk, "scores": sc, "labels": lab}
                if best["k"] is not None:
                    results.append({"algorithm":"Spectral (auto)", **best["scores"], "n_clusters": int(len(np.unique(best["labels"]))), "note": f"k={best['k']}"})
            except Exception:
                pass

        # GMM (auto k) — only if enabled
        if gmm_on:
            try:
                k_candidates = list(range(2, min(15, max(3, int(k_k)+4))+1))
                best = {"val": np.nan, "k": None, "scores": None, "labels": None}
                for kk in k_candidates:
                    gm = GaussianMixture(n_components=int(kk), random_state=42)
                    lab = gm.fit_predict(X_scaled)
                    sc = _safe_scores(lab, X_scaled)
                    if metric_key == "silhouette" and np.isnan(sc.get("silhouette", np.nan)):
                        continue
                    val = _get_objective_value(sc, metric_key)
                    if _is_better(val, best["val"], metric_key):
                        best = {"val": val, "k": kk, "scores": sc, "labels": lab}
                if best["k"] is not None:
                    results.append({"algorithm":"GaussianMixture (auto)", **best["scores"], "n_clusters": int(len(np.unique(best["labels"]))), "note": f"k={best['k']}"})
            except Exception:
                pass

        # ---------------------------
        # Results table + comparison chart
        # ---------------------------
        best_summary = None
        if results:
            dfres = pd.DataFrame(results)
            st.dataframe(dfres.set_index("algorithm"), use_container_width=True)

            # Comparison chart
            try:
                melt = dfres.melt(id_vars=["algorithm"], value_vars=["silhouette","davies_bouldin","calinski_harabasz"], var_name="metric", value_name="value")
                fig_bar = px.bar(melt, x="algorithm", y="value", color="metric", barmode="group", title="Algorithm Comparison (Internal Metrics)")
                st.plotly_chart(fig_bar, use_container_width=True)
            except Exception as e:
                st.info(f"Comparison graph unavailable: {e}")

            # Winner by selected metric across ALL rows (manual + auto)
            vals = []
            for i, r in dfres.iterrows():
                v = r.get(metric_key, np.nan)
                vals.append(v)
            if any([not np.isnan(v) for v in vals]):
                # choose best by selected metric
                if metric_key == "davies_bouldin":
                    idx = dfres[metric_key].astype(float).idxmin()
                else:
                    idx = dfres[metric_key].astype(float).idxmax()
                winner = dfres.loc[idx]
                best_summary = winner

        # ---------------------------
        # PCA views
        # - If best_mode: show a dedicated "Best Answer" section and plot best model.
        # - Regardless, show MANUAL plots below (Spectral/GMM still controlled by checkboxes).
        # ---------------------------
                # ---------------------------
        # Best Answer rendering (FIXED)
        # ---------------------------
        if best_mode and best_summary is not None:
            st.markdown("**Best Answer (by selected optimization metric)**")

            # Pull safe fields from the row (don't rely on .name)
            algo_name = str(best_summary.get("algorithm", "")).strip()
            v = float(best_summary.get(metric_key, np.nan))
            ncl = int(best_summary.get("n_clusters", np.nan)) if not np.isnan(best_summary.get("n_clusters", np.nan)) else None
            note_val = best_summary.get("note", None)
            note = str(note_val).strip() if (note_val is not None and not (isinstance(note_val, float) and np.isnan(note_val))) else ""

            # Header summary
            summary_bits = [f"{algo_name} selected", f"{metric_key}={v:.3f}"]
            if ncl is not None:
                summary_bits.append(f"clusters={ncl}")
            if note:
                summary_bits.append(note)
            st.success(" | ".join(summary_bits))

            # Helpers to parse k / eps,min_samples from "note" like "k=5" or "eps=0.3, min_samples=7"
            def _parse_k_from_note(note_str: str, default_k: int) -> int:
                try:
                    if "k=" in note_str:
                        return int(note_str.split("k=")[-1].split(",")[0].strip())
                except Exception:
                    pass
                return int(default_k)

            def _parse_dbscan_from_note(note_str: str, default_eps: float, default_min: int) -> Tuple[float, int]:
                try:
                    s = note_str.replace(" ", "")
                    parts = s.split(",")
                    eps_val = default_eps
                    min_val = default_min
                    for p in parts:
                        if p.startswith("eps="):
                            eps_val = float(p.split("=")[-1])
                        elif p.startswith("min_samples="):
                            min_val = int(p.split("=")[-1])
                    return float(eps_val), int(min_val)
                except Exception:
                    return float(default_eps), int(default_min)

            # Re-run the winning model for plotting (on full or sampled data as appropriate)
            best_plot_labels = None
            best_plot_coords = proj2d

            try:
                if algo_name.startswith("Agglomerative"):
                    # manual fallback k if no auto note
                    k_val = _parse_k_from_note(note, params_manual.get("Agglomerative", {}).get("k", n_clusters))
                    best_plot_labels = run_agglomerative(
                        X_scaled, n_clusters=int(k_val),
                        linkage_method=linkage_method, metric=metric
                    )

                elif algo_name.startswith("KMeans"):
                    k_val = _parse_k_from_note(note, params_manual.get("KMeans", {}).get("k", k_k))
                    best_plot_labels = KMeans(n_clusters=int(k_val), n_init="auto", random_state=42).fit_predict(X_scaled)

                elif algo_name.startswith("DBSCAN"):
                    eps_def = params_manual.get("DBSCAN", {}).get("eps", db_eps)
                    min_def = params_manual.get("DBSCAN", {}).get("min_samples", db_min)
                    eps_val, min_val = _parse_dbscan_from_note(note, eps_def, min_def)
                    best_plot_labels = DBSCAN(eps=float(eps_val), min_samples=int(min_val)).fit_predict(X_scaled)

                elif algo_name.startswith("Spectral"):
                    # Respect sampling guard
                    if X_scaled.shape[0] > SPECTRAL_MAX_SAMPLES:
                        rng = np.random.RandomState(42)
                        idx = rng.choice(X_scaled.shape[0], SPECTRAL_MAX_SAMPLES, replace=False)
                        X_spec = X_scaled[idx]
                        best_plot_coords = proj2d[idx]
                    else:
                        X_spec = X_scaled
                        best_plot_coords = proj2d

                    k_val = _parse_k_from_note(note, params_manual.get("Spectral", {}).get("k", k_k))
                    best_plot_labels = SpectralClustering(
                        n_clusters=int(k_val), assign_labels="kmeans",
                        random_state=42, affinity="nearest_neighbors"
                    ).fit_predict(X_spec)

                elif algo_name.startswith("GaussianMixture"):
                    k_val = _parse_k_from_note(note, params_manual.get("GaussianMixture", {}).get("k", k_k))
                    best_plot_labels = GaussianMixture(n_components=int(k_val), random_state=42).fit_predict(X_scaled)

            except Exception as e:
                st.info(f"Could not re-run best model for plotting: {e}")

            if best_plot_labels is not None:
                fig_best = px.scatter(
                    x=best_plot_coords[:,0], y=best_plot_coords[:,1],
                    color=np.asarray(best_plot_labels).astype(str),
                    labels={"x":"PC1","y":"PC2","color":"cluster"},
                    title=f"Best Model View — {algo_name}"
                )
                st.plotly_chart(fig_best, use_container_width=True)

        # Manual plots (always)
        st.markdown("**2D PCA views**")
        row1 = st.columns(2)
        with row1[0]:
            if "Agglomerative" in labels_manual:
                fig1 = px.scatter(x=proj2d[:,0], y=proj2d[:,1],
                                  color=labels_manual["Agglomerative"].astype(str),
                                  labels={"x":"PC1","y":"PC2","color":"cluster"},
                                  title="Agglomerative")
                st.plotly_chart(fig1, use_container_width=True)
        with row1[1]:
            if "KMeans" in labels_manual:
                fig2 = px.scatter(x=proj2d[:,0], y=proj2d[:,1],
                                  color=labels_manual["KMeans"].astype(str),
                                  labels={"x":"PC1","y":"PC2","color":"cluster"},
                                  title=f"KMeans (k={params_manual['KMeans']['k']})")
                st.plotly_chart(fig2, use_container_width=True)

        row2 = st.columns(3)
        with row2[0]:
            if "DBSCAN" in labels_manual:
                pm = params_manual["DBSCAN"]
                fig_db = px.scatter(x=proj2d[:,0], y=proj2d[:,1],
                                    color=labels_manual["DBSCAN"].astype(str),
                                    labels={"x":"PC1","y":"PC2","color":"cluster"},
                                    title=f"DBSCAN (eps={pm['eps']:.2g}, min={pm['min_samples']})")
                st.plotly_chart(fig_db, use_container_width=True)
        with row2[1]:
            if spectral_on and ("Spectral" in labels_manual):
                lab_sp, proj2d_spec = labels_manual["Spectral"]
                fig_sp = px.scatter(x=proj2d_spec[:,0], y=proj2d_spec[:,1],
                                    color=lab_sp.astype(str),
                                    labels={"x":"PC1","y":"PC2","color":"cluster"},
                                    title=f"Spectral (k={params_manual['Spectral']['k']})")
                st.plotly_chart(fig_sp, use_container_width=True)
        with row2[2]:
            if gmm_on and ("GaussianMixture" in labels_manual):
                fig_gm = px.scatter(x=proj2d[:,0], y=proj2d[:,1],
                                    color=labels_manual["GaussianMixture"].astype(str),
                                    labels={"x":"PC1","y":"PC2","color":"cluster"},
                                    title=f"GMM (k={params_manual['GaussianMixture']['k']})")
                st.plotly_chart(fig_gm, use_container_width=True)

with TabData:
    st.subheader("Dataset Preview")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.dataframe(df.head(200), use_container_width=True)
    st.caption("Showing first 200 rows for responsiveness.")

with TabRefs:
    st.subheader("Algorithm cheat-sheet")
    st.markdown(
        """
        **Agglomerative (Hierarchical)** — Bottom-up merging of clusters using a linkage criterion.
        Works with a variety of distance metrics; dendrogram reveals the merge tree. Sensitive to scaling.

        **KMeans** — Partitions data into k spherical-ish clusters by minimizing within-cluster variance. Fast, needs k.

        **DBSCAN** — Density-based; discovers arbitrary shapes and marks noise. Two key params: `eps`, `min_samples`.

        **Spectral Clustering** — Graph-based; useful for complex manifolds. Can be slow on large datasets.

        **Gaussian Mixture (GMM)** — Probabilistic soft assignments assuming Gaussian components; expect elliptical clusters.
        """
    )

    st.markdown("**Further reading**")
    st.write("- Scikit-learn AgglomerativeClustering docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html")
    st.write("- Clustering overview (sklearn): https://scikit-learn.org/stable/modules/clustering.html")
    st.write("- SciPy dendrogram: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html")
    st.write("- Silhouette score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html")
    st.write("- Davies–Bouldin: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html")
    st.write("- Calinski–Harabasz: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html")

    st.caption("All links open official project documentation.")
