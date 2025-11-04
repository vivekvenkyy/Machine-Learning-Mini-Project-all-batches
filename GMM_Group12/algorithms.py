from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture

def run_algorithm(algorithm, X, n_clusters=None, eps=None, min_samples=None):
    if algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        return model.fit_predict(X)

    elif algorithm == "Gaussian Mixture Model":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        return model.fit_predict(X)

    elif algorithm == "Hierarchical Clustering":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        return model.fit_predict(X)

    elif algorithm == "Spectral Clustering":
        model = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans', random_state=42)
        return model.fit_predict(X)

    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        return model.fit_predict(X)
