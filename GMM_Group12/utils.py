from sklearn.metrics import silhouette_score

def compute_silhouette_score(X, labels):
    if len(set(labels)) > 1 and -1 not in labels:
        return round(silhouette_score(X, labels), 3)
    else:
        return "Not available (only 1 cluster or noise detected)"
