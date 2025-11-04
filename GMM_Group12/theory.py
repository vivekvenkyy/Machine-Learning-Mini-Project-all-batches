def get_algorithm_theory(algorithm):
    """Return theoretical explanation for each clustering algorithm."""
    if algorithm == "K-Means":
        return """
        **K-Means Clustering**  
        - K-Means partitions the dataset into *k* clusters by minimizing the variance within each cluster.  
        - It iteratively assigns points to the nearest centroid and updates centroids until convergence.  

        **Objective Function:**  
        \[
        J = \sum_{i=1}^{k} \sum_{x_j \in C_i} ||x_j - \mu_i||^2
        \]

        **Key Points:**  
        - Works well for spherical clusters.  
        - Sensitive to outliers and initialization.  
        """

    elif algorithm == "Gaussian Mixture Model":
        return """
        **Gaussian Mixture Model (GMM)**  
        - GMM assumes that the data is generated from a mixture of several Gaussian distributions.  
        - Each cluster is represented by a probability distribution.  

        **Equation:**  
        \[
        p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
        \]

        **Key Points:**  
        - Uses *Expectation-Maximization (EM)* algorithm.  
        - Handles overlapping clusters and soft assignments.
        """

    elif algorithm == "DBSCAN":
        return """
        **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**  
        - Groups together closely packed points and marks points in low-density regions as noise.  

        **Core Idea:**  
        - A point is a *core point* if it has at least *min_samples* neighbors within distance *eps*.  

        **Key Points:**  
        - Can find arbitrarily shaped clusters.  
        - Automatically detects noise and outliers.  
        - No need to specify number of clusters.
        """

    elif algorithm == "Hierarchical Clustering":
        return """
        **Hierarchical (Agglomerative) Clustering**  
        - Builds a hierarchy of clusters using a bottom-up approach.  
        - Starts with each data point as its own cluster and merges the closest pairs.  

        **Distance Metric Example:**  
        \[
        d(A, B) = \max_{i \in A, j \in B} d(i, j)
        \]  
        (Complete Linkage Example)

        **Key Points:**  
        - Produces a dendrogram showing merge steps.  
        - No need to predefine number of clusters.  
        """

    elif algorithm == "Spectral Clustering":
        return """
        **Spectral Clustering**  
        - Uses eigenvalues of a similarity matrix to perform dimensionality reduction before clustering.  
        - Captures non-linear relationships between points.  

        **Steps:**  
        1. Build similarity graph (adjacency matrix).  
        2. Compute Laplacian matrix \( L = D - W \).  
        3. Compute eigenvectors of L.  
        4. Apply K-Means on eigenvector representation.  

        **Key Points:**  
        - Works well for non-convex clusters.  
        - Requires careful choice of similarity function.
        """

    else:
        return "No theory available for the selected algorithm."
