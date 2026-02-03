#!/usr/bin/env python
"""Basic example of using fastCPF for clustering."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from fastcpf import FastCPF


def generate_data(n_per_cluster=150, random_state=42):
    """Generate 3 Gaussian clusters (2 close, 1 far)."""
    np.random.seed(random_state)
    # Cluster 1 (Top right - isolated)
    c1 = np.random.randn(n_per_cluster, 2) * 0.3 + [1.5, 1.5]
    # Cluster 2 (Bottom left)
    c2 = np.random.randn(n_per_cluster, 2) * 0.3 + [-1.0, -0.8]
    # Cluster 3 (Bottom center - close to cluster 2, bridge structure)
    c3 = np.random.randn(n_per_cluster, 2) * 0.3 + [-0.1, -0.7]

    X = np.vstack([c1, c2, c3])
    y_true = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)
    return X, y_true


def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y_true = generate_data(n_per_cluster=150, random_state=42)
    X = StandardScaler().fit_transform(X).astype(np.float32)

    # Create and fit the model
    print("Fitting FastCPF model...")
    model = FastCPF(
        min_samples=10,
        rho=0.4,
        alpha=1.0,
        cutoff=1,
        knn_backend="kd",
    )
    model.fit(X)

    # Get results
    labels = model.labels_
    n_clusters = model.n_clusters_
    n_outliers = model.n_outliers_

    print(f"\nResults:")
    print(f"  Samples: {len(X)}")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Outliers: {n_outliers}")

    # Show cluster sizes
    unique_labels = np.unique(labels)
    print("\nCluster sizes:")
    for label in sorted(unique_labels):
        size = np.sum(labels == label)
        if label == -1:
            print(f"  Outliers: {size}")
        else:
            print(f"  Cluster {label}: {size}")

    # Access intermediate results
    print("\nIntermediate data shapes:")
    print(f"  knn_indices: {model.knn_indices_.shape}")
    print(f"  knn_distances: {model.knn_distances_.shape}")
    print(f"  knn_radius: {model.knn_radius_.shape}")
    print(f"  components: {model.components_.shape}")
    print(f"  big_brother: {model.big_brother_.shape}")
    print(f"  peak_score: {model.peak_score_.shape}")


if __name__ == "__main__":
    main()
