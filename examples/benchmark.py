#!/usr/bin/env python
"""
Benchmark fastCPF (Rust) vs Python CPFcluster.

Compares performance and validates result consistency.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from fastcpf import FastCPF

# Try to import Python CPFcluster for comparison
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CPFcluster"))
    from src import CPFcluster

    HAS_PYTHON_CPF = True
except ImportError:
    HAS_PYTHON_CPF = False
    print("Warning: Python CPFcluster not found. Running fastCPF only.")


def run_fastcpf(X, min_samples, rho, alpha, cutoff):
    """Run clustering using fastCPF (Rust backend)."""
    start = time.perf_counter()
    model = FastCPF(
        min_samples=min_samples,
        rho=rho,
        alpha=alpha,
        cutoff=cutoff,
        knn_backend="kd",
    )
    model.fit(X)
    elapsed = time.perf_counter() - start
    return model.labels_, elapsed


def run_python_cpf(X, min_samples, rho, alpha, cutoff):
    """Run clustering using pure Python backend."""
    start = time.perf_counter()
    cpf = CPFcluster(
        min_samples=min_samples, rho=[rho], alpha=[alpha], cutoff=cutoff, n_jobs=1
    )
    cpf.fit(X)
    labels = list(cpf.clusterings.values())[0]
    elapsed = time.perf_counter() - start
    return np.array(labels), elapsed


def main():
    parser = argparse.ArgumentParser(description="fastCPF Benchmark")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    args = parser.parse_args()

    # Generate data
    print(f"Generating data: {args.n_samples} samples, {args.n_clusters} clusters")
    X, y_true = make_blobs(
        n_samples=args.n_samples,
        n_features=2,
        centers=args.n_clusters,
        cluster_std=0.5,
        random_state=args.seed,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)

    # Parameters
    min_samples = args.k
    rho = 0.4
    alpha = 1.0
    cutoff = 1

    # Benchmark fastCPF (Rust)
    print("\n--- fastCPF (Rust) Backend ---")
    times_rust = []
    labels_rust = None
    for i in range(args.runs):
        labels_rust, elapsed = run_fastcpf(X, min_samples, rho, alpha, cutoff)
        times_rust.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.4f}s")

    n_clusters_rust = len(set(labels_rust) - {-1})
    n_outliers_rust = np.sum(labels_rust == -1)
    ari_rust = adjusted_rand_score(y_true, labels_rust)
    avg_time_rust = np.mean(times_rust)

    print(f"  Average time: {avg_time_rust:.4f}s")
    print(f"  Clusters: {n_clusters_rust}")
    print(f"  Outliers: {n_outliers_rust}")
    print(f"  ARI: {ari_rust:.4f}")

    if HAS_PYTHON_CPF:
        # Benchmark Python CPFcluster
        print("\n--- Python CPFcluster Backend ---")
        times_python = []
        labels_python = None
        for i in range(args.runs):
            labels_python, elapsed = run_python_cpf(X, min_samples, rho, alpha, cutoff)
            times_python.append(elapsed)
            print(f"  Run {i + 1}: {elapsed:.4f}s")

        n_clusters_python = len(set(labels_python) - {-1})
        n_outliers_python = np.sum(labels_python == -1)
        ari_python = adjusted_rand_score(y_true, labels_python)
        avg_time_python = np.mean(times_python)

        print(f"  Average time: {avg_time_python:.4f}s")
        print(f"  Clusters: {n_clusters_python}")
        print(f"  Outliers: {n_outliers_python}")
        print(f"  ARI: {ari_python:.4f}")

        # Comparison
        print("\n--- Comparison ---")
        speedup = avg_time_python / avg_time_rust if avg_time_rust > 0 else float("inf")
        print(f"  Speedup: {speedup:.2f}x")
        label_match = np.array_equal(labels_rust, labels_python)
        print(f"  Labels match: {label_match}")
        ari_between = adjusted_rand_score(labels_rust, labels_python)
        print(f"  ARI between results: {ari_between:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
