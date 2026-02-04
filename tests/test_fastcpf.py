"""Tests for fastCPF package."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


class TestFastCPF:
    """Test suite for FastCPF class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        X, y_true = make_blobs(
            n_samples=300,
            n_features=2,
            centers=3,
            cluster_std=0.5,
            random_state=42,
        )
        X = StandardScaler().fit_transform(X).astype(np.float32)
        return X, y_true

    def test_import(self):
        """Test that fastcpf can be imported."""
        from fastcpf import FastCPF

        assert FastCPF is not None

    def test_basic_clustering(self, sample_data):
        """Test basic clustering functionality."""
        from fastcpf import FastCPF

        X, y_true = sample_data

        model = FastCPF(min_samples=10, rho=0.4, alpha=1.0)
        model.fit(X)

        labels = model.labels_
        assert len(labels) == len(X)
        assert model.n_clusters_ >= 1

    def test_labels_shape(self, sample_data):
        """Test that labels have correct shape."""
        from fastcpf import FastCPF

        X, _ = sample_data

        model = FastCPF(min_samples=10)
        model.fit(X)

        assert model.labels_.shape == (len(X),)
        assert model.labels_.dtype == np.int32

    def test_intermediate_results(self, sample_data):
        """Test that intermediate results are accessible."""
        from fastcpf import FastCPF

        X, _ = sample_data
        k = 10

        model = FastCPF(min_samples=k)
        model.fit(X)

        # Check knn results
        assert model.knn_indices_.shape == (len(X), k)
        assert model.knn_distances_.shape == (len(X), k)
        assert model.knn_radius_.shape == (len(X),)

        # Check components
        assert model.components_.shape == (len(X),)

        # Check big brother
        assert model.big_brother_.shape == (len(X),)
        assert model.big_brother_dist_.shape == (len(X),)

        # Check peak score
        assert model.peak_score_.shape == (len(X),)

    def test_unfitted_model_raises(self):
        """Test that accessing results before fit raises error."""
        from fastcpf import FastCPF

        model = FastCPF()

        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = model.labels_

        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = model.n_clusters_

    def test_parameter_getters(self):
        """Test that parameter getters work."""
        from fastcpf import FastCPF

        model = FastCPF(min_samples=15, rho=0.5, alpha=1.2, cutoff=2, density_method="median")

        assert model.min_samples == 15
        assert model.rho == pytest.approx(0.5)
        assert model.alpha == pytest.approx(1.2)
        assert model.cutoff == 2
        assert model.density_method == "median"

    def test_knn_backends(self, sample_data):
        """Test both k-NN backends produce similar results."""
        from fastcpf import FastCPF

        X, _ = sample_data

        model_kd = FastCPF(min_samples=10, knn_backend="kd")
        model_kd.fit(X)

        model_brute = FastCPF(min_samples=10, knn_backend="brute")
        model_brute.fit(X)

        # Results should be identical or very similar
        ari = adjusted_rand_score(model_kd.labels_, model_brute.labels_)
        assert ari > 0.95, f"ARI between backends: {ari}"

    def test_invalid_backend_raises(self):
        """Test that invalid k-NN backend raises error."""
        from fastcpf import FastCPF

        with pytest.raises(ValueError, match="Unsupported knn_backend"):
            FastCPF(knn_backend="invalid")

    def test_density_method_median(self, sample_data):
        """Test median-based density proxy runs."""
        from fastcpf import FastCPF

        X, _ = sample_data
        model = FastCPF(min_samples=10, density_method="median")
        model.fit(X)
        assert model.labels_.shape == (len(X),)

    def test_density_method_mean(self, sample_data):
        """Test mean-based density proxy runs."""
        from fastcpf import FastCPF

        X, _ = sample_data
        model = FastCPF(min_samples=10, density_method="mean")
        model.fit(X)
        assert model.labels_.shape == (len(X),)

    def test_invalid_density_method_raises(self):
        """Test that invalid density method raises error."""
        from fastcpf import FastCPF

        with pytest.raises(ValueError, match="Unsupported density_method"):
            FastCPF(density_method="invalid")

    def test_clustering_quality(self, sample_data):
        """Test that clustering quality is reasonable."""
        from fastcpf import FastCPF

        X, y_true = sample_data

        model = FastCPF(min_samples=10, rho=0.4)
        model.fit(X)

        # ARI should be reasonably high for well-separated clusters
        ari = adjusted_rand_score(y_true, model.labels_)
        assert ari > 0.8, f"ARI too low: {ari}"

    def test_outlier_count(self, sample_data):
        """Test that outlier count matches labels."""
        from fastcpf import FastCPF

        X, _ = sample_data

        model = FastCPF(min_samples=10, cutoff=1)
        model.fit(X)

        expected_outliers = np.sum(model.labels_ == -1)
        assert model.n_outliers_ == expected_outliers

    def test_n_clusters_count(self, sample_data):
        """Test that n_clusters matches unique positive labels."""
        from fastcpf import FastCPF

        X, _ = sample_data

        model = FastCPF(min_samples=10)
        model.fit(X)

        unique_labels = set(model.labels_) - {-1}
        assert model.n_clusters_ == len(unique_labels)

    def test_empty_input_handling(self):
        """Test handling of edge cases."""
        from fastcpf import FastCPF

        # Very small dataset
        X = np.random.randn(5, 2).astype(np.float32)
        model = FastCPF(min_samples=3)
        model.fit(X)

        assert len(model.labels_) == 5

    def test_high_dimensional_data(self):
        """Test with higher dimensional data."""
        from fastcpf import FastCPF

        X, _ = make_blobs(n_samples=200, n_features=10, centers=3, random_state=42)
        X = X.astype(np.float32)

        model = FastCPF(min_samples=10)
        model.fit(X)

        assert len(model.labels_) == 200
        assert model.n_clusters_ >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
