"""Type stubs for fastCPF."""

from typing import Literal
import numpy as np
import numpy.typing as npt

class FastCPF:
    """
    Fast Component-wise Peak-Finding (CPF) clustering algorithm.

    A density-based clustering algorithm that identifies cluster centers
    as local density maxima and assigns points by following density gradients.

    Parameters
    ----------
    min_samples : int, default=10
        Number of neighbors for k-NN computation.
    rho : float, default=0.4
        Density scale parameter for modal-set selection.
    alpha : float, default=1.0
        Edge cutoff parameter.
    cutoff : int, default=1
        Outlier filter threshold (minimum edge count).
    knn_backend : {"kd", "brute"}, default="kd"
        Backend for k-NN search. "kd" uses KD-tree, "brute" uses brute-force.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each sample. -1 indicates outlier.
    n_clusters_ : int
        Number of clusters found (excluding outliers).
    n_outliers_ : int
        Number of outlier samples.
    knn_indices_ : ndarray of shape (n_samples, min_samples)
        Indices of k-nearest neighbors for each sample.
    knn_distances_ : ndarray of shape (n_samples, min_samples)
        Distances to k-nearest neighbors for each sample.
    knn_radius_ : ndarray of shape (n_samples,)
        k-NN radius r_k(x) for each sample (Definition 1).
    components_ : ndarray of shape (n_samples,)
        Connected component labels from mutual k-NN graph.
    big_brother_ : ndarray of shape (n_samples,)
        Big Brother index b(x) for each sample (Definition 2).
    big_brother_dist_ : ndarray of shape (n_samples,)
        Distance to Big Brother omega(x) for each sample (Definition 2).
    peak_score_ : ndarray of shape (n_samples,)
        Peak score gamma(x) for each sample (Definition 3).

    Examples
    --------
    >>> from fastcpf import FastCPF
    >>> import numpy as np
    >>> X = np.random.randn(1000, 2).astype(np.float32)
    >>> model = FastCPF(min_samples=10, rho=0.4)
    >>> model.fit(X)
    >>> labels = model.labels_
    >>> print(f"Found {model.n_clusters_} clusters")
    """

    def __init__(
        self,
        min_samples: int = 10,
        rho: float = 0.4,
        alpha: float = 1.0,
        cutoff: int = 1,
        knn_backend: Literal["kd", "brute"] = "kd",
    ) -> None: ...
    def fit(self, x: npt.NDArray[np.float32]) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Training data. Must be float32.
        """
        ...

    # Result properties
    @property
    def labels_(self) -> npt.NDArray[np.int32]:
        """Cluster labels for each sample. -1 indicates outlier."""
        ...

    @property
    def n_clusters_(self) -> int:
        """Number of clusters found (excluding outliers)."""
        ...

    @property
    def n_outliers_(self) -> int:
        """Number of outlier samples."""
        ...

    # Intermediate result properties
    @property
    def knn_indices_(self) -> npt.NDArray[np.intp]:
        """k-NN indices of shape (n_samples, k)."""
        ...

    @property
    def knn_distances_(self) -> npt.NDArray[np.float32]:
        """k-NN distances of shape (n_samples, k)."""
        ...

    @property
    def knn_radius_(self) -> npt.NDArray[np.float32]:
        """k-NN radius r_k(x) for each sample."""
        ...

    @property
    def components_(self) -> npt.NDArray[np.int32]:
        """Connected component labels (before clustering)."""
        ...

    @property
    def big_brother_(self) -> npt.NDArray[np.int32]:
        """Big Brother index b(x) for each sample."""
        ...

    @property
    def big_brother_dist_(self) -> npt.NDArray[np.float32]:
        """Big Brother distance omega(x) for each sample."""
        ...

    @property
    def peak_score_(self) -> npt.NDArray[np.float32]:
        """Peak score gamma(x) for each sample."""
        ...

    # Parameter properties
    @property
    def min_samples(self) -> int:
        """Number of neighbors for k-NN."""
        ...

    @property
    def rho(self) -> float:
        """Density scale parameter."""
        ...

    @property
    def alpha(self) -> float:
        """Edge cutoff parameter."""
        ...

    @property
    def cutoff(self) -> int:
        """Outlier filter threshold."""
        ...

__all__: list[str]
__version__: str
