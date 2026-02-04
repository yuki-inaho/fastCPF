//! fastCPF: Fast Component-wise Peak-Finding clustering algorithm.
//!
//! Python bindings via PyO3 for the Rust implementation.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;

mod core;
mod data;
mod graph;
mod math;
mod types;

use crate::core::{
    assign_labels_for_component, compute_big_brother, compute_density_radius,
    compute_peak_score, select_centers_for_component, BigBrotherResult,
};
use crate::data::Dataset;
use crate::graph::{
    apply_outlier_filter, build_mutual_knn_graph, extract_components, knn_search, KnnBackend,
    KnnResult, OutlierFilter, WeightedGraph,
};
use crate::types::{DensityMethod, OUTLIER};

/// FastCPF: scikit-learn style API for CPF clustering.
#[pyclass]
pub struct FastCPF {
    // Parameters
    min_samples: usize,
    rho: f32,
    alpha: f32,
    cutoff: usize,
    knn_backend: KnnBackend,
    density_method: DensityMethod,

    // Results (None before fit)
    labels: Option<Vec<i32>>,
    n_clusters: Option<i32>,

    // Intermediate results
    knn_result: Option<KnnResult>,
    components: Option<Vec<i32>>,
    graph: Option<WeightedGraph>,
    big_brother: Option<BigBrotherResult>,
    peak_score: Option<Vec<f32>>,
}

#[pymethods]
impl FastCPF {
    /// Create a new FastCPF instance.
    ///
    /// # Arguments
    /// * `min_samples` - Number of neighbors for k-NN (default: 10)
    /// * `rho` - Density scale parameter (default: 0.4)
    /// * `alpha` - Edge cutoff parameter (default: 1.0)
    /// * `cutoff` - Outlier filter threshold (default: 1)
    /// * `knn_backend` - k-NN backend: "kd" or "brute" (default: "kd")
    /// * `density_method` - Density proxy: "rk", "median", or "mean" (default: "rk")
    #[new]
    #[pyo3(signature = (min_samples=10, rho=0.4, alpha=1.0, cutoff=1, knn_backend="kd", density_method="rk"))]
    fn new(
        min_samples: usize,
        rho: f32,
        alpha: f32,
        cutoff: usize,
        knn_backend: &str,
        density_method: &str,
    ) -> PyResult<Self> {
        let backend = match knn_backend.to_lowercase().as_str() {
            "kd" | "kdtree" => KnnBackend::KdTree,
            "brute" | "bruteforce" => KnnBackend::BruteForce,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported knn_backend: '{}'. Use 'kd' or 'brute'.",
                    other
                )))
            }
        };
        let density_method = DensityMethod::from_str(density_method).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported density_method: '{}'. Use 'rk', 'median', or 'mean'.",
                density_method
            ))
        })?;

        Ok(Self {
            min_samples,
            rho,
            alpha,
            cutoff,
            knn_backend: backend,
            density_method,
            labels: None,
            n_clusters: None,
            knn_result: None,
            components: None,
            graph: None,
            big_brother: None,
            peak_score: None,
        })
    }

    /// Fit the model to the data.
    ///
    /// # Arguments
    /// * `x` - Data matrix of shape (n_samples, n_features)
    fn fit(&mut self, x: PyReadonlyArray2<f32>) -> PyResult<()> {
        let dataset = dataset_from_numpy(x)?;
        let n = dataset.n();
        let d = dataset.d();

        // Step 1: k-NN search
        let knn = knn_search(&dataset, self.min_samples, self.knn_backend);
        let knn_radius = knn.radius.clone();
        let density_radius = compute_density_radius(&knn, self.density_method);

        // Step 2: Build mutual k-NN graph
        let graph = build_mutual_knn_graph(&knn);

        // Step 3: Extract connected components
        let raw_components = extract_components(&graph);

        // Step 4: Apply outlier filter (EdgeCount)
        let components =
            apply_outlier_filter(&graph, &raw_components, OutlierFilter::EdgeCount, self.cutoff);

        // Step 5: Compute Big Brother for each component
        let bb = compute_big_brother(&dataset, &density_radius, &components, self.min_samples);

        // Step 6: Compute peak scores
        let peak_score = compute_peak_score(&bb.parent_dist, &density_radius);

        // Step 7: Select centers and assign labels for each component
        let mut labels = vec![OUTLIER; n];
        let mut label_offset = 0i32;

        // Group points by component
        let mut comp_map: std::collections::BTreeMap<i32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for i in 0..n {
            let c = components[i];
            if c != OUTLIER {
                comp_map.entry(c).or_default().push(i);
            }
        }

        for (_comp_id, cc_idx) in comp_map.iter() {
            if cc_idx.is_empty() {
                continue;
            }

            // Extract component-local data
            let nc = cc_idx.len();
            let mut cc_density_radius = Vec::with_capacity(nc);
            let mut cc_peaked = Vec::with_capacity(nc);
            for &gi in cc_idx {
                cc_density_radius.push(density_radius[gi]);
                cc_peaked.push(peak_score[gi]);
            }

            // Select centers
            let centers = select_centers_for_component(
                &graph,
                cc_idx,
                &cc_density_radius,
                &cc_peaked,
                self.rho,
                self.alpha,
                d,
            );

            // Assign labels
            let (n_new_clusters, cc_labels) =
                assign_labels_for_component(cc_idx, &bb.parent, &centers, label_offset);

            // Copy back to global labels
            for (li, &gi) in cc_idx.iter().enumerate() {
                labels[gi] = cc_labels[li];
            }

            label_offset += n_new_clusters;
        }

        // Store results
        self.labels = Some(labels);
        self.n_clusters = Some(label_offset);
        self.knn_result = Some(knn);
        self.components = Some(components);
        self.graph = Some(graph);
        self.big_brother = Some(bb);
        self.peak_score = Some(peak_score);

        Ok(())
    }

    // ========== Result getters ==========

    /// Cluster labels for each sample. -1 indicates outlier.
    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        match &self.labels {
            Some(labels) => Ok(labels.clone().into_pyarray_bound(py)),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model has not been fitted. Call fit() first.",
            )),
        }
    }

    /// Number of clusters found (excluding outliers).
    #[getter]
    fn n_clusters_(&self) -> PyResult<i32> {
        match self.n_clusters {
            Some(n) => Ok(n),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model has not been fitted. Call fit() first.",
            )),
        }
    }

    /// Number of outlier samples.
    #[getter]
    fn n_outliers_(&self) -> PyResult<i32> {
        match &self.labels {
            Some(labels) => {
                let count = labels.iter().filter(|&&l| l == OUTLIER).count();
                Ok(count as i32)
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model has not been fitted. Call fit() first.",
            )),
        }
    }

    // ========== Intermediate result getters ==========

    /// k-NN indices of shape (n_samples, k).
    #[getter]
    fn knn_indices_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<usize>>> {
        match &self.knn_result {
            Some(knn) => {
                let n = knn.indices.len();
                let k = knn.k;
                let mut data = Vec::with_capacity(n * k);
                for row in &knn.indices {
                    for j in 0..k {
                        data.push(if j < row.len() { row[j] } else { 0 });
                    }
                }
                Ok(numpy::PyArray::from_vec_bound(py, data)
                    .reshape([n, k])
                    .unwrap())
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model has not been fitted. Call fit() first.",
            )),
        }
    }

    /// k-NN distances of shape (n_samples, k).
    #[getter]
    fn knn_distances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        match &self.knn_result {
            Some(knn) => {
                let n = knn.distances.len();
                let k = knn.k;
                let mut data = Vec::with_capacity(n * k);
                for row in &knn.distances {
                    for j in 0..k {
                        data.push(if j < row.len() { row[j] } else { f32::INFINITY });
                    }
                }
                Ok(numpy::PyArray::from_vec_bound(py, data)
                    .reshape([n, k])
                    .unwrap())
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model has not been fitted. Call fit() first.",
            )),
        }
    }

    /// k-NN radius r_k(x) for each sample.
    #[getter]
    fn knn_radius_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        match &self.knn_result {
            Some(knn) => Ok(knn.radius.clone().into_pyarray_bound(py)),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model has not been fitted. Call fit() first.",
            )),
        }
    }

    /// Connected component labels (before clustering).
    #[getter]
    fn components_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        match &self.components {
            Some(comp) => Ok(comp.clone().into_pyarray_bound(py)),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model has not been fitted. Call fit() first.",
            )),
        }
    }

    /// Big Brother index b(x) for each sample.
    #[getter]
    fn big_brother_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        match &self.big_brother {
            Some(bb) => Ok(bb.parent.clone().into_pyarray_bound(py)),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model has not been fitted. Call fit() first.",
            )),
        }
    }

    /// Big Brother distance omega(x) for each sample.
    #[getter]
    fn big_brother_dist_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        match &self.big_brother {
            Some(bb) => Ok(bb.parent_dist.clone().into_pyarray_bound(py)),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model has not been fitted. Call fit() first.",
            )),
        }
    }

    /// Peak score gamma(x) for each sample.
    #[getter]
    fn peak_score_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        match &self.peak_score {
            Some(ps) => Ok(ps.clone().into_pyarray_bound(py)),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model has not been fitted. Call fit() first.",
            )),
        }
    }

    // ========== Parameter getters ==========

    #[getter]
    fn min_samples(&self) -> usize {
        self.min_samples
    }

    #[getter]
    fn rho(&self) -> f32 {
        self.rho
    }

    #[getter]
    fn alpha(&self) -> f32 {
        self.alpha
    }

    #[getter]
    fn cutoff(&self) -> usize {
        self.cutoff
    }

    #[getter]
    fn density_method(&self) -> String {
        self.density_method.as_str().to_string()
    }
}

/// Convert a NumPy float32 2D array into the Rust Dataset (row-major).
fn dataset_from_numpy(array: PyReadonlyArray2<f32>) -> PyResult<Dataset> {
    let arr = array.as_array();
    let (n, d) = arr.dim();
    let mut data = Vec::with_capacity(n * d);
    for i in 0..n {
        for j in 0..d {
            data.push(arr[(i, j)]);
        }
    }
    Dataset::new(n, d, data).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}

/// Python module definition.
#[pymodule]
fn _fastcpf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastCPF>()?;
    Ok(())
}
