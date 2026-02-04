//! Density proxy computation used in CPF.

use crate::graph::KnnResult;
use crate::types::DensityMethod;

/// Compute the density radius used for peak finding.
///
/// - Rk: r_k(x) (distance to k-th neighbor)
/// - Median: median of kNN distances (excluding self)
pub fn compute_density_radius(knn: &KnnResult, method: DensityMethod) -> Vec<f32> {
    match method {
        DensityMethod::Rk => knn.radius.clone(),
        DensityMethod::Median => compute_median_radius(knn),
        DensityMethod::Mean => compute_mean_radius(knn),
    }
}

fn compute_median_radius(knn: &KnnResult) -> Vec<f32> {
    let n = knn.distances.len();
    let mut out = Vec::with_capacity(n);
    for (i, row) in knn.distances.iter().enumerate() {
        if row.len() <= 1 {
            // Fallback to r_k when only self is present.
            out.push(knn.radius[i]);
            continue;
        }
        let mut vals: Vec<f32> = row.iter().skip(1).copied().collect();
        if vals.is_empty() {
            out.push(knn.radius[i]);
            continue;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = vals.len();
        let median = if m % 2 == 1 {
            vals[m / 2]
        } else {
            0.5 * (vals[m / 2 - 1] + vals[m / 2])
        };
        out.push(median);
    }
    out
}

fn compute_mean_radius(knn: &KnnResult) -> Vec<f32> {
    let n = knn.distances.len();
    let mut out = Vec::with_capacity(n);
    for (i, row) in knn.distances.iter().enumerate() {
        if row.len() <= 1 {
            out.push(knn.radius[i]);
            continue;
        }
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for &v in row.iter().skip(1) {
            sum += v;
            count += 1;
        }
        if count == 0 {
            out.push(knn.radius[i]);
        } else {
            out.push(sum / count as f32);
        }
    }
    out
}
