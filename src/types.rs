//! Shared sentinels and type aliases.

pub const OUTLIER: i32 = -1;
pub const NO_PARENT: i32 = -1;

/// Density proxy used for peak finding and center selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensityMethod {
    /// Use r_k(x): distance to the k-th neighbor (paper default).
    Rk,
    /// Use median of kNN distances (robust to outliers).
    Median,
    /// Use mean of kNN distances.
    Mean,
}

impl DensityMethod {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rk" | "r_k" | "knn_radius" => Some(DensityMethod::Rk),
            "median" => Some(DensityMethod::Median),
            "mean" | "avg" | "average" => Some(DensityMethod::Mean),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            DensityMethod::Rk => "rk",
            DensityMethod::Median => "median",
            DensityMethod::Mean => "mean",
        }
    }
}
