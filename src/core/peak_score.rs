//! Peak score computation (Definition 3).

/// Compute gamma(x) ~ omega(x) / r(x), where r(x) is the chosen density radius.
pub fn compute_peak_score(best_distance: &[f32], density_radius: &[f32]) -> Vec<f32> {
    assert_eq!(best_distance.len(), density_radius.len());
    let mut out = vec![0.0f32; best_distance.len()];
    for i in 0..best_distance.len() {
        let r = density_radius[i];
        out[i] = if r == 0.0 { f32::INFINITY } else { best_distance[i] / r };
    }
    out
}
