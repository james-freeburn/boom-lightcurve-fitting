//! Smoothly Broken Power Law (SBPL) fitting across all bands simultaneously.
//!
//! Model (from afterglow physics):
//!
//!   F(t, ν) = 10^loga · (ν/1e15)^β · τ^α₁ · [0.5·(1 + τ^(1/D))]^((α₂−α₁)·D)
//!   where τ = (t−t₀)/tb
//!
//! During fitting, fluxes are normalised by max_flux and ν is divided by 1e15 Hz.
//! After fitting, loga is corrected back to physical units:
//!   loga_phys = loga_fit + log10(max_flux) − 15·β
//!
/// Optimisation: deterministic multi-restart PSO for global search, followed
/// by L-BFGS polishing for local refinement. The best result is returned.

use std::collections::HashMap;

use argmin::core::{CostFunction, Error as ArgminError, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::common::BandData;
use crate::gp2d::get_band_wavelength;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

const C_ANGSTROM_PER_SEC: f64 = 2.997_924_58e18; // c in Å/s

fn band_frequency_hz(band: &str) -> Option<f64> {
    let lambda = get_band_wavelength(band)?;
    if lambda <= 0.0 || !lambda.is_finite() {
        return None;
    }
    Some(C_ANGSTROM_PER_SEC / lambda)
}

// ---------------------------------------------------------------------------
// SBPL model (internal: nu_scaled = nu / 1e15)
// ---------------------------------------------------------------------------

/// Evaluate the SBPL model at a single (t, nu_scaled) point.
///
/// `nu_scaled` is ν in units of 1e15 Hz.
/// `loga` is log10 of the normalised flux amplitude.
fn sbpl_model(
    t: f64,
    nu_scaled: f64,
    alpha1: f64,
    alpha2: f64,
    beta: f64,
    logd: f64,
    loga: f64,
    tb: f64,
    t0: f64,
) -> f64 {
    let tau = t - t0;

    if tb <= 0.0 || nu_scaled <= 0.0 {
        return f64::NAN;
    }
    if tau < 0.0 {
        return 0.0;
    }
    let ratio = tau / tb;
    if ratio == 0.0 || !ratio.is_finite() {
        return f64::NAN;
    }
    let term1 = nu_scaled.powf(beta);
    let term2 = ratio.powf(alpha1);
    let inner = 0.5 * (1.0 + ratio.powf(1.0 / 10f64.powf(logd)));
    if !inner.is_finite() || inner <= 0.0 {
        return f64::NAN;
    }
    let term3 = inner.powf((alpha2 - alpha1) * 10f64.powf(logd));
    let result = 10f64.powf(loga) * term1 * term2 * term3;
    if result.is_finite() { result } else { f64::NAN }
}

// ---------------------------------------------------------------------------
// Cost function
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct SbplObs {
    time: f64,
    nu_scaled: f64, // ν / 1e15 Hz
    flux: f64,      // normalised by max_flux
    flux_err: f64,  // normalised by max_flux
}

#[derive(Clone)]
struct SbplCost {
    observations: Vec<SbplObs>,
}

impl SbplCost {
    /// Evaluate chi2/n_valid for parameter vector `p = [alpha1, alpha2, beta, logd, loga, tb, t0]`.
    fn eval(&self, p: &[f64]) -> f64 {
        let (alpha1, alpha2, beta, logd, loga, tb, t0) =
            (p[0], p[1], p[2], p[3], p[4], p[5], p[6]);

        let mut chi2 = 0.0;
        let mut n_valid = 0usize;
        for obs in &self.observations {
            let model =
                sbpl_model(obs.time, obs.nu_scaled, alpha1, alpha2, beta, logd, loga, tb, t0);
            if !model.is_finite() {
                return 1e10;
            }
            let residual = obs.flux - model;
            let err_sq = obs.flux_err * obs.flux_err + 1e-30;
            chi2 += residual * residual / err_sq;
            n_valid += 1;
        }
        if n_valid == 0 {
            return 1e10;
        }
        chi2 / n_valid as f64
    }
}

// ---------------------------------------------------------------------------
// Scaled wrapper for L-BFGS (all params mapped to [0, 1])
// ---------------------------------------------------------------------------
//
// Transforming to unit-cube space means every parameter looks the same scale
// to L-BFGS, giving it a well-conditioned Hessian approximation — equivalent
// to scipy.optimize.least_squares' internal x_scale handling.

struct ScaledSbplCost<'a> {
    inner: &'a SbplCost,
    lower: Vec<f64>,
    scale: Vec<f64>, // upper - lower
}

impl ScaledSbplCost<'_> {
    /// Map unit-cube params back to physical space, clamping to [0, 1] first.
    fn unscale(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter()
            .enumerate()
            .map(|(i, &v)| self.lower[i] + v.clamp(0.0, 1.0) * self.scale[i])
            .collect()
    }
}

impl CostFunction for ScaledSbplCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, xs: &Self::Param) -> Result<Self::Output, ArgminError> {
        Ok(self.inner.eval(&self.unscale(xs)))
    }
}

impl Gradient for ScaledSbplCost<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    /// Central finite-difference gradient in scaled space (h=1e-5 appropriate
    /// for all params since they all live in [0, 1]).
    fn gradient(&self, xs: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let n = xs.len();
        let h = 1e-5;
        let mut grad = vec![0.0; n];
        for i in 0..n {
            let mut xp = xs.clone();
            let mut xm = xs.clone();
            xp[i] = (xs[i] + h).min(1.0);
            xm[i] = (xs[i] - h).max(0.0);
            let step = xp[i] - xm[i];
            if step > 0.0 {
                grad[i] = (self.inner.eval(&self.unscale(&xp))
                    - self.inner.eval(&self.unscale(&xm)))
                    / step;
            }
        }
        Ok(grad)
    }
}

/// Refine a starting point with L-BFGS in unit-cube parameter space.
/// Falls back to the starting point on failure.
fn lbfgs_refine(
    problem: &SbplCost,
    start: Vec<f64>,
    start_cost: f64,
    lower: &[f64],
    upper: &[f64],
) -> (Vec<f64>, f64) {
    let scale: Vec<f64> = lower
        .iter()
        .zip(upper.iter())
        .map(|(&lo, &hi)| hi - lo)
        .collect();

    // Map start to [0, 1]^7
    let xs0: Vec<f64> = start
        .iter()
        .enumerate()
        .map(|(i, &v)| ((v - lower[i]) / scale[i]).clamp(0.0, 1.0))
        .collect();

    let scaled = ScaledSbplCost {
        inner: problem,
        lower: lower.to_vec(),
        scale: scale.clone(),
    };

    let linesearch = MoreThuenteLineSearch::new();
    let solver = match LBFGS::new(linesearch, 10).with_tolerance_grad(1e-7) {
        Ok(s) => s,
        Err(_) => return (start, start_cost),
    };

    let result = Executor::new(scaled, solver)
        .configure(|state| state.param(xs0).max_iters(200))
        .run();

    match result {
        Ok(res) => {
            let xs_best = res.state().best_param.clone().unwrap_or_default();
            if xs_best.is_empty() {
                return (start, start_cost);
            }
            // Unscale back to physical space
            let x_best: Vec<f64> = xs_best
                .iter()
                .enumerate()
                .map(|(i, &v)| lower[i] + v.clamp(0.0, 1.0) * scale[i])
                .collect();
            let final_cost = problem.eval(&x_best);
            if final_cost < start_cost && final_cost.is_finite() {
                (x_best, final_cost)
            } else {
                (start, start_cost)
            }
        }
        Err(_) => (start, start_cost),
    }
}

/// Global search with particle swarm optimisation in physical parameter space.
/// Returns (best_params, best_cost).
fn pso_search(
    problem: &SbplCost,
    lower: &[f64],
    upper: &[f64],
    n_particles: usize,
    n_iters: usize,
    rng: &mut rand::rngs::SmallRng,
) -> (Vec<f64>, f64) {
    let n_dim = lower.len();
    let span: Vec<f64> = lower
        .iter()
        .zip(upper.iter())
        .map(|(&lo, &hi)| hi - lo)
        .collect();

    let mut pos: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| {
            lower
                .iter()
                .zip(upper.iter())
                .map(|(&lo, &hi)| rng.random_range(lo..hi))
                .collect()
        })
        .collect();

    let mut vel: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| {
            span.iter()
                .map(|&s| rng.random_range(-0.1 * s..0.1 * s))
                .collect()
        })
        .collect();

    let mut pbest_pos = pos.clone();
    let mut pbest_cost: Vec<f64> = pos.iter().map(|p| problem.eval(p)).collect();

    let mut gbest_idx = 0usize;
    for i in 1..n_particles {
        if pbest_cost[i] < pbest_cost[gbest_idx] {
            gbest_idx = i;
        }
    }
    let mut gbest_pos = pbest_pos[gbest_idx].clone();
    let mut gbest_cost = pbest_cost[gbest_idx];

    let w = 0.7298;
    let c1 = 1.4962;
    let c2 = 1.4962;

    for _ in 0..n_iters {
        for i in 0..n_particles {
            for d in 0..n_dim {
                let r1: f64 = rng.random();
                let r2: f64 = rng.random();
                vel[i][d] = w * vel[i][d]
                    + c1 * r1 * (pbest_pos[i][d] - pos[i][d])
                    + c2 * r2 * (gbest_pos[d] - pos[i][d]);
                pos[i][d] = (pos[i][d] + vel[i][d]).clamp(lower[d], upper[d]);
            }

            let cost = problem.eval(&pos[i]);
            if cost < pbest_cost[i] {
                pbest_cost[i] = cost;
                pbest_pos[i] = pos[i].clone();
            }
            if cost < gbest_cost {
                gbest_cost = cost;
                gbest_pos = pos[i].clone();
            }
        }
    }

    (gbest_pos, gbest_cost)
}

// ---------------------------------------------------------------------------
// Result struct
// ---------------------------------------------------------------------------

/// Result of a multi-band SBPL fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SbplResult {
    /// Pre-break temporal power-law index.
    pub alpha1: Option<f64>,
    /// Post-break temporal power-law index.
    pub alpha2: Option<f64>,
    /// Spectral index (F_ν ∝ ν^β).
    pub beta: Option<f64>,
    /// logarithm of the break smoothness parameter D.
    pub logd: Option<f64>,
    /// Physical log10 flux normalisation (after rescaling from normalised fit space).
    pub loga: Option<f64>,
    /// Break time tb (days).
    pub tb: Option<f64>,
    /// Time zero-point t0 (days).
    pub t0: Option<f64>,
    /// Uncertainties (std dev of physical parameters across restarts).
    pub alpha1_err: Option<f64>,
    pub alpha2_err: Option<f64>,
    pub beta_err: Option<f64>,
    pub logd_err: Option<f64>,
    pub loga_err: Option<f64>,
    pub tb_err: Option<f64>,
    pub t0_err: Option<f64>,
    /// Reduced chi-squared of the best fit (evaluated in normalised space).
    pub reduced_chi2: Option<f64>,
    /// Number of observations used.
    pub n_obs: usize,
    /// Number of bands used.
    pub n_bands: usize,
}

impl SbplResult {
    fn empty(n_obs: usize, n_bands: usize) -> Self {
        SbplResult {
            alpha1: None,
            alpha2: None,
            beta: None,
            logd: None,
            loga: None,
            tb: None,
            t0: None,
            alpha1_err: None,
            alpha2_err: None,
            beta_err: None,
            logd_err: None,
            loga_err: None,
            tb_err: None,
            t0_err: None,
            reduced_chi2: None,
            n_obs,
            n_bands,
        }
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Fit the SBPL model to multi-band flux-space data.
///
/// `bands` should contain linear flux values (not magnitudes) in standard
/// `BandData` format. Bands without a known effective frequency are skipped.
///
/// Returns `None` if there are fewer than 7 usable observations.
pub fn fit_sbpl(bands: &HashMap<String, BandData>) -> Option<SbplResult> {
    // -----------------------------------------------------------------------
    // Collect observations
    // -----------------------------------------------------------------------
    let mut observations: Vec<SbplObs> = Vec::new();
    let mut n_bands = 0usize;
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;

    let mut band_names: Vec<&String> = bands.keys().collect();
    band_names.sort_unstable();

    for band_name in band_names {
        let Some(band_data) = bands.get(band_name) else {
            continue;
        };
        let nu = match band_frequency_hz(band_name) {
            Some(n) => n,
            None => continue,
        };
        if band_data.times.is_empty() {
            continue;
        }
        n_bands += 1;
        for i in 0..band_data.times.len() {
            let t = band_data.times[i];
            let flux = band_data.values[i];
            let flux_err = band_data.errors[i];
            if !t.is_finite() || !flux.is_finite() || !flux_err.is_finite() || flux_err <= 0.0 {
                continue;
            }
            t_min = t_min.min(t);
            t_max = t_max.max(t);
            // Store nu scaled to 1e15 Hz (same as notebook: nu / 1e15)
            observations.push(SbplObs {
                time: t,
                nu_scaled: nu / 1e15,
                flux,
                flux_err,
            });
        }
    }

    let n_obs = observations.len();
    if n_obs < 7 || n_bands < 2 {
        return Some(SbplResult::empty(n_obs, n_bands));
    }

    let duration = t_max - t_min;
    if duration <= 0.0 {
        return Some(SbplResult::empty(n_obs, n_bands));
    }

    // -----------------------------------------------------------------------
    // Normalise fluxes by max positive flux
    // -----------------------------------------------------------------------
    let max_flux = observations
        .iter()
        .map(|o| o.flux)
        .filter(|f| f.is_finite() && *f > 0.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_flux = if max_flux > 0.0 { max_flux } else { 1.0 };
    for obs in &mut observations {
        obs.flux /= max_flux;
        obs.flux_err /= max_flux;
    }

    // -----------------------------------------------------------------------
    // Bounds: [alpha1, alpha2, beta, d, loga, tb, t0]
    // loga here is in normalised space; physical loga is recovered after the fit.
    // -----------------------------------------------------------------------
    let lower = vec![
        -10.0,                   // alpha1
        -10.0,                   // alpha2
        -5.0,                   // beta
        -3.0,                    // logd
        -10.0,                   // loga (normalised)
        t_min + 0.01 * duration, // tb
        t_min - 100.0,  // t0
    ];
    let upper = vec![
        10.0,  // alpha1
        10.0,  // alpha2
        5.0,  // beta
        0.0,   // logd
        10.0,  // loga (normalised)
        t_max, // tb
        t_max, // t0
    ];

    let problem = SbplCost { observations };

    // -----------------------------------------------------------------------
    // Multi-start PSO + L-BFGS: PSO provides global exploration and L-BFGS
    // performs local polishing.
    // -----------------------------------------------------------------------
    const N_RESTARTS: usize = 3;
    const N_PARTICLES: usize = 30;
    const N_ITERS: usize = 200;
    const SEED: u64 = 1234;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);

    let mut all_params: Vec<Vec<f64>> = Vec::new();
    let mut best_cost = f64::INFINITY;
    let mut best_params: Vec<f64> = lower.clone();

    for _ in 0..N_RESTARTS {
        let (pso_best, pso_cost) =
            pso_search(&problem, &lower, &upper, N_PARTICLES, N_ITERS, &mut rng);
        let (refined_best, refined_cost) = 
            lbfgs_refine(&problem, pso_best, pso_cost, &lower, &upper);
        all_params.push(refined_best.clone());
        if refined_cost < best_cost {
            best_cost = refined_cost;
            best_params = refined_best;
        }
    }

    if all_params.is_empty() {
        return Some(SbplResult::empty(n_obs, n_bands));
    }

    // -----------------------------------------------------------------------
    // Convert fitted loga → physical loga for every restart
    //   loga_phys = loga_fit + log10(max_flux) − 15·beta
    // -----------------------------------------------------------------------
    let log10_max_flux = max_flux.log10();
    for params in &mut all_params {
        let beta_i = params[2];
        params[4] = params[4] + log10_max_flux - 15.0 * beta_i;
    }
    {
        let beta = best_params[2];
        best_params[4] = best_params[4] + log10_max_flux - 15.0 * beta;
    }

    // -----------------------------------------------------------------------
    // Uncertainty estimate: std dev of physical parameters across restarts
    // -----------------------------------------------------------------------
    let n_params = 7;
    let n_r = all_params.len() as f64;
    let means: Vec<f64> = (0..n_params)
        .map(|i| all_params.iter().map(|p| p[i]).sum::<f64>() / n_r)
        .collect();
    let stds: Vec<f64> = (0..n_params)
        .map(|i| {
            let var = all_params
                .iter()
                .map(|p| (p[i] - means[i]).powi(2))
                .sum::<f64>()
                / n_r;
            var.sqrt()
        })
        .collect();

    let finite_or_none = |v: f64| if v.is_finite() { Some(v) } else { None };

    Some(SbplResult {
        alpha1: finite_or_none(best_params[0]),
        alpha2: finite_or_none(best_params[1]),
        beta: finite_or_none(best_params[2]),
        logd: finite_or_none(best_params[3]),
        loga: finite_or_none(best_params[4]),
        tb: finite_or_none(best_params[5]),
        t0: finite_or_none(best_params[6]),
        alpha1_err: finite_or_none(stds[0]),
        alpha2_err: finite_or_none(stds[1]),
        beta_err: finite_or_none(stds[2]),
        logd_err: finite_or_none(stds[3]),
        loga_err: finite_or_none(stds[4]),
        tb_err: finite_or_none(stds[5]),
        t0_err: finite_or_none(stds[6]),
        reduced_chi2: finite_or_none(best_cost),
        n_obs,
        n_bands,
    })
}
