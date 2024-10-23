use std::cmp::max;
use std::str::FromStr;

use faer::prelude::*;
use faer::solvers::SolverCore;
use faer::{MatRef, Side};
use faer_ext::{IntoFaer, IntoNdarray};

#[allow(unused_imports)]
use ndarray::{
    array, s, Array, Array1, Array2, ArrayBase, ArrayView1, Axis, Dim, Dimension, Ix2, NewAxis,
    OwnedRepr,
};

// Enums and their implementations
#[derive(Debug, PartialEq)]
pub enum SolveMethod {
    QR,
    SVD,
    Cholesky,
    LU,
    CD,          // coordinate-descent for elastic net problem
    CDActiveSet, // coordinate-descent w/ active set
}

impl FromStr for SolveMethod {
    type Err = ();

    fn from_str(input: &str) -> Result<SolveMethod, Self::Err> {
        match input {
            "qr" => Ok(SolveMethod::QR),
            "svd" => Ok(SolveMethod::SVD),
            "chol" => Ok(SolveMethod::Cholesky),
            "lu" => Ok(SolveMethod::LU),
            "cd" => Ok(SolveMethod::CD),
            "cd_active_set" => Ok(SolveMethod::CDActiveSet),
            _ => Err(()),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum NullPolicy {
    Zero,
    Drop,
    Ignore,
    DropZero,
    DropYZeroX,
    DropWindow,
}

impl FromStr for NullPolicy {
    type Err = ();

    fn from_str(input: &str) -> Result<NullPolicy, Self::Err> {
        match input {
            "zero" => Ok(NullPolicy::Zero),
            "drop" => Ok(NullPolicy::Drop),
            "drop_window" => Ok(NullPolicy::DropWindow),
            "ignore" => Ok(NullPolicy::Ignore),
            "drop_y_zero_x" => Ok(NullPolicy::DropYZeroX),
            "drop_zero" => Ok(NullPolicy::DropZero),
            _ => Err(()),
        }
    }
}

// Helper functions
#[inline]
fn soft_threshold(x: &f64, alpha: f64, positive: bool) -> f64 {
    let mut result = x.signum() * (x.abs() - alpha).max(0.0);
    if positive {
        result = result.max(0.0);
    }
    result
}

// Matrix operations
#[inline]
pub fn inv(array: &Array2<f64>, use_cholesky: bool) -> Array2<f64> {
    let m = array.view().into_faer();
    if use_cholesky {
        match m.cholesky(Side::Lower) {
            Ok(cholesky) => {
                return cholesky.inverse().as_ref().into_ndarray().to_owned();
            }
            Err(_) => {
                #[cfg(debug_assertions)]
                println!("Cholesky decomposition failed, falling back to LU decomposition");
            }
        }
    }
    // fall back to LU decomposition
    m.partial_piv_lu()
        .inverse()
        .as_ref()
        .into_ndarray()
        .to_owned()
}

// Least Squares Solvers
#[inline]
pub fn lstsq_solver1(x: MatRef<f64>, y: MatRef<f64>) -> Array2<f64> {
    let xt = x.transpose();
    let xtx = xt * x;
    let cholesky = xtx.cholesky(Side::Lower).unwrap();
    let xtx_inv = cholesky.inverse();
    let beta = xtx_inv * xt * y;
    beta.as_ref().into_ndarray().to_owned()
}

#[inline]
fn solve_ols_qr(y: &Array1<f64>, x: &Array2<f64>) -> Array1<f64> {
    let x_faer = x.view().into_faer();
    let y_faer = y.slice(s![.., NewAxis]).into_faer();
    let coefficients = x_faer.col_piv_qr().solve_lstsq(&y_faer);
    coefficients
        .as_ref()
        .into_ndarray()
        .slice(s![.., 0])
        .to_owned()
}

#[inline]
fn solve_ols_lu(y: &Array1<f64>, x: &Array2<f64>) -> Array1<f64> {
    let x_faer = x.view().into_faer();
    x_faer
        .partial_piv_lu()
        .solve(&y.slice(s![.., NewAxis]).into_faer())
        .as_ref()
        .into_ndarray()
        .slice(s![.., 0])
        .into_owned()
}

// Ridge Regression
#[inline]
fn solve_ridge_svd<I>(
    y: &ArrayBase<OwnedRepr<f64>, I>,
    x: &Array2<f64>,
    alpha: f64,
    rcond: Option<f64>,
) -> ArrayBase<OwnedRepr<f64>, I>
where
    I: Dimension,
{
    // Implementation details...
}

// OLS Solvers
pub fn solve_ols(
    y: &Array1<f64>,
    x: &Array2<f64>,
    solve_method: Option<SolveMethod>,
    rcond: Option<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let x_ref = x.view().into_faer();
    let y_ref = y.view().into_faer();

    let (coefficients, residuals) = match solve_method.unwrap_or(SolveMethod::QR) {
        SolveMethod::QR => lstsq_solver1(x_ref, y_ref),
        SolveMethod::SVD => lstsq_solver2(x_ref, y_ref, rcond.unwrap_or(1e-15)),
        _ => unimplemented!("Other solve methods are not implemented yet"),
    };

    let n = y.len();
    let p = x.ncols();
    let degrees_of_freedom = n - p;

    // Calculate the mean squared error
    let mse = residuals.iter().map(|&r| r * r).sum::<f64>() / degrees_of_freedom as f64;

    // Calculate the diagonal elements of (X^T X)^(-1)
    let xtx_inv = (x_ref.transpose() * x_ref).inv().unwrap();
    let var_coef = xtx_inv.diagonal().to_vec();

    // Calculate standard errors
    let standard_errors: Array1<f64> = var_coef.iter().map(|&v| (v * mse).sqrt()).collect();

    // Calculate t-values
    let t_values: Array1<f64> = coefficients.iter().zip(standard_errors.iter())
        .map(|(&coef, &se)| coef / se)
        .collect();

    let residuals = y - x.dot(&coefficients);

    (Array1::from(coefficients), standard_errors, t_values, residuals)
}

// Normal Equations Solver
#[inline]
fn solve_normal_equations(
    xtx: &Array2<f64>,
    xty: &Array1<f64>,
    solve_method: Option<SolveMethod>,
    fallback_solve_method: Option<SolveMethod>,
) -> Array1<f64> {
    // Implementation details...
}

// Ridge Regression Solver
#[inline]
pub fn solve_ridge(
    y: &Array1<f64>,
    x: &Array2<f64>,
    alpha: f64,
    solve_method: Option<SolveMethod>,
    rcond: Option<f64>,
) -> Array1<f64> {
    // Implementation details...
}

// Elastic Net Solver
#[allow(clippy::too_many_arguments)]
pub fn solve_elastic_net(
    y: &Array1<f64>,
    x: &Array2<f64>,
    alpha: f64,
    l1_ratio: Option<f64>,
    max_iter: Option<usize>,
    tol: Option<f64>,
    positive: Option<bool>,
    solve_method: Option<SolveMethod>,
) -> Array1<f64> {
    // Implementation details...
}

use statrs::distribution::{FisherSnedecor, Continuous};

pub fn calculate_ols_statistics(
    y: &Array1<f64>,
    x: &Array2<f64>,
    coefficients: &Array1<f64>,
    residuals: &Array1<f64>,
) -> (f64, usize, f64, f64, f64, usize, usize, f64) {
    let n = y.len();
    let p = x.ncols();
    let df = n - p;

    // Residual standard error
    let rss: f64 = residuals.iter().map(|&r| r * r).sum();
    let residual_std_error = (rss / df as f64).sqrt();

    // Total sum of squares
    let y_mean = y.mean().unwrap();
    let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    // R-squared
    let r_squared = 1.0 - (rss / tss);

    // Adjusted R-squared
    let adj_r_squared = 1.0 - ((1.0 - r_squared) * (n - 1) as f64 / df as f64);

    // F-statistic
    let msm = (tss - rss) / (p - 1) as f64;
    let mse = rss / df as f64;
    let f_statistic = msm / mse;

    // F-statistic p-value
    let f_distribution = FisherSnedecor::new((p - 1) as f64, df as f64).unwrap();
    let f_p_value = 1.0 - f_distribution.cdf(f_statistic);

    (
        residual_std_error,
        df,
        r_squared,
        adj_r_squared,
        f_statistic,
        p - 1,
        df,
        f_p_value,
    )
}

pub fn calculate_elastic_net_stats(
    x: &Array2<f64>,
    coefficients: &Array1<f64>,
    residuals: &Array1<f64>
) -> (Array1<f64>, Array1<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let degrees_of_freedom = n - p;

    // Calculate the mean squared error
    let mse = residuals.iter().map(|&r| r * r).sum::<f64>() / degrees_of_freedom as f64;

    // Calculate the diagonal elements of (X^T X)^(-1)
    // Note: This is an approximation for Elastic Net
    let xtx_inv = inv(&(x.t().dot(x)), true);
    let var_coef = xtx_inv.diag().to_owned();

    // Calculate standard errors
    let standard_errors = var_coef.mapv(|v| (v * mse).sqrt());

    // Calculate t-values
    let t_values = coefficients.iter().zip(standard_errors.iter())
        .map(|(&coef, &se)| coef / se)
        .collect();

    (standard_errors, t_values)
}

pub fn calculate_regression_statistics(
    y: &Array1<f64>,
    x: &Array2<f64>,
    coefficients: &Array1<f64>,
    residuals: &Array1<f64>,
) -> (f64, usize, f64, f64, f64, usize, usize, f64) {
    let n = y.len();
    let p = x.ncols();
    let df = n - p;

    // Residual standard error
    let rss: f64 = residuals.iter().map(|&r| r * r).sum();
    let residual_std_error = (rss / df as f64).sqrt();

    // Total sum of squares
    let y_mean = y.mean().unwrap();
    let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    // R-squared
    let r_squared = 1.0 - (rss / tss);

    // Adjusted R-squared
    let adj_r_squared = 1.0 - ((1.0 - r_squared) * (n - 1) as f64 / df as f64);

    // F-statistic
    let msm = (tss - rss) / (p - 1) as f64;
    let mse = rss / df as f64;
    let f_statistic = msm / mse;

    // F-statistic p-value
    let f_distribution = FisherSnedecor::new((p - 1) as f64, df as f64).unwrap();
    let f_p_value = 1.0 - f_distribution.cdf(f_statistic);

    (
        residual_std_error,
        df,
        r_squared,
        adj_r_squared,
        f_statistic,
        p - 1,
        df,
        f_p_value,
    )
}
