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
) -> Array1<f64> {
    // Implementation details...
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
