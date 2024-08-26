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



#[inline]
pub fn lstsq_solver1(x: MatRef<f64>, y: MatRef<f64>) -> Array2<f64> {
    // Solver1. Use closed form solution to solve the least square
    // This is faster because xtx has small dimension. So we use the closed
    // form solution approach.
    let xt = x.transpose();
    let xtx = xt * x;
    let cholesky = xtx.cholesky(Side::Lower).unwrap(); // Can unwrap because xtx is positive semidefinite
    let xtx_inv = cholesky.inverse();
    // Solution
    let beta = xtx_inv * xt * y;
    let out = beta.as_ref().into_ndarray();
    out.to_owned()
}


/// Invert square matrix input using either Cholesky or LU decomposition
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


/// Solves ridge regression using Singular Value Decomposition (SVD).
///
/// # Arguments
///
/// * `y` - Target vector.
/// * `x` - Feature matrix.
/// * `alpha` - Ridge parameter.
/// * `rcond` - Relative condition number used to determine cutoff for small singular values.
///
/// # Returns
///
/// * Result of ridge regression as a 1-dimensional array.
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
    let x_faer = x.view().into_faer();

    let y_faer = if y.ndim() == 2 {
        y.view()
            .into_dimensionality::<Ix2>()
            .expect("could not reshape y")
            .into_faer()
    } else {
        y.view()
            .insert_axis(Axis(1))
            .into_dimensionality::<Ix2>()
            .expect("could not reshape y")
            .into_faer()
    };

    let is_multi_target = y_faer.ncols() > 1;

    // compute SVD and extract u, s, vt
    let svd = x_faer.thin_svd();
    let u = svd.u();
    let v = svd.v().into_ndarray();
    let s = svd.s_diagonal();

    // convert s into ndarray
    let s: Array1<f64> = s.as_2d().into_ndarray().slice(s![.., 0]).into_owned();
    let max_value = s.iter().skip(1).copied().fold(s[0], f64::max);

    // set singular values less than or equal to ``rcond * largest_singular_value`` to zero.
    let cutoff =
        rcond.unwrap_or(f64::EPSILON * max(x_faer.ncols(), x_faer.nrows()) as f64) * max_value;
    let s = s.map(|v| if v < &cutoff { 0. } else { *v });

    let binding = u.transpose() * y_faer;
    let d = &s / (&s * &s + alpha);

    if is_multi_target {
        let u_t_y = binding.as_ref().into_ndarray().to_owned();
        let d_ut_y_t = &d * &u_t_y.t();
        v.dot(&d_ut_y_t.t())
            .to_owned()
            .into_dimensionality::<I>()
            .expect("could not reshape output")
    } else {
        let u_t_y: Array1<f64> = binding
            .as_ref()
            .into_ndarray()
            .slice(s![.., 0])
            .into_owned();
        let d_ut_y = &d * &u_t_y;
        v.dot(&d_ut_y)
            .into_dimensionality::<I>()
            .expect("could not reshape output")
    }
} 

fn solve_ols_svd<D>(y: &Array<f64, D>, x: &Array2<f64>, rcond: Option<f64>) -> Array<f64, D>
where
    D: Dimension,
{
    // fallback SVD solver for platforms which don't (easily) support LAPACK solver
    solve_ridge_svd(y, x, 1.0e-64, rcond) // near zero ridge penalty
}

/// Solves least-squares regression using divide and conquer SVD. Thin wrapper to LAPACK: DGESLD.
#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), target_os = "macos"))]
#[allow(unused_variables)]
#[inline]
fn solve_ols_svd<D>(y: &Array<f64, D>, x: &Array2<f64>, rcond: Option<f64>) -> Array<f64, D>
where
    D: Dimension,
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>: LeastSquaresSvd<OwnedRepr<f64>, f64, D>,
{
    x.least_squares(y)
        .expect("Failed to compute LAPACK SVD solution!")
        .solution
}

/// Solves least-squares regression using QR decomposition w/ pivoting.
#[inline]
fn solve_ols_qr(y: &Array1<f64>, x: &Array2<f64>) -> Array1<f64> {
    // compute least squares solution via QR
    let x_faer = x.view().into_faer();
    let y_faer = y.slice(s![.., NewAxis]).into_faer();
    let coefficients = x_faer.col_piv_qr().solve_lstsq(&y_faer);
    coefficients
        .as_ref()
        .into_ndarray()
        .slice(s![.., 0])
        .to_owned()
}

/// Solves an ordinary least squares problem using either QR (faer) or LAPACK SVD
/// Inputs: features (2d ndarray), targets (1d ndarray), and an optional enum denoting solve method
/// Outputs: 1-d OLS coefficients
pub fn solve_ols(
    y: &Array1<f64>,
    x: &Array2<f64>,
    solve_method: Option<SolveMethod>,
    rcond: Option<f64>,
) -> Array1<f64> {
    let n_features = x.len_of(Axis(1));
    let n_samples = x.len_of(Axis(0));

    let solve_method = match solve_method {
        Some(SolveMethod::QR) => SolveMethod::QR,
        Some(SolveMethod::SVD) => SolveMethod::SVD,
        None => {
            // automatically determine recommended solution method based on shape of data
            if n_samples > n_features {
                SolveMethod::QR
            } else {
                SolveMethod::SVD
            }
        }
        _ => panic!("Only 'QR' and 'SVD' are currently supported solve methods for OLS."),
    };

    if solve_method == SolveMethod::QR {
        // compute least squares solution via QR
        solve_ols_qr(y, x)
    } else {
        solve_ols_svd(y, x, rcond)
    }
}


/// Solves least-squares regression using LU with partial pivoting
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

/// Solves the normal equations: (X^T X) coefficients = X^T Y
#[inline]
fn solve_normal_equations(
    xtx: &Array2<f64>,
    xty: &Array1<f64>,
    solve_method: Option<SolveMethod>,
    fallback_solve_method: Option<SolveMethod>,
) -> Array1<f64> {
    // Attempt to solve via Cholesky decomposition by default
    let solve_method = solve_method.unwrap_or(SolveMethod::Cholesky);

    match solve_method {
        SolveMethod::Cholesky => {
            let xtx_faer = xtx.view().into_faer();
            match xtx_faer.cholesky(Side::Lower) {
                Ok(cholesky) => {
                    // Cholesky decomposition successful
                    cholesky
                        .solve(&xty.slice(s![.., NewAxis]).into_faer())
                        .as_ref()
                        .into_ndarray()
                        .slice(s![.., 0])
                        .into_owned()
                }
                Err(_) => {
                    // Cholesky decomposition failed, fallback to SVD
                    let fallback_solve_method = fallback_solve_method.unwrap_or(SolveMethod::SVD);

                    match fallback_solve_method {
                        SolveMethod::SVD => {
                            #[cfg(debug_assertions)]
                            println!("Cholesky decomposition failed, falling back to SVD");
                            solve_ols_svd(xty, xtx, None)
                        }
                        SolveMethod::LU => {
                            #[cfg(debug_assertions)]
                            println!(
                                "Cholesky decomposition failed, \
                            falling back to LU w/ pivoting"
                            );
                            solve_ols_lu(xty, xtx)
                        }
                        SolveMethod::QR => {
                            #[cfg(debug_assertions)]
                            println!("Cholesky decomposition failed, falling back to QR");
                            solve_ols_qr(xty, xtx)
                        }
                        _ => panic!(
                            "unsupported fallback solve method: {:?}",
                            fallback_solve_method
                        ),
                    }
                }
            }
        }
        SolveMethod::LU => {
            // LU with partial pivoting
            solve_ols_lu(xty, xtx)
        }
        SolveMethod::QR | SolveMethod::SVD => solve_ols(xty, xtx, Some(solve_method), None),
        _ => panic!("Unsupported solve_method for solving normal equations!"),
    }
}

/// Solves a ridge regression problem of the form: ||y - x B|| + alpha * ||B||
/// Inputs: features (2d ndarray), targets (1d ndarray), ridge alpha scalar
#[inline]
pub fn solve_ridge(
    y: &Array1<f64>,
    x: &Array2<f64>,
    alpha: f64,
    solve_method: Option<SolveMethod>,
    rcond: Option<f64>,
) -> Array1<f64> {
    assert!(alpha >= 0., "alpha must be non-negative");
    match solve_method {
        Some(SolveMethod::Cholesky) | Some(SolveMethod::LU) | None => {
            let x_t = &x.t();
            let x_t_x = x_t.dot(x);
            let x_t_y = x_t.dot(y);
            let eye = Array::eye(x_t_x.shape()[0]);
            let ridge_matrix = &x_t_x + &eye * alpha;
            // use cholesky if specifically chosen, and otherwise LU.
            solve_normal_equations(
                &ridge_matrix,
                &x_t_y,
                solve_method,
                Some(SolveMethod::LU), // if cholesky fails fallback to LU
            )
        }
        Some(SolveMethod::SVD) => solve_ridge_svd(y, x, alpha, rcond),
        _ => panic!(
            "Only 'Cholesky', 'LU', & 'SVD' are currently supported solver \
        methods for Ridge."
        ),
    }
}

fn soft_threshold(x: &f64, alpha: f64, positive: bool) -> f64 {
    let mut result = x.signum() * (x.abs() - alpha).max(0.0);
    if positive {
        result = result.max(0.0);
    }
    result
}

/// Solves an elastic net regression problem of the form: 1 / (2 * n_samples) * ||y - Xw||_2
/// + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||_2.
///   Uses cyclic coordinate descent with efficient 'naive updates' and a
///   general soft thresholding function.
#[allow(clippy::too_many_arguments)]
pub fn solve_elastic_net(
    y: &Array1<f64>,
    x: &Array2<f64>,
    alpha: f64,            // strictly positive regularization parameter
    l1_ratio: Option<f64>, // scalar strictly between 0 (full ridge) and 1 (full lasso)
    max_iter: Option<usize>,
    tol: Option<f64>,       // controls convergence criteria between iterations
    positive: Option<bool>, // enforces non-negativity constraint
    solve_method: Option<SolveMethod>,
) -> Array1<f64> {
    let l1_ratio = l1_ratio.unwrap_or(0.5);
    let max_iter = max_iter.unwrap_or(1_000);
    let tol = tol.unwrap_or(0.00001);
    let positive = positive.unwrap_or(false);
    let solve_method = solve_method.unwrap_or(SolveMethod::CD);

    match solve_method {
        SolveMethod::CD | SolveMethod::CDActiveSet => {}
        _ => panic!(
            "Only solve_method 'CD' (coordinate descent) is currently supported \
        for Elastic Net / Lasso problems."
        ),
    }
    assert!(alpha > 0., "'alpha' must be strictly positive");
    assert!(
        (0.0..=1.).contains(&l1_ratio),
        "'l1_ratio' must be strictly between 0. and 1."
    );

    let (n_samples, n_features) = (x.shape()[0], x.shape()[1]);
    let mut w = Array1::<f64>::zeros(n_features);
    let xtx = x.t().dot(x);
    let mut residuals = y.to_owned(); // Initialize residuals
    let alpha = alpha * n_samples as f64;

    // Do cyclic coordinate descent
    if solve_method == SolveMethod::CD {
        for _ in 0..max_iter {
            let w_old = w.clone();
            for j in 0..n_features {
                let xj = x.slice(s![.., j]);
                // Naive update: add contribution of current feature to residuals
                residuals = &residuals + &xj * w[j];
                // Apply soft thresholding: compute updated weights
                w[j] = soft_threshold(&xj.dot(&residuals.view()), alpha * l1_ratio, positive)
                    / (xtx[[j, j]] + alpha * (1.0 - l1_ratio));
                // Naive update: subtract contribution of current feature from residuals
                residuals = &residuals - &xj * w[j];
            }

            if (&w - &w_old)
                .view()
                .insert_axis(Axis(0))
                .into_faer()
                .norm_l2()
                < tol
            {
                break;
            }
        }
    } else {
        // Do Coordinate Descent w/ Active Set
        // Initialize active set indices
        let mut active_indices: Vec<usize> = (0..n_features).collect();

        for _ in 0..max_iter {
            let w_old = w.clone();

            // // randomly shuffle a copy of the active set
            // let mut active_indices_shuffle = active_indices.clone();
            // let mut rng = rand::thread_rng();
            // active_indices_shuffle.shuffle(&mut rng);

            for j in active_indices.clone() {
                let xj = x.slice(s![.., j]);
                // Naive update: add contribution of current feature to residuals
                residuals = &residuals + &xj * w[j];

                // Apply soft thresholding: compute updated weights
                w[j] = soft_threshold(&xj.dot(&residuals.view()), alpha * l1_ratio, positive)
                    / (xtx[[j, j]] + alpha * (1.0 - l1_ratio));

                // Naive update: subtract contribution of current feature from residuals
                residuals = &residuals - &xj * w[j];

                // Check if weight for feature j has converged to zero and remove index from active set
                if w[j].abs() < tol {
                    if let Ok(pos) = active_indices.binary_search(&j) {
                        active_indices.remove(pos);
                    }
                }
            }

            if (&w - &w_old)
                .view()
                .insert_axis(Axis(0))
                .into_faer()
                .norm_l2()
                < tol
            {
                break;
            }
        }
    }

    w
}