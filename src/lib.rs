use crate::least_squares::*;
use faer_ext::IntoFaer;
use js_sys::Float64Array;
use ndarray::{Array1, Array2};
use serde::Serialize;
use serde_wasm_bindgen::to_value;
use wasm_bindgen::prelude::*;
use statrs::distribution::{StudentsT, Continuous};

pub mod least_squares;

#[derive(Serialize)]
struct RegressionResults {
    coefficients: Vec<f64>,
    standard_errors: Vec<f64>,
    t_values: Vec<f64>,
    p_values: Vec<f64>,
    residual_std_error: f64,
    residual_df: usize,
    r_squared: f64,
    adj_r_squared: f64,
    f_statistic: f64,
    f_df1: usize,
    f_df2: usize,
    f_p_value: f64,
}

#[wasm_bindgen]
pub fn perform_ols_regression(
    y_data: Float64Array,
    x_data: Float64Array,
    n: usize,
    m: usize,
) -> JsValue {
    // Convert the input Float64Array to Rust Array1 and Array2
    let y_vec: Vec<f64> = y_data.to_vec();
    let x_vec: Vec<f64> = x_data.to_vec();

    let y: Array1<f64> = Array1::from(y_vec);
    let x: Array2<f64> = Array2::from_shape_vec((n, m), x_vec).unwrap();

    // Convert the string to SolveMethod enum if provided
    let solve_method = Some(SolveMethod::QR);

    // Call the OLS solver
    let (coefficients, standard_errors, t_values, residuals) = solve_ols(
        &y,
        &x,
        solve_method,
        None, // Assuming rcond is not needed, remove if it is
    );

    // Calculate p-values
    let degrees_of_freedom = n - m;
    let t_distribution = StudentsT::new(0.0, 1.0, degrees_of_freedom as f64).unwrap();
    let p_values: Vec<f64> = t_values.iter()
        .map(|&t| 2.0 * (1.0 - t_distribution.cdf(t.abs())))
        .collect();

    // Calculate additional statistics
    let (
        residual_std_error,
        residual_df,
        r_squared,
        adj_r_squared,
        f_statistic,
        f_df1,
        f_df2,
        f_p_value,
    ) = calculate_ols_statistics(&y, &x, &coefficients, &residuals);

    let result = RegressionResults {
        coefficients: coefficients.into_raw_vec(),
        standard_errors: standard_errors.into_raw_vec(),
        t_values: t_values.into_raw_vec(),
        p_values,
        residual_std_error,
        residual_df,
        r_squared,
        adj_r_squared,
        f_statistic,
        f_df1,
        f_df2,
        f_p_value,
    };

    // Serialize the result to JsValue using serde_wasm_bindgen
    to_value(&result).unwrap()
}


#[wasm_bindgen]
pub fn perform_ols_prediction(
    x_data: Float64Array,
    x_test_data: Float64Array,
    y_data: Float64Array,
    n: usize,
    m: usize
) -> JsValue {
    // Convert JavaScript arrays to Rust ndarray::Array2<f64>
    let x_vec: Vec<f64> = x_data.to_vec();
    let x_test_vec: Vec<f64> = x_test_data.to_vec();
    let y_vec: Vec<f64> = y_data.to_vec();

    let x: Array2<f64> = Array2::from_shape_vec((n, m), x_vec).unwrap();
    let x_test: Array2<f64> = Array2::from_shape_vec((n, m), x_test_vec).unwrap();
    let y: Array2<f64> = Array2::from_shape_vec((n, 1), y_vec).unwrap();

    // Create MatRef views from the input matrices
    let x_ref = x.view().into_faer();
    let y_ref = y.view().into_faer();

    // Call the least squares solver
    let coefficients = lstsq_solver1(x_ref, y_ref);

    // Convert the MatRef back to ndarray to perform dot product
    let coeff_array = Array1::from(coefficients.into_raw_vec()); // Assuming coefficients is a column vector

    // Compute predictions by multiplying x_test with coefficients
    let out_of_sample_predictions = x_test.dot(&coeff_array);
    let in_sample_predictions = x.dot(&coeff_array);

    // Convert the Array2<f64> to Vec<Vec<f64>> for serialization
    let result = OLSPredictions{
        in_sample_predictions: in_sample_predictions.into_raw_vec(),
        out_of_sample_predictions: out_of_sample_predictions.into_raw_vec()
    };

    // Serialize the result to JsValue using serde_wasm_bindgen
    to_value(&result).unwrap()
}


#[wasm_bindgen]
pub fn perform_elastic_net_regression(
    y_data: Float64Array, 
    x_data: Float64Array, 
    alpha: f64,            
    l1_ratio: Option<f64>, 
    max_iter: Option<usize>,
    tol: Option<f64>,       
    positive: Option<bool>, 
    solve_method_str: Option<String>,
    n: usize,
    m: usize
) -> JsValue {
    // Convert the input Float64Array to Rust Array1 and Array2
    let y_vec: Vec<f64> = y_data.to_vec();
    let x_vec: Vec<f64> = x_data.to_vec();

    let y: Array1<f64> = Array1::from(y_vec);
    let x: Array2<f64> = Array2::from_shape_vec((n, m), x_vec).unwrap();

    // Convert the string to SolveMethod enum
    let solve_method = solve_method_str.map(|s| SolveMethod::from_str(s.as_str()).expect("invalid solve_method detected!"));

    // Call the elastic net solver
    let coefficients = solve_elastic_net(
        &y,
        &x,
        alpha,
        l1_ratio,
        max_iter,
        tol,
        positive,
        solve_method,
    );

    // Calculate residuals
    let residuals = &y - x.dot(&coefficients);

    // Calculate standard errors (this is an approximation for Elastic Net)
    let (standard_errors, t_values) = calculate_elastic_net_stats(&x, &coefficients, &residuals);

    // Calculate p-values
    let degrees_of_freedom = n - coefficients.len();
    let t_distribution = StudentsT::new(0.0, 1.0, degrees_of_freedom as f64).unwrap();
    let p_values: Vec<f64> = t_values.iter()
        .map(|&t| 2.0 * (1.0 - t_distribution.cdf(t.abs())))
        .collect();

    // Calculate additional statistics
    let (
        residual_std_error,
        residual_df,
        r_squared,
        adj_r_squared,
        f_statistic,
        f_df1,
        f_df2,
        f_p_value,
    ) = calculate_regression_statistics(&y, &x, &coefficients, &residuals);

    let result = RegressionResults {
        coefficients: coefficients.to_vec(),
        standard_errors: standard_errors.to_vec(),
        t_values: t_values.to_vec(),
        p_values,
        residual_std_error,
        residual_df,
        r_squared,
        adj_r_squared,
        f_statistic,
        f_df1,
        f_df2,
        f_p_value,
    };

    // Serialize the result to JsValue using serde_wasm_bindgen
    to_value(&result).unwrap()
}
