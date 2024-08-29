use crate::least_squares::*;
use std::str::FromStr;
use faer_ext::IntoFaer;
use js_sys::Float64Array;
#[warn(unused_imports)]
use ndarray::{Array1, Array2, Ix2};
use serde::Serialize;
use serde_wasm_bindgen::to_value;
use wasm_bindgen::prelude::*;

pub mod least_squares;

#[derive(Serialize)]
struct OLSResults {
    coefficients: Vec<f64>,
}


#[wasm_bindgen]
pub fn get_ols_coefficients(x_data: Float64Array, y_data: Float64Array, n: usize, m: usize) -> JsValue {
    // Convert JavaScript arrays to Rust ndarray::Array2<f64>
    let x_vec: Vec<f64> = x_data.to_vec();
    let y_vec: Vec<f64> = y_data.to_vec();

    let x: Array2<f64> = Array2::from_shape_vec((n, m), x_vec).unwrap();
    let y: Array2<f64> = Array2::from_shape_vec((n, 1), y_vec).unwrap();

    // Create MatRef views from the input matrices
    let x_ref = x.view().into_faer();
    let y_ref = y.view().into_faer();

    // Call the least squares solver
    let coefficients = lstsq_solver1(x_ref, y_ref);

    let result = OLSResults{
        coefficients: coefficients.into_raw_vec()
    };

    // Serialize the result to JsValue using serde_wasm_bindgen
    to_value(&result).unwrap()
}

#[derive(Serialize)]
struct OLSPredictions {
    in_sample_predictions: Vec<f64>,
    out_of_sample_predictions: Vec<f64>
}

#[wasm_bindgen]
pub fn get_ols_predictions(x_data: Float64Array, x_test_data: Float64Array, y_data: Float64Array, n: usize, m: usize) -> JsValue {
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
pub fn get_elastic_net_coefficients(
    y_data: Float64Array, 
    x_data: Float64Array, 
    alpha: f64,            
    l1_ratio: Option<f64>, 
    max_iter: Option<usize>,
    tol: Option<f64>,       
    positive: Option<bool>, 
    solve_method_str: Option<String>, // Change to accept a string
    n: usize,
    m: usize
) -> Result<JsValue, JsValue> {
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

    // Serialize the result to JsValue using serde_wasm_bindgen
    let serialized = to_value(&coefficients.to_vec()).map_err(|_| {
        JsValue::from_str("Failed to serialize coefficients")
    })?;

    Ok(serialized)
}