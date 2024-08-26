use crate::least_squares::lstsq_solver1;
use faer_ext::IntoFaer;
use js_sys::Float64Array;
use ndarray::{Array2, Ix2};
use serde::Serialize;
use serde_wasm_bindgen::to_value;
use wasm_bindgen::prelude::*;

pub mod least_squares;


#[wasm_bindgen]
pub fn ols_coefficients(x_data: Float64Array, y_data: Float64Array, n: usize, m: usize) -> JsValue {
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

    // Convert the Array2<f64> to Vec<Vec<f64>> for serialization
    let serializable_result: Vec<Vec<f64>> = coefficients.outer_iter().map(|row| row.to_vec()).collect();

    // Serialize the result to JsValue using serde_wasm_bindgen
    to_value(&serializable_result).unwrap()
}
