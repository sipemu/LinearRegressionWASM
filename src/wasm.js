async function get_ols_coefficients_(x_data, y_data, n, m, solve_method_str) {
  const wasm = await import("./pkg/wasm_regression_bg.wasm");
  const { __wbg_set_wasm, get_ols_coefficients } = await import("./pkg/wasm_regression_bg.js");
  __wbg_set_wasm(wasm);

  // Convert x_data and y_data to Float64Array if they aren't already
  const xDataArray = new Float64Array(x_data);
  const yDataArray = new Float64Array(y_data);

  // Call the Rust function and return the result
  const result = get_ols_coefficients(xDataArray, yDataArray, n, m);
  return result;
}


async function get_ols_prediction_(x_data, y_data, n, m) {
  const wasm = await import("./pkg/wasm_regression_bg.wasm");
  const { __wbg_set_wasm, get_ols_coefficients } = await import("./pkg/wasm_regression_bg.js");
  __wbg_set_wasm(wasm);

  // Convert x_data and y_data to Float64Array if they aren't already
  const xDataArray = new Float64Array(x_data);
  const yDataArray = new Float64Array(y_data);

  // Call the Rust function and return the result
  const result = get_ols_coefficients(xDataArray, yDataArray, n, m);
  return result;
}


async function get_elastic_net_coefficients_(x_data, y_data, n, m) {
  const wasm = await import("./pkg/wasm_regression_bg.wasm");
  const { __wbg_set_wasm, get_elastic_net_coefficients } = await import("./pkg/wasm_regression_bg.js");
  __wbg_set_wasm(wasm);

  // Convert x_data and y_data to Float64Array if they aren't already
  const xDataArray = new Float64Array(x_data);
  const yDataArray = new Float64Array(y_data);

  // Call the Rust function and return the result
  const result = get_elastic_net_coefficients(xDataArray, yDataArray, n, m);
  return result;
}


globalThis.get_ols_coefficients_ = get_ols_coefficients_;
