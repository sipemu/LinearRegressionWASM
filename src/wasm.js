async function ols_coefficients_(x_data, y_data, n, m) {
  const wasm = await import("./pkg/wasm_test_bg.wasm");
  const { __wbg_set_wasm, ols_coefficients } = await import("./pkg/wasm_test_bg.js");
  __wbg_set_wasm(wasm);

  // Convert x_data and y_data to Float64Array if they aren't already
  const xDataArray = new Float64Array(x_data);
  const yDataArray = new Float64Array(y_data);

  // Call the Rust function and return the result
  const result = ols_coefficients(xDataArray, yDataArray, n, m);
  return result;
}


globalThis.ols_coefficients_ = ols_coefficients_;
