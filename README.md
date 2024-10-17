# Rust WASM in Google Appsscripts

This project demonstrates how to use Rust-generated WebAssembly (WASM) in Google Apps Script. It provides a simple linear regression implementation that can be used within Google Sheets or other Google Workspace applications.

## What This Module Does

This module implements a basic linear regression algorithm using Rust, which is then compiled to WebAssembly for use in Google Apps Script. The main features include:

1. **Linear Regression Calculation**: The Rust code provides functions to calculate the slope and intercept of a linear regression line based on input data points.

2. **WASM Integration**: The Rust code is compiled to WebAssembly, allowing it to be used efficiently within the JavaScript environment of Google Apps Script.

3. **Google Sheets Integration**: The module can be used to perform linear regression calculations directly within Google Sheets, enabling users to analyze data and make predictions based on trends.

4. **Performance Boost**: By using Rust and WebAssembly, the module offers improved performance for computationally intensive tasks compared to pure JavaScript implementations.

## Resources

- Original Repository: [https://github.com/googleworkspace/apps-script-samples/tree/main/wasm/hello-world](https://github.com/googleworkspace/apps-script-samples/tree/main/wasm/hello-world)
- Blog Post: [https://justin.poehnelt.com/posts/apps-script-wasm/](https://justin.poehnelt.com/posts/apps-script-wasm/)
- Additional resource: [Rust to WebAssembly the hard way](https://surma.dev/things/rust-to-webassembly/)

## How to build:

0. Initialize project with `npm install`

1. You will need to use cargo to build your Rust code.
```
cargo build --target wasm32-unknown-unknown --release
```

2. You will need to use wasm-bindgen to generate the JavaScript bindings for your Rust code.
```
wasm-bindgen \
  --out-dir src/pkg \
  --target bundler \
  ./target/wasm32-unknown-unknown/release/wasm_regression.wasm
```

3. You will need to use wasm-opt to optimize your WebAssembly module. This is optional but recommended.
```
wasm-opt \
  src/pkg/wasm_regression_bg.wasm \
  -Oz \
  -o src/pkg/wasm_regression_bg.wasm
```

4. You will need to use a bundler such as ESBuild to bundle your JavaScript code and WebAssembly module.
```
node build.js
```

## Deployment

Add `src/main.js` and `dist/wasm.js` to your Appsscripts project.


## Additional Resources

Added following options to `Cargo.toml` according to [Rust to WebAssembly the hard way](https://surma.dev/things/rust-to-webassembly/):
```
[profile.release]
opt-level = 's'
strip = "debuginfo"
lto = true
```
