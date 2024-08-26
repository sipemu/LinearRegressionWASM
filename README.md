# Rust WASM in Google Appsscripts

- Original Repository [https://github.com/googleworkspace/apps-script-samples/tree/main/wasm/hello-world](https://github.com/googleworkspace/apps-script-samples/tree/main/wasm/hello-world)
- Blog Post [https://justin.poehnelt.com/posts/apps-script-wasm/](https://justin.poehnelt.com/posts/apps-script-wasm/)
- Additional resource: [Rust to WebAssembly the hard way](https://surma.dev/things/rust-to-webassembly/)

## Tests



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
