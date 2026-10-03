## C/C++

To compile and run the C/C++ code, you can use the following commands in your terminal:
```bash
cargo build --release --features c     # For C
cargo build --release --features cpp   # For C++
```

The compiled `.so` files will be located in the `target/release` directory.
The headers for the C/C++ code can be found in the `bindings` directory.

## Python

To run the Python code, you can use the following command in your terminal:
```bash
cargo build --release --features python
```

Then you can use the generated `.so` file in your Python code by importing it as a module.

Currently there is no stubs for the Python code.

## JavaScript

To build the JavaScript code it is recommended to use `wasm-pack`. You can install it using the following command:
```bash
cargo install wasm-pack
```

then you can build the JavaScript code using the following command:
```bash
wasm-pack build --release --target web --features=js
```

Integer arguments (the voxelizer constructor parameters) and the `PointI` fields are converted the way wasm-bindgen converts every integer: out of range or fractional values are wrapped and truncated instead of rejected. For example `-1` passed as an unsigned value becomes `4294967295` and `1.9` becomes `1`.

Multi-threaded voxelizer is not supported in the JavaScript build.
