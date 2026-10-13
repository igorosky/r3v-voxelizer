## Parameters

Every voxelizer takes `resolution`, `hit_multiplier`, `target_multiplier`, `destroy_threshold`, `construct_threshold`, `max_value` and `compress`, in this order (in Rust as a `VoxelizerParams` struct). `max_value` caps voxel values: a voxel seen many times saturates at it instead of growing without bound, so as many hits as `max_value` (divided by `hit_multiplier`) always clear it again. Pass `4294967295` (`u32::MAX`) for no cap. With `compress` a `max_value` below `construct_threshold` means no voxel ever gets constructed.

## C/C++

To compile and run the C/C++ code, you can use the following commands in your terminal:
```bash
cargo build --release --features c     # For C
cargo build --release --features cpp   # For C++
```

The compiled `.so` files will be located in the `target/release` directory.
The headers for the C/C++ code can be found in the `bindings` directory.

The GPU voxelizer needs the `wgpu` feature as well (e.g. `--features c,wgpu`). `create_wgpu_voxelizer` takes the usual arguments plus `bucket_count`, the size of its hashmap: each bucket holds 2 pages of 4x4x4 voxels and takes about 1 KiB of GPU memory (`wgpu_memory_size(bucket_count)` gives the exact size in bytes). It returns NULL when there is no GPU adapter capable of running compute shaders, `bucket_count` is 0 or the hashmap does not fit the adapter's limits. Without the `wgpu` feature it always returns NULL.

## Python

To run the Python code, you can use the following command in your terminal:
```bash
cargo build --release --features python
```

Then you can use the generated `.so` file in your Python code by importing it as a module.

With `--features python,wgpu` the module also has `WgpuAccelerated`, the GPU voxelizer. Its constructor takes the usual arguments plus `bucket_count` (see C/C++ above, `WgpuAccelerated.memory_size(bucket_count)` gives the GPU memory it takes) and raises `RuntimeError` when no GPU voxelizer can be created.

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

With `--features=js,wgpu` the module also has `WgpuAccelerated`, the GPU voxelizer running on WebGPU. Talking to the GPU is asynchronous on the web, so it has no constructor:
```js
const voxelizer = await WgpuAccelerated.create(resolution, hitMultiplier, targetMultiplier,
    destroyThreshold, constructThreshold, maxValue, compress, bucketCount);
const update = await voxelizer.add_points(sensor, points);
```
`create` rejects when there is no WebGPU adapter capable of running compute shaders, `bucketCount` is 0 or the hashmap does not fit the adapter's limits (see C/C++ above for `bucketCount`, `WgpuAccelerated.memory_size(bucketCount)` gives the GPU memory it takes). Await each `add_points` call before starting the next one: a call made while another one is running rejects.

The JS tests of the `wgpu` feature run on Node.js, which has no WebGPU, so they use the `webgpu` package (Dawn). Install it with `npm ci` in `tests/js`.
