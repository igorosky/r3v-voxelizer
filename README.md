## Parameters

Every voxelizer takes `resolution`, `intensity_multiplier`, `hit_multiplier`, `target_multiplier`, `destroy_threshold`, `construct_threshold`, `max_value` and `compress`, in this order (in Rust as a `VoxelizerParams` struct). `intensity_multiplier` scales the intensity of every point before it is truncated to an integer, so a point of intensity `i` adds `trunc(i * intensity_multiplier) * target_multiplier` to the voxel it ends in and subtracts `trunc(i * intensity_multiplier) * hit_multiplier` from every voxel its ray passes through (negative products count as 0). Pass `1.0` to leave intensities as they are. `max_value` caps voxel values: a voxel seen many times saturates at it instead of growing without bound, so as many hits as `max_value` (divided by `hit_multiplier`) always clear it again. Pass `4294967295` (`u32::MAX`) for no cap. With `compress` a voxel gets constructed when it reaches `construct_threshold` and destroyed when, being constructed, it drops to `destroy_threshold` or below. Between the thresholds a voxel keeps its state, so one that never reached `construct_threshold` is never destroyed. With `compress` a `max_value` below `construct_threshold` means no voxel ever gets constructed.

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
const voxelizer = await WgpuAccelerated.create(resolution, intensityMultiplier, hitMultiplier,
    targetMultiplier, destroyThreshold, constructThreshold, maxValue, compress, bucketCount);
const update = await voxelizer.add_points(sensor, points);
```
`create` rejects when there is no WebGPU adapter capable of running compute shaders, `bucketCount` is 0 or the hashmap does not fit the adapter's limits (see C/C++ above for `bucketCount`, `WgpuAccelerated.memory_size(bucketCount)` gives the GPU memory it takes). Await each `add_points` call before starting the next one: a call made while another one is running rejects.

The JS tests of the `wgpu` feature run on Node.js, which has no WebGPU, so they use the `webgpu` package (Dawn). Install it with `npm ci` in `tests/js`.

## Benchmarks

`./bench.sh` runs every implementation (`SingleThreaded`, `MultiThreaded` at several thread counts and `WgpuAccelerated`) on the same generated workloads and writes the results to `target/bench/<date>/`: `results.csv`, `run.log` and, when docker is available, plots of the CSV (`benches/plot.py`). `./bench.sh --quick` runs a smaller sweep, `./bench.sh --help` lists the options (implementations, thread counts, points per frame, ranges, `compress`, frame counts, ...) and `--no-plot` skips the plots.

Two scenarios are generated: `room`, a sensor circling inside a box room with rays in every direction, and `cone`, a static sensor whose rays all end on a small patch of a wall (many rays hitting the same voxels). The `match` column tells whether an implementation returned the same voxels as the first one run on the workload. A run that panics is reported as such and the sweep goes on.

`FEATURES` picks the implementations built (default `single_threaded,multi_threaded,wgpu`), e.g. `FEATURES=single_threaded,multi_threaded ./bench.sh` on a machine without a GPU. The benchmark itself is `benches/voxelizer.rs`, also runnable with `cargo bench --features multi_threaded,wgpu --bench voxelizer -- <options>`.
