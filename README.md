## Parameters

Every voxelizer takes `resolution`, `intensity_multiplier`, `hit_multiplier`, `target_multiplier`, `destroy_threshold`, `construct_threshold`, `max_value` and `compress`, in this order (in Rust as a `VoxelizerParams` struct, holding the three multipliers as a `Multipliers` struct of `intensity`, `hit` and `target`). `intensity_multiplier` scales the intensity of every point before it is truncated to an integer, so a point of intensity `i` adds `trunc(i * intensity_multiplier) * target_multiplier` to the voxel it ends in and subtracts `trunc(i * intensity_multiplier) * hit_multiplier` from every voxel its ray passes through (negative products count as 0). Pass `1.0` to leave intensities as they are. `max_value` caps voxel values: a voxel seen many times saturates at it instead of growing without bound, so as many hits as `max_value` (divided by `hit_multiplier`) always clear it again. Pass `4294967295` (`u32::MAX`) for no cap. With `compress` a voxel gets constructed when it reaches `construct_threshold` and destroyed when, being constructed, it drops to `destroy_threshold` or below. Between the thresholds a voxel keeps its state, so one that never reached `construct_threshold` is never destroyed. With `compress` a `max_value` below `construct_threshold` means no voxel ever gets constructed.

The multipliers can be changed after a voxelizer is created: `set_multipliers` takes a `Multipliers` (`intensity`, `hit`, `target`) used by the following `add_points` calls, voxels keep the values they got so far. `get_multipliers` returns a copy of the current ones (`multipliers()` in Rust, both are methods of the `Voxelizer` trait there). In C/C++ they are `set_multipliers(voxelizer, multipliers)` and `get_multipliers(voxelizer)`, the latter returns all 0 for NULL. In Python and JS `Multipliers` is a class like `PointF` and both are methods of every voxelizer class. The JS `WgpuAccelerated` throws from both while another call (`add_points` or a dump) is running.

The whole state of a voxelizer can be dumped. `get_all_points` returns every voxel that is not 0 with its value, as `PointI`s (also with `compress`). `get_all_points_compressed` returns every constructed voxel, as groups of 64x64x64 voxels holding at least one: the group key (voxel coordinates shifted right by 6, `i` is 0) and 4096 (`POINTS_IN_VOXEL_GROUP`) 64 bit words, one per 4x4x4 fine page in z-order (bits of the page offset in the group interleaved x, y, z from the lowest one). Bit `x + 4 * y + 16 * z` of a word is set when voxel (x, y, z) of its page is constructed. Without `compress` a voxel counts as constructed once it reaches `construct_threshold`. Both are methods of the `Voxelizer` trait in Rust and of every voxelizer class in Python and JS, where a group is a `CompressedVoxelGroup` (`point` and `values`, methods in Python, properties in JS where `values` is a `BigUint64Array`). In C/C++ `get_all_points(voxelizer)` returns a `VoxelsUpdate` to free with `drop_update` and `get_all_points_compressed(voxelizer)` returns `VoxelGroups`, an array of `VoxelGroup` (`point` and `values[POINTS_IN_VOXEL_GROUP]`) to free with `drop_voxel_groups`, both empty for NULL. The JS `WgpuAccelerated` returns Promises from both, which reject while another call is running.

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

Type stubs for type checkers and IDEs are in `bindings/r3v_voxelizer.pyi`, put the file next to the module's `.so` file. They describe the module built with every voxelizer feature, the classes of features left out are missing at runtime.

## JavaScript

To build the JavaScript code it is recommended to use `wasm-pack`. You can install it using the following command:
```bash
cargo install wasm-pack
```

then you can build the JavaScript code using the following command:
```bash
wasm-pack build --release --target web --features=js
```

The build also generates the TypeScript types (`r3v_voxelizer.d.ts`).

Integer arguments (the voxelizer constructor parameters) and the `PointI` fields are converted the way wasm-bindgen converts every integer: out of range or fractional values are wrapped and truncated instead of rejected. For example `-1` passed as an unsigned value becomes `4294967295` and `1.9` becomes `1`.

Multi-threaded voxelizer is not supported in the JavaScript build.

With `--features=js,wgpu` the module also has `WgpuAccelerated`, the GPU voxelizer running on WebGPU. Talking to the GPU is asynchronous on the web, so it has no constructor:
```js
const voxelizer = await WgpuAccelerated.create(resolution, intensityMultiplier, hitMultiplier,
    targetMultiplier, destroyThreshold, constructThreshold, maxValue, compress, bucketCount);
const update = await voxelizer.add_points(sensor, points);
```
`create` rejects when there is no WebGPU adapter capable of running compute shaders, `bucketCount` is 0 or the hashmap does not fit the adapter's limits (see C/C++ above for `bucketCount`, `WgpuAccelerated.memory_size(bucketCount)` gives the GPU memory it takes). `get_all_points` and `get_all_points_compressed` return Promises as well. Await each of these calls before starting the next one: a call made while another one is running rejects.

The JS tests of the `wgpu` feature run on Node.js, which has no WebGPU, so they use the `webgpu` package (Dawn). Install it with `npm ci` in `tests/js`.

## Benchmarks

`./bench.sh` runs every implementation (`SingleThreaded`, `MultiThreaded` at several thread counts and `WgpuAccelerated`) on the same generated workloads and writes the results to `target/bench/<date>/`: `results.csv`, `run.log` and, when docker is available, plots of the CSV (`benches/plot.py`). `./bench.sh --quick` runs a smaller sweep, `./bench.sh --help` lists the options (implementations, thread counts, points per frame, ranges, `compress`, frame counts, ...) and `--no-plot` skips the plots.

Two scenarios are generated: `room`, a sensor circling inside a box room with rays in every direction, and `cone`, a static sensor whose rays all end on a small patch of a wall (many rays hitting the same voxels). The `match` column tells whether an implementation returned the same voxels as the first one run on the workload. A run that panics is reported as such and the sweep goes on.

`FEATURES` picks the implementations built (default `single_threaded,multi_threaded,wgpu`), e.g. `FEATURES=single_threaded,multi_threaded ./bench.sh` on a machine without a GPU. The benchmark itself is `benches/voxelizer.rs`, also runnable with `cargo bench --features multi_threaded,wgpu --bench voxelizer -- <options>`.

### Quick check while optimizing

`./bench.sh --check` runs a fixed set of four cases (`compress` on) on every implementation for about a minute and compares the run with a saved baseline and with the previous run. Save the baseline before changing anything, then run the check after every change:

```bash
./bench.sh --check --save-baseline   # on the code you start from
./bench.sh --check                   # after a change
./bench.sh --check --impls st        # only the implementation you work on
```

Per case it prints the median `ms/frame` over the reps, `±` (median absolute deviation of the reps), and the change against the baseline and the previous run: `faster` or `slower` only when the change is larger than `--threshold` (5% by default) and twice the `±` of both runs, `~` otherwise. `output` is `CHANGED` when the voxels returned differ from the baseline, which an optimization should never do. A summary line per implementation gives the geometric mean over the cases.

Runs are written to `target/bench/check/runs/`, the baseline to `target/bench/check/baseline.csv`. Compare runs of the same machine and the same `--impls` only. On laptops the numbers depend a lot on the power profile and on other programs: use a performance profile and close heavy programs. On CPUs with performance and efficiency cores, `st` and the host side of `wgpu` are pinned to one performance core (`--pin` changes or disables it), `mt` runs on every CPU. `wgpu` timings are noisier than the CPU ones, its bands are wider. `./bench.sh --check --help` lists the options.

The check is built with the `bench-check` profile (`Cargo.toml`): release with `debug = "line-tables-only"`, so a profiler such as `perf` maps samples to source lines. The binary is in `target/bench-check/deps/check-*`, apart from the other benches.

### Flamegraphs

`--flamegraph` runs the benchmark (with `--check` the check) through [`cargo flamegraph`](https://github.com/flamegraph-rs/flamegraph) (`cargo install flamegraph`, needs `perf`):

```bash
./bench.sh --check --flamegraph --impls st
```

It builds the `bench-check` profile with `RUSTFLAGS="-C force-frame-pointers=yes"` in `target/flamegraph/`, so the other builds are kept, and perf walks the call stacks through the frame pointers. The flamegraph is written to `target/bench/flamegraph/<date>-<benchmark>.svg`. Next to it, `<date>-<benchmark>.txt` lists every function sorted by CPU time, and the top of the list is printed:

```
12.138 s of CPU time, summed over every thread

            total              self  function
 94.67%   11.491 s  18.29%    2.221 s  <SingleThreaded as Voxelizer>::add_points
 73.05%    8.867 s  66.79%    8.107 s  <SingleThreaded>::hit_voxel
  8.98%    1.090 s   8.98%    1.090 s  bmi2::index_of_unchecked::<u16, 3>
```

`total` is the time with the functions called (counted once per stack, so recursion does not count twice), `self` the time in the function itself, both summed over every thread. Inlined functions are listed on their own. Names are shortened to the type, or the last module of a free function, generic arguments stay. `benches/hotspots.awk` makes the table from the folded stacks of the latest run, `target/flamegraph/stacks.folded`; the perf data of the latest run is `target/flamegraph/perf.data` (`perf report -i target/flamegraph/perf.data`). Frame pointers and perf slow the run down a little, so a profiled check run is not saved (`--no-save`) and its comparisons are only a hint. Profile one implementation at a time. perf samples every running thread at 997 Hz, and cargo flamegraph keeps the whole `perf script` output in memory (a few KB per sample), so with `mt` the frequency is divided by its thread count (`--threads`, every CPU by default): about 997 samples a second whatever the implementation. perf needs `kernel.perf_event_paranoid` at 2 or lower (Ubuntu sets 4): `sudo sysctl kernel.perf_event_paranoid=2`.
