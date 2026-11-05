# Bindings

[Back to README](../README.md)

The C, C++, Python and JavaScript bindings expose the same voxelizers as Rust.
Parameters and outputs are described in [parameters.md](parameters.md).

Only one binding fits in a build: `c`/`cpp`, `python` and `js` cannot be
combined. Add `multi_threaded` and `wgpu` for the other implementations
(`single_threaded` is on by default).

## C and C++

```bash
cargo build --release --features c     # C
cargo build --release --features cpp   # C++
cargo build --release --features c,wgpu,multi_threaded   # with every implementation
```

The library is `target/release/libr3v_voxelizer.so`, the headers are in
`bindings/` (`r3v-voxelizer.h`, and `r3v-voxelizer.hpp` with the namespace
`r3v::voxelizer`). Link with `-L target/release -lr3v_voxelizer`.

| Function | Notes |
|---|---|
| `create_single_threaded_voxelizer(...)` | The usual parameters |
| `create_multi_threaded_voxelizer(..., thread_count)` | |
| `create_wgpu_voxelizer(..., bucket_count)` | NULL when it cannot be created, see below |
| `add_points(voxelizer, sensor, points, count)` | Returns a `VoxelsUpdate` (`data`, `len`) |
| `get_all_points(voxelizer)` | Returns a `VoxelsUpdate` |
| `get_all_points_compressed(voxelizer)` | Returns `VoxelGroups` |
| `set_multipliers(voxelizer, multipliers)`, `get_multipliers(voxelizer)` | |
| `drop_update`, `drop_voxel_groups`, `drop_voxelizer` | Free what the functions above returned |

Every function accepts NULL and does nothing (or returns an empty result) for
it.

### GPU voxelizer

The GPU voxelizer needs the `wgpu` feature. `create_wgpu_voxelizer` takes the
usual arguments plus `bucket_count`, the size of its hashmap. Each bucket holds
2 pages of 4x4x4 voxels and takes about 1 KiB of GPU memory;
`wgpu_memory_size(bucket_count)` gives the exact size in bytes.

It returns NULL when:

- there is no GPU adapter that can run compute shaders,
- `bucket_count` is 0,
- the hashmap does not fit the adapter's limits.

Without the `wgpu` feature it always returns NULL.

## Python

```bash
cargo build --release --features python            # add ,wgpu or ,multi_threaded as needed
cp target/release/libr3v_voxelizer.so r3v_voxelizer.so
```

Put `r3v_voxelizer.so` where Python finds it and `import r3v_voxelizer`.
Constructor arguments can be passed by name.

```python
import r3v_voxelizer as rv

voxelizer = rv.MultiThreaded(0.1, 1.0, 1, 2, 0, 2, 10, True, thread_count=4)
update = voxelizer.add_points(rv.PointF(0, 0, 0, 1), [rv.PointF(1.0, 0.5, 0.2, 1.0)])
groups = voxelizer.get_all_points_compressed()
voxelizer.set_multipliers(rv.Multipliers(2.0, 1, 3))
```

`PointI` and `CompressedVoxelGroup` fields are read with methods (`p.x()`,
`group.values()`).

With `--features python,wgpu` the module also has `WgpuAccelerated`. Its
constructor takes the usual arguments plus `bucket_count` (see
[GPU voxelizer](#gpu-voxelizer)) and raises `RuntimeError` when no GPU
voxelizer can be created. `WgpuAccelerated.memory_size(bucket_count)` gives
the GPU memory it takes.

Type stubs for type checkers and IDEs are in `bindings/r3v_voxelizer.pyi`. Put
the file next to the module's `.so`. They describe the module built with every
voxelizer feature, so classes of features left out are missing at runtime.
`help(r3v_voxelizer)` shows the documentation of the module itself.

## JavaScript (WebAssembly)

Install `wasm-pack` and the wasm target once:

```bash
cargo install wasm-pack
rustup target add wasm32-unknown-unknown
```

Then build:

```bash
wasm-pack build --release --target web --features js        # CPU only
wasm-pack build --release --target web --features js,wgpu   # with WebGPU
```

`pkg/` gets the module and its TypeScript types (`r3v_voxelizer.d.ts`).
`MultiThreaded` is not available in the JavaScript build.

```js
import init, * as rv from "./pkg/r3v_voxelizer.js";

await init();
const voxelizer = new rv.SingleThreaded(0.1, 1.0, 1, 2, 0, 2, 10, true);
const update = voxelizer.add_points(new rv.PointF(0, 0, 0, 1), [{ x: 1, y: 0.5, z: 0.2, i: 1 }]);
voxelizer.free();
```

Points can be `PointF` objects or plain `{ x, y, z, i }` objects.

### Integer arguments

Integer arguments (the constructor parameters) and the `PointI` fields are
converted the way wasm-bindgen converts every integer: values out of range or
with a fraction are wrapped and truncated, not rejected. For example `-1`
passed as an unsigned value becomes `4294967295` and `1.9` becomes `1`.

### GPU voxelizer on WebGPU

With `--features js,wgpu` the module also has `WgpuAccelerated`, running on
WebGPU. Talking to the GPU is asynchronous on the web, so it has no
constructor:

```js
const voxelizer = await rv.WgpuAccelerated.create(resolution, intensityMultiplier, hitMultiplier,
    targetMultiplier, destroyThreshold, constructThreshold, maxValue, compress, bucketCount);
const update = await voxelizer.add_points(sensor, points);
```

- `create` rejects when there is no WebGPU adapter that can run compute
  shaders, `bucketCount` is 0 or the hashmap does not fit the adapter's limits
  (see [GPU voxelizer](#gpu-voxelizer)).
- `WgpuAccelerated.memory_size(bucketCount)` gives the GPU memory it takes.
- `add_points`, `get_all_points` and `get_all_points_compressed` return
  Promises.
- Await each call before starting the next one. A call made while another one
  is running rejects (`set_multipliers` and `get_multipliers` throw).

### Tests on Node.js

Node.js has no WebGPU, so the JS tests of the `wgpu` feature use the `webgpu`
package (Dawn). Install it with `npm ci` in `tests/js`.
