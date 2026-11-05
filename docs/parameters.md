# Parameters and output

[Back to README](../README.md)

## Creating a voxelizer

Every voxelizer takes the same parameters, in this order: `resolution`,
`intensity_multiplier`, `hit_multiplier`, `target_multiplier`,
`destroy_threshold`, `construct_threshold`, `max_value`, `compress`.

In Rust they are a `VoxelizerParams` struct. The three multipliers sit in a
`Multipliers` struct with the fields `intensity`, `hit` and `target`.

The implementations take one more argument at the end:

| Implementation | Extra argument |
|---|---|
| `SingleThreaded` | none |
| `MultiThreaded` | `thread_count` |
| `WgpuAccelerated` | `bucket_count`, the size of its GPU hashmap (see [bindings](bindings.md#gpu-voxelizer)) |

## How a point changes voxels

`resolution` is the voxel edge length: point coordinates are divided by it.

`intensity_multiplier` scales the intensity of every point before it is
truncated to an integer. For a point of intensity `i`:

- the voxel the ray ends in gets `trunc(i * intensity_multiplier) * target_multiplier` added,
- every voxel the ray passes through gets `trunc(i * intensity_multiplier) * hit_multiplier` subtracted.

Negative products count as 0. Pass `1.0` as `intensity_multiplier` to use the
intensities as they are.

## Value cap

`max_value` caps voxel values. A voxel seen many times saturates at it instead
of growing without bound, so `max_value / hit_multiplier` passing rays are
always enough to clear it again. Pass `4294967295` (`u32::MAX`) for no cap.

## Thresholds and `compress`

With `compress` a voxel is:

- **constructed** when it reaches `construct_threshold`,
- **destroyed** when it is constructed and drops to `destroy_threshold` or below.

Between the two thresholds a voxel keeps its state. A voxel that never reached
`construct_threshold` is never destroyed. With `compress`, a `max_value` below
`construct_threshold` means no voxel ever gets constructed.

## What `add_points` returns

`add_points(sensor_pos, points)` returns a list of `PointI` (`x`, `y`, `z` as
signed integers, `i` unsigned).

**Without `compress`:** one `PointI` per voxel whose value changed in this call.
`x`, `y`, `z` are the voxel coordinates and `i` its new value (not the
difference). A voxel that dropped to 0 is reported too, with `i = 0`.

**With `compress`:** only voxels that got constructed or destroyed count. They
are reported per 4x4x2 block of voxels, one `PointI` per changed block:

- `x`, `y`, `z` is the block origin: `x` and `y` are multiples of 4, `z` a multiple of 2,
- `i` is the occupancy mask of the whole block, bit `(x & 3) + 4 * (y & 3) + 16 * (z & 1)` for voxel `(x, y, z)`.

A block whose voxels were all destroyed comes with `i = 0`. This is the format
[r3v-renderer](https://github.com/igorosky/r3v-renderer) takes.

## Changing the multipliers

The multipliers can be changed after a voxelizer is created. The new ones are
used by the following `add_points` calls. Voxels keep the values they got so
far.

| Language | Set | Get |
|---|---|---|
| Rust | `set_multipliers(Multipliers)` | `multipliers()` |
| C / C++ | `set_multipliers(voxelizer, multipliers)` | `get_multipliers(voxelizer)` (all 0 for NULL) |
| Python, JS | `set_multipliers(Multipliers)` | `get_multipliers()` |

In Python and JS `Multipliers` is a class like `PointF`. The JS
`WgpuAccelerated` throws from both while another call (`add_points` or a dump)
is running.

## Dumping the whole map

`get_all_points` returns every voxel that is not 0 with its value, as `PointI`s
(with or without `compress`).

`get_all_points_compressed` returns every constructed voxel, in groups of
64x64x64 voxels that hold at least one:

- `point` is the group key: voxel coordinates shifted right by 6, `i` is 0,
- `values` are 4096 (`POINTS_IN_VOXEL_GROUP`) 64 bit words, one per 4x4x4 page
  of the group, in z-order (bits of the page offset in the group interleaved
  x, y, z from the lowest one),
- bit `x + 4 * y + 16 * z` of a word is set when voxel `(x, y, z)` of its page
  is constructed.

Without `compress` a voxel counts as constructed once it reaches
`construct_threshold`. r3v-renderer takes these groups directly
(`replace_input_geometry`).

| Language | `get_all_points` | `get_all_points_compressed` |
|---|---|---|
| Rust | `Vec<PointI>` | `Vec<CompressedVoxelGroup>` |
| C / C++ | `VoxelsUpdate`, free with `drop_update` | `VoxelGroups` (array of `VoxelGroup` with `point` and `values[POINTS_IN_VOXEL_GROUP]`), free with `drop_voxel_groups` |
| Python | list of `PointI` | list of `CompressedVoxelGroup`, `point()` and `values()` are methods |
| JS | array of `PointI` | array of `CompressedVoxelGroup`, `point` and `values` (a `BigUint64Array`) are properties |

In C and C++ both return an empty result for NULL. The JS `WgpuAccelerated`
returns Promises from both, and they reject while another call is running.
