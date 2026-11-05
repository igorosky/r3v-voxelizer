# Benchmarks

[Back to README](../README.md)

- [Full benchmark](#full-benchmark)
- [Quick check while optimizing](#quick-check-while-optimizing)
- [Flamegraphs](#flamegraphs)

## Full benchmark

```bash
./bench.sh            # full sweep
./bench.sh --quick    # smaller sweep
./bench.sh --help     # every option
```

`bench.sh` runs every implementation (`SingleThreaded`, `MultiThreaded` at
several thread counts and `WgpuAccelerated`) on the same generated workloads.
Options select implementations, thread counts, points per frame, ranges,
`compress`, frame counts and more.

Results go to `target/bench/<date>/`:

- `results.csv`
- `run.log`
- plots of the CSV (`benches/plot.py`), when docker is available. `--no-plot` skips them.

There are two scenarios:

| Scenario | What it does |
|---|---|
| `room` | A sensor circling inside a box room, rays in every direction |
| `cone` | A static sensor whose rays all end on a small patch of a wall (many rays hitting the same voxels) |

The `match` column tells whether an implementation returned the same voxels as
the first one run on the workload. A run that panics is reported as such and
the sweep goes on.

`FEATURES` picks the implementations that get built (default
`single_threaded,multi_threaded,wgpu`). On a machine without a GPU:

```bash
FEATURES=single_threaded,multi_threaded ./bench.sh
```

The benchmark itself is `benches/voxelizer.rs`, also runnable with
`cargo bench --features multi_threaded,wgpu --bench voxelizer -- <options>`.

## Quick check while optimizing

`./bench.sh --check` runs a fixed set of four cases (`compress` on) on every
implementation in about a minute. It compares the run with a saved baseline
and with the previous run.

```bash
./bench.sh --check --save-baseline   # once, on the code you start from
./bench.sh --check                   # after every change
./bench.sh --check --impls st        # only the implementation you work on
```

For every case it prints:

| Column | Meaning |
|---|---|
| `ms/frame` | Median over the reps |
| `±` | Median absolute deviation of the reps |
| vs baseline, vs previous | `faster` or `slower` only when the change is larger than `--threshold` (5% by default) and twice the `±` of both runs, `~` otherwise |
| `output` | `CHANGED` when the returned voxels differ from the baseline. An optimization should never do that. |

A summary line per implementation gives the geometric mean over the cases.
`./bench.sh --check --help` lists the options.

Runs are saved to `target/bench/check/runs/`, the baseline to
`target/bench/check/baseline.csv`.

Tips for stable numbers:

- Compare runs of the same machine and the same `--impls` only.
- On laptops the numbers depend a lot on the power profile and on other
  programs. Use a performance profile and close heavy programs.
- On CPUs with performance and efficiency cores, `st` and the host side of
  `wgpu` are pinned to one performance core (`--pin` changes or disables it).
  `mt` runs on every CPU.
- `wgpu` timings are noisier than the CPU ones, so its bands are wider.

The check is built with the `bench-check` profile (`Cargo.toml`): release with
`debug = "line-tables-only"`, so a profiler such as `perf` maps samples to
source lines. The binary is `target/bench-check/deps/check-*`, apart from the
other benches.

## Flamegraphs

`--flamegraph` runs the benchmark (with `--check`, the check) through
[`cargo flamegraph`](https://github.com/flamegraph-rs/flamegraph). It needs
`cargo install flamegraph` and `perf`.

```bash
./bench.sh --check --flamegraph --impls st
```

perf needs `kernel.perf_event_paranoid` at 2 or lower (Ubuntu sets 4):

```bash
sudo sysctl kernel.perf_event_paranoid=2
```

The build uses the `bench-check` profile with
`RUSTFLAGS="-C force-frame-pointers=yes"` in `target/flamegraph/`, so other
builds are kept, and perf walks the call stacks through the frame pointers.

Output:

- `target/bench/flamegraph/<date>-<benchmark>.svg`: the flamegraph,
- `target/bench/flamegraph/<date>-<benchmark>.txt`: every function sorted by
  CPU time. The top of the list is printed:

```
12.138 s of CPU time, summed over every thread

            total              self  function
 94.67%   11.491 s  18.29%    2.221 s  <SingleThreaded as Voxelizer>::add_points
 73.05%    8.867 s  66.79%    8.107 s  <SingleThreaded>::hit_voxel
  8.98%    1.090 s   8.98%    1.090 s  bmi2::index_of_unchecked::<u16, 3>
```

- `total` is the time including called functions (counted once per stack, so
  recursion does not count twice), `self` the time in the function itself.
  Both are summed over every thread.
- Inlined functions are listed on their own. Names are shortened to the type,
  or the last module of a free function. Generic arguments stay.

Files of the latest run:

- `target/flamegraph/stacks.folded`: folded stacks, turned into the table by
  `benches/hotspots.awk`,
- `target/flamegraph/perf.data`: open with `perf report -i target/flamegraph/perf.data`.

Notes:

- Frame pointers and perf slow the run down a little, so a profiled check run
  is not saved (`--no-save`) and its comparisons are only a hint.
- Profile one implementation at a time.
- perf samples every running thread at 997 Hz, and cargo flamegraph keeps the
  whole `perf script` output in memory (a few KB per sample). With `mt` the
  frequency is divided by its thread count (`--threads`, every CPU by default),
  which gives about 997 samples a second for any implementation.
