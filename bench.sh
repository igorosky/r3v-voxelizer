#!/usr/bin/env bash
#
# Benchmarks every voxelizer implementation and plots the results.
#
#   ./bench.sh                          full sweep (several minutes)
#   ./bench.sh --quick                  smaller sweep
#   ./bench.sh --no-plot --impls st,mt  only the CPU implementations, no plots
#   ./bench.sh --help                   every option of the benchmark
#   ./bench.sh --check                  about a minute, compared with the
#                                       baseline and the previous run
#   ./bench.sh --check --save-baseline  the same, then saved as the baseline
#   ./bench.sh --check --help           every option of the check
#   ./bench.sh --check --flamegraph --impls st
#                                       the check under perf, then a
#                                       flamegraph of it
#
# Every argument but --no-plot, --check and --flamegraph goes to the benchmark
# (benches/voxelizer.rs). Results land in target/bench/<date>/: results.csv,
# run.log (everything the benchmark printed) and, with docker, the plots of
# benches/plot.py, which runs in an image built from benches/Dockerfile.
#
# --check runs benches/check.rs instead, a fixed small set of cases meant for
# optimization work: each case is reported as faster, slower or within the
# noise (~) of the baseline and of the previous run, and output tells whether
# the voxels returned changed. Runs land in target/bench/check/runs/, the
# baseline in target/bench/check/baseline.csv. Compare runs of the same
# machine only. It is built with the bench-check profile of Cargo.toml
# (release with debug = "line-tables-only", binary in target/bench-check/),
# so that a profiler such as perf can map samples to source lines.
#
# --flamegraph runs the benchmark (with --check the check) through
# cargo flamegraph (cargo install flamegraph) instead: the bench-check profile
# with RUSTFLAGS="-C force-frame-pointers=yes", built in target/flamegraph/,
# and perf records the call stacks through the frame pointers. The flamegraph
# lands in target/bench/flamegraph/<date>-<benchmark>.svg, next to it a table
# (.txt, the top printed) of the CPU time per function, with and without the
# functions called, made by benches/hotspots.awk. The perf data of the latest
# run is in target/flamegraph/perf.data (perf report -i ...), its folded
# stacks in target/flamegraph/stacks.folded. Frame
# pointers and perf slow it down a little: the check run is not saved, and
# its comparisons are only a hint. Profile one implementation at a time
# (--impls). With mt the sampling frequency (997 Hz) is divided by its thread
# count, see below. perf needs kernel.perf_event_paranoid at 2 or lower:
#   sudo sysctl kernel.perf_event_paranoid=2
#
# FEATURES picks the implementations built, e.g.
# FEATURES=single_threaded,multi_threaded on a machine without a GPU. The
# wgpu implementation runs on the default adapter (printed at the start),
# VK_ICD_FILENAMES picks another Vulkan driver. A software one such as
# lavapipe works but says little about GPU performance.

set -euo pipefail

cd "$(dirname "$0")"

features=${FEATURES:-single_threaded,multi_threaded,wgpu}
bench=(cargo bench --no-default-features --features="$features")

plot=1
check=0
flamegraph=0
help=0
args=()
for arg in "$@"; do
    case "$arg" in
        --no-plot) plot=0 ;;
        --check) check=1 ;;
        --flamegraph) flamegraph=1 ;;
        --help | -h) help=1 ;;
        *) args+=("$arg") ;;
    esac
done

if [[ $check -eq 1 ]]; then
    name=check
    # profile.bench-check in Cargo.toml: release with line tables
    bench+=(--profile bench-check --bench check)
else
    name=voxelizer
    bench+=(--bench voxelizer)
fi

if [[ $help -eq 1 ]]; then
    "${bench[@]}" -- --help
    exit
fi

if [[ $flamegraph -eq 1 ]]; then
    if ! cargo flamegraph --version >/dev/null 2>&1; then
        echo "cargo flamegraph not found: cargo install flamegraph" >&2
        exit 1
    fi
    if ! command -v perf >/dev/null; then
        echo "perf not found (linux-tools on Ubuntu)" >&2
        exit 1
    fi
    paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 2)
    if [[ $paranoid -gt 2 ]]; then
        echo "kernel.perf_event_paranoid is $paranoid, perf needs 2 or lower:" >&2
        echo "  sudo sysctl kernel.perf_event_paranoid=2" >&2
        exit 1
    fi
    if [[ $check -eq 1 ]]; then
        # Slower under perf, it must not become the previous run
        args+=(--no-save)
    fi

    # perf samples every running thread at the frequency, and cargo flamegraph
    # holds all the perf script output in memory, a few KB per sample: a
    # check of mt on 16 threads at 997 Hz is 2.3M samples, 20 GB of text, and
    # gets killed. So the frequency is divided by the threads mt runs on
    # (--threads, the largest of a list, every CPU by default), about 997
    # samples a second whatever the implementation.
    threads=$(nproc)
    impls=
    for ((i = 0; i < ${#args[@]}; i++)); do
        case "${args[i]}" in
            --threads)
                threads=$(tr ',' '\n' <<<"${args[i + 1]:-}" | sort -n | tail -n 1)
                ;;
            --impls) impls=${args[i + 1]:-} ;;
        esac
    done
    freq=997
    if [[ ,$impls, == *,mt,* || (-z $impls && ,$features, == *,multi_threaded,*) ]] &&
        [[ $threads =~ ^[0-9]+$ && $threads -gt 1 ]]; then
        freq=$((997 / threads))
        echo "mt on $threads threads, perf samples each at $freq Hz"
    fi

    out=target/bench/flamegraph
    svg=$out/$(date +%Y%m%d-%H%M%S)-$name.svg
    table=${svg%.svg}.txt
    stacks=target/flamegraph/stacks.folded
    mkdir -p "$out" target/flamegraph
    # Other RUSTFLAGS rebuild everything, so a target dir of its own keeps the
    # builds of the other runs. With frame pointers perf walks the stacks
    # itself (--call-graph fp), far less data than copying them (dwarf).
    # cpu-clock: on CPUs with two kinds of cores cycles are two events
    # (cpu_core, cpu_atom) and the flamegraph keeps only the first one seen.
    # The folded stacks the flamegraph is drawn from are kept for the table.
    RUSTFLAGS="${RUSTFLAGS:-} -C force-frame-pointers=yes" \
        CARGO_TARGET_DIR=target/flamegraph \
        cargo flamegraph --no-default-features --features="$features" \
        --profile bench-check --bench "$name" --output "$svg" \
        --cmd "record -e cpu-clock -F $freq --call-graph fp -o target/flamegraph/perf.data" \
        --post-process "tee $stacks" \
        -- "${args[@]}"

    awk -f benches/hotspots.awk "$stacks" >"$table"
    echo
    head -n 43 "$table"
    echo
    echo "flamegraph in $svg"
    echo "every function in $table"
    exit
fi

# Built first, so that the log holds only what the benchmark prints
"${bench[@]}" --no-run

if [[ $check -eq 1 ]]; then
    exec "${bench[@]}" --quiet -- "${args[@]}"
fi

out=target/bench/$(date +%Y%m%d-%H%M%S)
mkdir -p "$out"
{
    commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
    if ! git diff --quiet HEAD 2>/dev/null; then
        commit+=" (uncommitted changes)"
    fi
    echo "commit:          $commit"
    echo "cpu:             $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -n 1)"
    echo "features:        $features"
    "${bench[@]}" --quiet -- --csv "$out/results.csv" "${args[@]}" 2>&1
} | tee "$out/run.log"

if [[ $plot -eq 1 ]]; then
    if command -v docker >/dev/null; then
        docker build -q -t r3v-voxelizer-plot benches/ >/dev/null
        docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/work" -w /work \
            r3v-voxelizer-plot python benches/plot.py "$out"
    else
        echo "docker not found, skipping the plots" >&2
    fi
fi

echo "results in $out"
