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
#
# Every argument but --no-plot and --check goes to the benchmark
# (benches/voxelizer.rs). Results land in target/bench/<date>/: results.csv,
# run.log (everything the benchmark printed) and, with docker, the plots of
# benches/plot.py, which runs in an image built from benches/Dockerfile.
#
# --check runs benches/check.rs instead, a fixed small set of cases meant for
# optimization work: each case is reported as faster, slower or within the
# noise (~) of the baseline and of the previous run, and output tells whether
# the voxels returned changed. Runs land in target/bench/check/runs/, the
# baseline in target/bench/check/baseline.csv. Compare runs of the same
# machine only.
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
help=0
args=()
for arg in "$@"; do
    case "$arg" in
        --no-plot) plot=0 ;;
        --check) check=1 ;;
        --help | -h) help=1 ;;
        *) args+=("$arg") ;;
    esac
done

if [[ $check -eq 1 ]]; then
    bench+=(--bench check)
else
    bench+=(--bench voxelizer)
fi

if [[ $help -eq 1 ]]; then
    "${bench[@]}" -- --help
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
