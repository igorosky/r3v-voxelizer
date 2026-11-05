#!/usr/bin/env bash
#
# Line coverage of the Rust tests and of the C, C++ and Python binding suites
# with cargo-llvm-cov. Writes one lcov file per run to target/coverage, CI
# merges them into one report (.github/workflows/coverage.yml).
#
#   ./coverage.sh              all groups
#   ./coverage.sh core python  only the given groups: core, c, python
#
# One file per run because the runs build the same cdylib with different
# features: a later build replaces the binary an earlier run's profile belongs
# to. The JS suite is left out, wasm32 can not be instrumented.
#
# Needs cargo-llvm-cov and the llvm-tools-preview rustup component, plus what
# ./test.sh needs for the same groups.

set -euo pipefail

out=target/coverage
# Test code does not count.
ignore='(^|/)(tests|benches|examples)/|/src/test\.rs$'

locked=()
if [[ -n "${CI:-}" ]]; then
    locked=(--locked)
fi

# Runs a command, folded into its own section in GitHub Actions logs.
run() {
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::group::$*"
        "$@"
        echo "::endgroup::"
    else
        echo "==> $*"
        "$@"
    fi
}

# cov <name> <features> [cargo args...]
cov() {
    local name=$1 features=$2
    shift 2
    run cargo llvm-cov "${locked[@]}" --no-default-features --features="$features" \
        --ignore-filename-regex "$ignore" --lcov --output-path "$out/$name.info" "$@"
}

group_core() {
    cov core "single_threaded,multi_threaded,wgpu" --lib
}

group_c() {
    cov c "c,cpp,single_threaded,multi_threaded,wgpu" --test c_bindings
}

group_python() {
    cov python "python,single_threaded,multi_threaded,wgpu" --test python_bindings
}

groups=("$@")
if [[ ${#groups[@]} -eq 0 ]]; then
    groups=(core c python)
fi

mkdir -p "$out"
for group in "${groups[@]}"; do
    case "$group" in
        core | c | python) "group_$group" ;;
        *)
            echo "unknown group: $group (expected core, c or python)" >&2
            exit 1
            ;;
    esac
done
