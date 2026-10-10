#!/usr/bin/env bash
#
# Lints the important feature combinations (tests included) and runs every
# test suite once.
#
#   ./test.sh              all groups
#   ./test.sh core python  only the given groups: core, c, python, js
#
# CI runs one group per job (.github/workflows/ci.yml).
#
# Unit tests run once with every implementation enabled: the implementations
# do not depend on any other feature, so running them again next to a binding
# feature would only repeat the same tests. Binding suites run once per
# feature set that changes what the bindings export.
#
# Needs a Vulkan device for the wgpu tests (a software one such as lavapipe
# works), python3 and a C compiler for the bindings tests, and wasm-pack, the
# wasm32-unknown-unknown target and node for the JS tests.

set -euo pipefail

export RUSTFLAGS="-D warnings"

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

# clippy <features> [cargo args...]
clippy() {
    local features=$1
    shift
    run cargo clippy "${locked[@]}" --all-targets --no-default-features --features="$features" "$@"
}

# cargo_test <features> [cargo args...]
cargo_test() {
    local features=$1
    shift
    run cargo test "${locked[@]}" --no-default-features --features="$features" "$@"
}

group_core() {
    local features
    local sets=(
        ""
        "single_threaded"
        "multi_threaded"
        "wgpu"
        "single_threaded,multi_threaded"
        "single_threaded,wgpu"
        "multi_threaded,wgpu"
        "single_threaded,multi_threaded,wgpu"
    )
    for features in "${sets[@]}"; do
        clippy "$features"
    done
    cargo_test "single_threaded,multi_threaded,wgpu" --lib
}

group_c() {
    local features
    local sets=(
        "c"
        "c,single_threaded"
        "c,multi_threaded"
        "c,single_threaded,multi_threaded"
        "cpp"
        "cpp,single_threaded,multi_threaded"
        "c,cpp,single_threaded,multi_threaded"
    )
    for features in "${sets[@]}"; do
        clippy "$features"
    done
    cargo_test "c,single_threaded,multi_threaded" --test c_bindings
}

group_python() {
    local features
    local sets=(
        "python"
        "python,single_threaded"
        "python,multi_threaded"
        "python,single_threaded,multi_threaded"
    )
    for features in "${sets[@]}"; do
        clippy "$features"
    done
    for features in "${sets[@]}"; do
        cargo_test "$features" --test python_bindings
    done
}

group_js() {
    local features
    local sets=(
        "js"
        "js,single_threaded"
    )
    for features in "${sets[@]}"; do
        clippy "$features"
        clippy "$features" --target wasm32-unknown-unknown
    done
    for features in "${sets[@]}"; do
        cargo_test "$features" --test js_bindings
    done
}

groups=("$@")
if [[ ${#groups[@]} -eq 0 ]]; then
    groups=(core c python js)
fi

for group in "${groups[@]}"; do
    case "$group" in
        core | c | python | js) "group_$group" ;;
        *)
            echo "unknown group: $group (expected core, c, python or js)" >&2
            exit 1
            ;;
    esac
done
