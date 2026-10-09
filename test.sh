#!/bin/env bash

set -e

suites=(
    "single_threaded"
    "multi_threaded"
    "single_threaded,multi_threaded"
    "c"
    "c,single_threaded"
    "c,multi_threaded"
    "c,single_threaded,multi_threaded"
    "cpp"
    "python"
    "python,single_threaded"
    "python,multi_threaded"
    "python,single_threaded,multi_threaded"
    "js"
    "js,single_threaded"
    "wgpu"
    "wgpu,single_threaded"
)

echo "Running cargo check for all suites..."
for suite in "${suites[@]}"; do
    echo "Running cargo check for suite: $suite"
    RUSTFLAGS="-D warnings" cargo check --no-default-features --features="$suite"
done

echo "Running cargo clippy for all suites..."
for suite in "${suites[@]}"; do
    echo "Running cargo clippy for suite: $suite"
    RUSTFLAGS="-D warnings" cargo clippy --no-default-features --features="$suite"
done

echo "Running cargo test for all suites..."
for suite in "${suites[@]}"; do
    echo "Running cargo test for suite: $suite"
    RUSTFLAGS="-D warnings" cargo test --no-default-features --features="$suite"
done
