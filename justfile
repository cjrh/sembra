# Sembra - justfile for common development tasks

# Show available commands
default:
    @just --list

# Run all tests
test:
    cargo test --all

# Run benchmarks
bench:
    cargo bench

# Run clippy lints
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Build release binary with profiling symbols (overrides global strip setting)
build-profile:
    CARGO_PROFILE_RELEASE_STRIP=none cargo build --release --example profile_test

# Profile the example and generate perf.data
profile: build-profile
    perf record -F 999 --call-graph fp ./target/release/examples/profile_test

# Show profiling hotspots as a table (top functions by CPU time)
profile-report:
    @echo "=== Top CPU Hotspots ==="
    @perf report --stdio --no-children --percent-limit 1 | head -50

# Show detailed profiling with call graphs
profile-detailed:
    @echo "=== Detailed Profile with Call Graphs ==="
    @perf report --stdio --percent-limit 1 --call-graph | head -150

# Complete profiling workflow: build, profile, and show report
profile-full: build-profile
    @echo "Running profiling..."
    perf record -F 999 --call-graph fp ./target/release/examples/profile_test
    @echo ""
    @echo "=== Top CPU Hotspots ==="
    @perf report --stdio --no-children --percent-limit 2 | head -60

# Generate flamegraph (requires cargo-flamegraph)
flamegraph:
    cargo flamegraph --example profile_test -o flamegraph.svg
    @echo "Flamegraph saved to flamegraph.svg"

# Clean up profiling data
profile-clean:
    rm -f perf.data perf.data.old flamegraph.svg

# Run benchmarks and save output
bench-save:
    cargo bench 2>&1 | tee benchmark_results.txt
    @echo "Benchmark results saved to benchmark_results.txt"

# Compare energy modes (backward vs forward)
bench-energy:
    @echo "Benchmarking energy modes..."
    @cargo bench --bench seam_carving_bench -- "energy_modes" | grep -E "(Backward|Forward|time:)"

# Quick performance test on a small image
perf-quick: build-profile
    @echo "Quick performance test..."
    @time ./target/release/examples/profile_test

# Full development check: test, lint, and build
check: test lint
    cargo build --release
    @echo "✅ All checks passed!"
