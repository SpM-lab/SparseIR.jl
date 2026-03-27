# SparseIR.jl Development Guide

This document provides detailed instructions for developing the `SparseIR.jl` Julia package.

## Package Overview

- **Language**: Julia
- **Build System**: Julia Package Manager (Pkg)
- **Test Framework**: Julia Test.jl
- **Documentation**: Documenter.jl
- **Main Files**: `src/`

## Development Environment Setup

```julia
# Run in Julia REPL
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Run tests
using Pkg
Pkg.test()

# Build documentation
using Documenter
include("docs/make.jl")
```

### Rust Backend Development

SparseIR now builds its Rust backend during `Pkg.build("SparseIR")`.

**Directory Structure:**
```
projects/
├── sparse-ir/
│   ├── sparse-ir-rs/      # Rust implementation with C API
│   └── SparseIR.jl/       # Julia bindings
```

**Build source priority:**

1. `../sparse-ir-rs` if the sibling checkout exists
2. pinned `sparse-ir-capi` `0.8.1` from crates.io otherwise

**Important:** `Pkg.add("SparseIR")` runs the build step automatically on first
install, but `Pkg.develop(...)` does not. For development checkouts, run:

```julia
using Pkg
Pkg.build("SparseIR")
```

This build step:

- builds the Rust backend with `cargo build --release --features system-blas`
- copies `libsparse_ir_capi.(dylib|so|dll)` into `deps/`
- regenerates `deps/C_API.jl` for installed package trees
- keeps `src/C_API.jl` as the source-tree fallback binding file
- updates `deps/backend.stamp` so backend rebuilds invalidate Julia precompile state
- records progress in `deps/build-state.toml`
- writes detailed logs to `deps/build.log`

**Build-time environment variables:**

- `SPARSEIR_BUILD_DEBUG=1`
  Keeps the temporary crates.io workspace after a successful build. Failed builds
  also keep the workspace path for inspection.
- `SPARSEIR_BUILD_DEBUGINFO=none|line|limited|full`
  Controls the debuginfo level embedded in the Rust library. The default is
  `line`, which maps to Rust's `line-tables-only`.

**Examples:**

```bash
julia --project=. -e 'using Pkg; Pkg.build("SparseIR")'
SPARSEIR_BUILD_DEBUG=1 julia --project=. -e 'using Pkg; Pkg.build("SparseIR")'
SPARSEIR_BUILD_DEBUGINFO=full julia --project=. -e 'using Pkg; Pkg.build("SparseIR")'
```

**Build diagnostics:**

```bash
cat deps/build-state.toml
tail -n 50 deps/build.log
```

Manual precompile-cache deletion should normally not be necessary. Backend
rebuilds update `deps/backend.stamp`, and the generated C API bindings track
that file as a precompile dependency.

## Code Structure

```
src/
├── SparseIR.jl          # Main file
├── abstract.jl          # Abstract type definitions
├── basis.jl             # Basis classes
├── basis_set.jl         # Basis sets
├── C_API.jl             # C API bindings
├── dlr.jl               # DLR implementation
├── freq.jl              # Frequency-related functions
├── kernel.jl            # Kernel functions
├── poly.jl              # Polynomial classes
├── sampling.jl          # Sampling
└── sve.jl               # SVE implementation
```

## Important Notes

### 1. Julia's Type System
- Design considering type immutability
- Type stability for performance
- Leverage generic programming

### 2. Memory Management
- Proper resource management using `finalizer`
- Prevent memory leaks in C API integration
- Design considering garbage collection

### 3. Performance
- Ensure type stability
- Avoid unnecessary allocations
- Leverage vectorized operations

### 4. Testing
- Unit tests for each function
- Integration tests
- Performance tests

## Commonly Used Commands

```julia
# Load the package
using SparseIR

# Run tests
using Pkg
Pkg.test()

# Run specific test file
include("test/runtests.jl")

# Build documentation
using Documenter
include("docs/make.jl")

# Performance testing
using BenchmarkTools
@benchmark some_function()

# Check memory usage
using Profile
@profile some_function()
```

```bash
# Clear precompilation cache (if code changes aren't reflected)
rm -rf ~/.julia/compiled/v1.*/SparseIR/

# Rebuild local library
julia --project=. -e "using Pkg; Pkg.build(\"SparseIR\")"
```

## Debugging

```julia
# Run in debug mode
julia --project=. -e "
using SparseIR
# Debug code
"

# Profiling
using Profile
@profile some_function()
Profile.print()

# Memory leak check
using Profile
@profile some_function()
Profile.print()
```

## C API Integration

- C API functions are defined in `C_API.jl`
- Implement error handling properly
- Perform type conversions explicitly
- Automate memory management with `finalizer`

## Release Procedure

1. Update version number in `Project.toml`
2. Update CHANGELOG
3. Ensure all tests pass
4. Build documentation
5. Create and push a tag
