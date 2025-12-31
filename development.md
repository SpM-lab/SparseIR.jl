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

### Using Local libsparseir for Development

During development, you may want to use a locally built version of `libsparseir` instead of the pre-built JLL package. This allows you to test changes to the C API immediately.

**Directory Structure:**
```
projects/
├── sparse-ir/
│   ├── sparse-ir-rs/      # Rust implementation with C API
│   └── SparseIR.jl/       # Julia bindings
```

**Steps:**

1. **Place repositories side by side**: Ensure `sparse-ir-rs` and `SparseIR.jl` are in the same parent directory.

2. **Rebuild SparseIR.jl**: The build script automatically detects `sparse-ir-rs` and builds it:
   ```julia
   using Pkg
   Pkg.build("SparseIR")
   ```

   This will:
   - Detect `../sparse-ir-rs` directory
   - Build `sparse-ir-rs` with `cargo build --release --features system-blas`
   - Copy `libsparse_ir_capi.dylib` (or `.so`/`.dll`) to `deps/`
   - Regenerate C API bindings in `src/C_API.jl`

3. **Verify the build**: Check that the local library was built:
   ```bash
   ls -lh deps/libsparse_ir_capi.dylib
   ```

4. **Test with local library**:
   ```julia
   using SparseIR
   basis = FiniteTempBasis{Fermionic}(10.0, 1.0, 1e-6)
   println("Basis size: ", length(basis))
   ```

**Note:** If `sparse-ir-rs` is not found in the expected location, SparseIR.jl will fall back to using the pre-built `libsparseir_jll` package from Julia's package registry.

### Clearing Build Cache

If you need to force a clean rebuild (e.g., after updating `sparse-ir-rs`), you can clear the build cache:

```bash
# Remove the built library and cached files
rm -rf deps/libsparse_ir_capi.*
rm -rf deps/build.log

# Then rebuild
julia --project=. -e "using Pkg; Pkg.build(\"SparseIR\")"
```

Alternatively, you can clear Julia's precompilation cache:

```bash
# Clear all precompiled files for SparseIR
rm -rf ~/.julia/compiled/v1.*/SparseIR/

# Then restart Julia and reload the package
julia --project=. -e "using SparseIR"
```

For a complete clean rebuild:

```bash
# 1. Clean local build artifacts
rm -rf deps/

# 2. Clear precompilation cache
rm -rf ~/.julia/compiled/v1.*/SparseIR/

# 3. Rebuild and precompile
julia --project=. -e "using Pkg; Pkg.build(\"SparseIR\"); using SparseIR"
```

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

