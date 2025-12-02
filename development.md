# Development Guide

This document describes how to set up and use the development environment for SparseIR.jl.

## Prerequisites

- Julia 1.6 or later
- Git

## Setting up the Development Environment

### 1. Clone the Repository

```bash
git clone https://github.com/SpM-lab/SparseIR.jl.git
cd SparseIR.jl
```

### 2. Activate the Project Environment

```bash
# Activate the project environment
julia --project=.
```

Or from within Julia:

```julia
using Pkg
Pkg.activate(".")
```

### 3. Install Dependencies

#### Install Main Dependencies

```julia
# Install main dependencies (automatically done with --project=.)
Pkg.instantiate()
```

#### Install Development Dependencies

```julia
# Install development dependencies (includes Clang for code generation)
Pkg.instantiate(; target="dev")
```

#### Install Test Dependencies

```julia
# Install test dependencies
Pkg.instantiate(; target="test")
```

### 4. Verify Installation

```julia
# Check installed packages
Pkg.status()

# Check development dependencies
Pkg.status(; target="dev")

# Check test dependencies
Pkg.status(; target="test")
```

## Development Workflow

### Running Tests

```bash
# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"

# Or run tests with specific target
julia --project=. --target=test -e "using Pkg; Pkg.test()"
```

### Code Generation

The project uses Clang.jl for generating C API bindings.

```julia
# Activate development environment
cd utils
julia --project=@. -e "import Pkg; Pkg.instantiate()"
julia --project=@. generate_C_API.jl
```

Then, `src/C_API.jl` is generated.

### Building Documentation

```bash
# Navigate to docs directory
cd docs

# Activate docs environment and build
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. docs/make.jl
```

## Project Structure

- `src/` - Main source code
- `test/` - Test files
- `deps/` - Build dependencies and code generation scripts
- `docs/` - Documentation source
- `assets/` - Static assets for documentation

## Troubleshooting

1. **Package not found**: Make sure you've activated the correct target:
   ```julia
   Pkg.instantiate(; target="dev")  # for development dependencies
   Pkg.instantiate(; target="test") # for test dependencies
   ```

2. **Clang.jl issues**: Ensure you have the development target activated:
   ```bash
   julia --project=. --target=dev
   ```

3. **Build failures**: Check that all dependencies are properly installed:
   ```julia
   Pkg.status()
   Pkg.status(; target="dev")
   ```

4. **Using locally built shared library**:

   SparseIR.jl can be run against a locally built Rust C-API shared library (`libsparse_ir_capi.dylib` on macOS, `libsparse_ir_capi.so` on Linux) by setting the `SPARSEIR_LIB_PATH` environment variable before Julia is started:

   ```bash
   # Example: point SparseIR.jl to the local Rust C-API build
   export SPARSEIR_LIB_PATH="$HOME/projects/sparse-ir/sparseir-rust/target/release/libsparse_ir_capi.dylib"

   # Optional: enable verbose Rust-side debug logging
   export SPARSEIR_DEBUG=1

   julia --project=. -e "using Pkg; Pkg.test()"
   ```

   The repository also provides a helper script that does this for you:

   ```bash
   cd SparseIR.jl
   ./run.sh   # sets SPARSEIR_LIB_PATH and SPARSEIR_DEBUG, then runs Pkg.test()
   ```

   If you previously ran tests **without** `SPARSEIR_LIB_PATH` set, Julia may have cached a precompiled SparseIR.jl that is still bound to the JLL package.  
   In that case, clear the compiled cache once so that the new setting takes effect:

   ```bash
   rm -rf ~/.julia/compiled/v*/SparseIR

   cd SparseIR.jl
   ./run.sh
   ```

### Getting Help

- Check the [main documentation](README.md)
- Open an issue on GitHub
- Check Julia's package manager documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

For more detailed contribution guidelines, see the main repository documentation.
