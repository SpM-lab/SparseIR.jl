# SparseIR.jl

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SpM-lab/SparseIR.jl)

Julia wrapper for the [libsparseir](https://github.com/SpM-lab/libsparseir) C++ library, providing high-performance sparse intermediate representation (IR) of correlation functions in many-body physics.

> [!WARNING]
> This Julia wrapper is still under development. For production use, please consider:
> - [SparseIR.jl](https://github.com/SpM-lab/SparseIR.jl) - Pure Julia implementation (recommended)
> - [sparse-ir](https://github.com/SpM-lab/sparse-ir) - Python implementation
> - [sparse-ir-fortran](https://github.com/SpM-lab/sparse-ir-fortran) - Fortran implementation

## Overview

SparseIR.jl provides Julia bindings to the C++ libsparseir library, offering optimized routines for:

- **Basis construction**: Finite temperature basis using singular value expansion (SVE)
- **Sparse sampling**: Efficient sampling in imaginary time and Matsubara frequencies
- **DLR transformations**: Discrete Lehmann Representation for compact representation
- **Augmented basis**: Extended basis for vertex functions and multi-point correlations

## Features

### Implemented Components

- ✅ **Statistics**: Fermionic and Bosonic
- ✅ **Kernels**: LogisticKernel, RegularizedBoseKernel
- ✅ **Basis Functions**: FiniteTempBasis with SVE
- ✅ **Sampling**: TauSampling, MatsubaraSampling
- ✅ **DLR**: DiscreteLehmannRepresentation
- ✅ **Augmentation**: TauConst, TauLinear, MatsubaraConst
- ✅ **Basis Sets**: FiniteTempBasisSet for multi-Λ calculations

### API Example

```julia
using SparseIR

# Create basis for fermionic system
β = 10.0        # Inverse temperature
ωmax = 1.0      # Frequency cutoff
ε = 1e-10       # Accuracy parameter

basis = FiniteTempBasis(Fermionic(), β, ωmax, ε)

# Sparse sampling in imaginary time
tau_sampling = TauSampling(basis)
tau_points = tau_sampling.sampling_points

# Transform between representations
coeffs = fit(tau_sampling, gtau_values)
gtau_reconstructed = evaluate(tau_sampling, coeffs)
```

## Installation

### Prerequisites

1. **Julia 1.10+**
2. **libsparseir C++ library** built with C API support

### Setup Instructions

1. **Build libsparseir with C API**:
   ```sh
   git clone https://github.com/SpM-lab/libsparseir.git
   cd libsparseir
   ./build_capi.sh
   ```

2. **Clone and install SparseIR.jl**:
   ```sh
   git clone https://github.com/SpM-lab/SparseIR.jl.git
   cd SparseIR.jl
   ```

3. **Build the Julia package**:
   ```sh
   julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
   ```
   This generates `src/C_API.jl` by parsing the C headers using Clang.jl.

4. **Run tests**:
   ```sh
   julia --project -e 'using Pkg; Pkg.test()'
   ```

## Testing

We use [ReTestItems.jl](https://github.com/JuliaTesting/ReTestItems.jl) for testing with tagged test groups:

- Run all tests: `julia -e 'using Pkg; Pkg.test()'`
- Run specific test groups:
  ```julia
  using Pkg
  Pkg.test(test_args=["--tags", "julia"])      # Pure Julia tests
  Pkg.test(test_args=["--tags", "wrapper"])    # C wrapper tests
  Pkg.test(test_args=["--tags", "cinterface"]) # C interface tests
  ```

## Tutorials

Interactive tutorials are available in the `tutorials/` directory:

- **Pluto notebooks** (`tutorials/pluto/`):
  - `sparse_sampling.jl` - Introduction to sparse sampling
  - `transformation_IR.jl` - IR basis transformations
  - `dlr.jl` - Discrete Lehmann Representation
  - `dmft_ipt.jl` - DMFT with IPT solver
  - `flex.jl` - FLEX approximation example
  - `BgG.jl` - Bethe-Goldstone calculations

- **Jupyter notebooks** (`tutorials/jupyter/`) - Similar content in Jupyter format

To run Pluto tutorials:
```julia
using Pluto
Pluto.run(notebook="tutorials/pluto/sparse_sampling.jl")
```

## Project Structure

```
SparseIR.jl/
├── src/
│   ├── SparseIR.jl      # Main module
│   ├── C_API.jl            # Auto-generated C bindings
│   ├── abstract.jl         # Abstract interfaces
│   ├── lib/                # C library wrappers
│   │   ├── basis.jl        # Basis functions
│   │   ├── kernel.jl       # Kernel implementations
│   │   ├── sampling.jl     # Sampling routines
│   │   ├── dlr.jl          # DLR transformations
│   │   └── sve.jl          # SVE computations
│   └── spir/               # Pure Julia extensions
│       ├── augment.jl      # Augmented basis
│       └── basis_set.jl    # Multi-Λ basis sets
├── test/                   # Test suites
├── tutorials/              # Interactive tutorials
└── builder/                # BinaryBuilder scripts
```
