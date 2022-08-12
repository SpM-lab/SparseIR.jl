# SparseIR

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][GHA-img]][GHA-url] [![][codecov-img]][codecov-url] |

Pure Julia implementation of [sparse-ir](`https://github.com/SpM-lab/sparse-ir`) for the intermediate representation of propagators.

## Installation
The package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add SparseIR
```

## Usage

```julia
using SparseIR
beta = 10.0
ωmax = 1.0
eps = 1e-7
basis_f = FiniteTempBasis(fermion, beta, ωmax, eps)
basis_b = FiniteTempBasis(boson, beta, ωmax, eps)
```

## Tutorial and sample codes
More detailed tutorial and sample codes are available [online](https://spm-lab.github.io/sparse-ir-tutorial/).



[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://spm-lab.github.io/SparseIR.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://spm-lab.github.io/SparseIR.jl/stable/

[GHA-img]: https://github.com/SpM-lab/SparseIR.jl/workflows/CI/badge.svg
[GHA-url]: https://github.com/SpM-lab/SparseIR.jl/actions?query=workflows/CI

[codecov-img]: https://codecov.io/gh/SpM-lab/SparseIR.jl/branch/main/graph/badge.svg?token=tdMvTruYa4
[codecov-url]: https://codecov.io/gh/SpM-lab/SparseIR.jl

[issues-url]: https://github.com/SpM-lab/SparseIR.jl/issues
