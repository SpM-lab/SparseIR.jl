# SparseIR

Pure Julia implementation of sparse-ir (`https://github.com/SpM-lab/sparse-ir`) for the intermediate representation of propagators.


### Usage

```Julia
using SparseIR
beta = 10.0
wmax = 1.0
eps = 1e-7
basis_f = FiniteTempBasis(fermion, beta, wmax, eps)
basis_b = FiniteTempBasis(boson, beta, wmax, eps)
```

A more detailed tutorial and sample codes are available at

* [PDF] https://www.overleaf.com/read/trcfnpmrhbcr
* [Sampling codes] https://github.com/shinaoka/numerical_recipes_many_body_physics/tree/main/julia

