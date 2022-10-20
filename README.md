SparseIR - intermediate representation of propagators in Julia
==============================================================
[![][docs-stable-img]][docs-stable-url]
[![][docs-dev-img]][docs-dev-url] 
[![][GHA-img]][GHA-url]
[![][codecov-img]][codecov-url]

This library provides routines for constructing and working with the
intermediate representation of correlation functions.  It provides:

 - on-the-fly computation of basis functions for arbitrary cutoff Λ
 - basis functions and singular values are accurate to full precision
 - routines for sparse sampling

Installation
------------
SparseIR can be installed with the Julia package manager.  Simply run the following from the command line:
```
julia -e 'import Pkg; Pkg.add("SparseIR")'
```
We support Julia, version 1.6 and above, and recommend Julia 1.8 or above for optimal performance.  We only 
depend on a few quad-precision libraries for the accurate computation of the singular value decomposition,
which are automatically installed.  (A full list of dependencies can be found in `Project.toml`.)

To manually install the current development version, you can use the following:
```
# Only recommended for developers - no automatic updates!
git clone https://github.com/SpM-lab/SparseIR.jl
julia -e 'import Pkg; Pkg.develop(path="SparseIR.jl")'
```

Documentation and tutorial
--------------------------
Check out our [comprehensive tutorial], where we self-contained
notebooks for several many-body methods - GF(2), GW, Eliashberg equations,
Lichtenstein formula, FLEX, ... - are presented.

Refer to the [API documentation] for more details on how to work
with the Julia library.

There is also a [Python library] and (currently somewhat restricted)
[Fortran library] available for the IR basis and sparse sampling.

[comprehensive tutorial]: https://spm-lab.github.io/sparse-ir-tutorial
[API documentation]: https://spm-lab.github.io/SparseIR.jl/stable/
[Python library]: https://github.com/SpM-lab/sparse-ir
[Fortran library]: https://github.com/SpM-lab/sparse-ir-fortran


Example usage
-------------

```julia
using SparseIR
β = 10
ωmax = 1
ε = 1e-7
basis_f = FiniteTempBasis(Fermionic(), β, ωmax, ε)
basis_b = FiniteTempBasis(Bosonic(), β, ωmax, ε)
```

License and citation
--------------------
This software is released under the MIT License.  See LICENSE for details.

If you find the intermediate representation, sparse sampling, or this software
useful in your research, please consider citing the following papers:

 - Hiroshi Shinaoka et al., [Phys. Rev. B 96, 035147]  (2017)
 - Jia Li et al., [Phys. Rev. B 101, 035144] (2020)
 - Markus Wallerberger et al., [arXiv 2206.11762] (2022)

If you are discussing sparse sampling in your research specifically, please
also consider citing an independently discovered, closely related approach, the
MINIMAX isometry method (Merzuk Kaltak and Georg Kresse,
[Phys. Rev. B 101, 205145], 2020).

[Phys. Rev. B 96, 035147]: https://doi.org/10.1103/PhysRevB.96.035147
[Phys. Rev. B 101, 035144]: https://doi.org/10.1103/PhysRevB.101.035144
[arXiv 2206.11762]: https://doi.org/10.48550/arXiv.2206.11762
[Phys. Rev. B 101, 205145]: https://doi.org/10.1103/PhysRevB.101.205145


[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://spm-lab.github.io/SparseIR.jl/dev/
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://spm-lab.github.io/SparseIR.jl/stable/
[GHA-img]: https://github.com/SpM-lab/SparseIR.jl/workflows/CI/badge.svg
[GHA-url]: https://github.com/SpM-lab/SparseIR.jl/actions?query=workflows/CI
[codecov-img]: https://codecov.io/gh/SpM-lab/SparseIR.jl/branch/main/graph/badge.svg?token=tdMvTruYa4
[codecov-url]: https://codecov.io/gh/SpM-lab/SparseIR.jl

[issues-url]: https://github.com/SpM-lab/SparseIR.jl/issues
