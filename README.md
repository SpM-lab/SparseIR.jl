SparseIR - intermediate representation of propagators in Julia
==============================================================

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][GHA-img]][GHA-url] [![][codecov-img]][codecov-url] |

Pure Julia implementation of [sparse-ir](`https://github.com/SpM-lab/sparse-ir`) for the intermediate representation of propagators.

## Installation
The package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add SparseIR
```

**_Note_**: We recommend using the current stable release version 1.8 of Julia while supporting the LTS release 1.6 and newer.
In case you find yourself running an older version, [juliaup](https://github.com/JuliaLang/juliaup) makes installing and maintaining an up-to-date version quite pleasant.

### Manual installation from source

You should almost never have to do this, but it is possible to install SparseIR.jl from source as follows:
```sh
git clone https://github.com/SpM-lab/SparseIR.jl.git
julia -e "import Pkg; Pkg.add(path=\"SparseIR.jl\")"
```
This is *not* recommended, as you will get the unstable development version and no updates.


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




[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://spm-lab.github.io/SparseIR.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://spm-lab.github.io/SparseIR.jl/stable/

[GHA-img]: https://github.com/SpM-lab/SparseIR.jl/workflows/CI/badge.svg
[GHA-url]: https://github.com/SpM-lab/SparseIR.jl/actions?query=workflows/CI

[codecov-img]: https://codecov.io/gh/SpM-lab/SparseIR.jl/branch/main/graph/badge.svg?token=tdMvTruYa4
[codecov-url]: https://codecov.io/gh/SpM-lab/SparseIR.jl

[issues-url]: https://github.com/SpM-lab/SparseIR.jl/issues
