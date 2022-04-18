# TODO: better module docstring
"Intermediate representation (IR) for many-body propagators"
module SparseIR

include("svd.jl")
include("gauss.jl")
include("kernel.jl")
include("sve.jl")
include("poly.jl")
include("basis.jl")
include("sampling.jl")
include("basis_set.jl")

end # module
