# TODO: better module docstring
"Intermediate representation (IR) for many-body propagators"
module SparseIR

include("util.jl")
include("svd.jl")
include("gauss.jl")
include("kernel.jl")
include("sve.jl")
include("poly.jl")
include("basis.jl")
include("augment.jl")
include("composite.jl")
include("sampling.jl")
include("spr.jl")
include("basis_set.jl")

end # module
