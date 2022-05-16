using Test
using SparseIR

include("conftest.jl")
include("_util.jl")

@testset verbose = true "SparseIR.jl" begin
    include("gauss.jl")
    include("kernel.jl")
    include("poly.jl")
    include("sve.jl")
    include("basis_set.jl")
    include("sampling.jl")
    include("matsubara.jl")
    include("augment.jl")
    include("bessels.jl")
    include("composite.jl")
    include("spr.jl")
    include("_linalg.jl")
    # Works only if Mathematica and MathLink.jl are available.
    # include("_bessels.jl")
end

nothing
