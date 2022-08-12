using Test
using SparseIR
using Random
using MultiFloats

include("_conftest.jl")
include("_util.jl")

@testset verbose=true "SparseIR.jl" begin
    include("freq.jl")
    include("gauss.jl")
    include("kernel.jl")
    include("poly.jl")
    include("sve.jl")
    include("basis_set.jl")
    include("sampling.jl")
    include("augment.jl")
    include("composite.jl")
    include("spr.jl")
    include("_linalg.jl")
end

nothing
