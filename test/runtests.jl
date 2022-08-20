using Test
using SparseIR
using Random
using MultiFloats

include("_conftest.jl")

@testset verbose=true "SparseIR.jl" begin
    include("freq.jl")
    include("gauss.jl")
    include("kernel.jl")
    include("poly.jl")
    include("sve.jl")
    include("svd.jl")
    include("basis_set.jl")
    include("sampling.jl")
    include("augment.jl")
    include("spr.jl")
    include("_linalg.jl")
    include("_roots.jl")
    include("_specfuncs.jl")
    include("scipost_sample_code.jl")
end

nothing # otherwise we get messy output from the testset printed in the REPL
