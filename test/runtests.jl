using Test
using Aqua

include("_conftest.jl")

Aqua.test_all(SparseIR; ambiguities=false)

@testset verbose=true "SparseIR.jl" begin
    include("freq.jl")
    include("gauss.jl")
    include("kernel.jl")
    include("poly.jl")
    include("sve.jl")
    include("svd.jl")
    include("basis.jl")
    include("sampling.jl")
    include("augment.jl")
    include("dlr.jl")
    include("_linalg.jl")
    include("_roots.jl")
    include("_specfuncs.jl")
    include("_multifloat_funcs.jl")
end

nothing # without this we get messy output from the testset printed in the REPL
